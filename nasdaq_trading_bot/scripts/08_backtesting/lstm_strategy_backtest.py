"""
LSTM Strategy Backtesting System
=================================
Backtesting for lstm_deploy.py with full strategy system support.

Usage:
    python lstm_strategy_backtest.py --strategy conservative_long --days 7
    python lstm_strategy_backtest.py --list-strategies
    python lstm_strategy_backtest.py --strategy aggressive_long --days 30 --plot
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import importlib.util
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml
import pytz
import joblib
import yfinance as yf

import torch
from torch import nn

import matplotlib.pyplot as plt

# -----------------------------
# Paths / Imports
# -----------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DEPLOYMENT_DIR = os.path.join(PROJECT_ROOT, "scripts", "07_deployment")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DEPLOYMENT_DIR)

# Import FeatureBuilder
FEATURES_PY_PATH = os.path.join(PROJECT_ROOT, "scripts", "03_pre_split_prep", "features.py")
spec = importlib.util.spec_from_file_location("features_module", FEATURES_PY_PATH)
features_module = importlib.util.module_from_spec(spec) if spec else None
if spec and spec.loader:
    spec.loader.exec_module(features_module)
else:
    raise RuntimeError(f"Could not load features.py from {FEATURES_PY_PATH}")
FeatureBuilder = getattr(features_module, "FeatureBuilder")

# Import strategy system from deployment
from strategies.strategy_config import StrategyConfig, load_all_strategies, list_available_strategies
from strategies.strategies import LongOnlyMomentumStrategy, ShortOnlyStrategy, CFDLeveragedStrategy

# Import NewsFeatureProvider
try:
    from news_features import NewsFeatureProvider
    NEWS_AVAILABLE = True
except ImportError:
    NEWS_AVAILABLE = False

# Paths
CONF_DIR = os.path.join(PROJECT_ROOT, "conf")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "lstm")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
SCALER_X_PATH = os.path.join(DATA_DIR, "scaler_X.joblib")
SCALER_Y_PATH = os.path.join(DATA_DIR, "scaler_y.joblib")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "features_clean.txt")

# Load configs
with open(os.path.join(CONF_DIR, "params.yaml"), "r") as f:
    params = yaml.safe_load(f)

# LSTM params (must match training)
SEQUENCE_LENGTH = 50
INPUT_SIZE = 14
HIDDEN_SIZE = 384
NUM_LAYERS = 2
OUTPUT_SIZE = 5
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EASTERN = pytz.timezone("US/Eastern")


# -----------------------------
# Model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        bidirectional: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, c_n) = self.lstm(x)
        last_layer_h = h_n[-self.num_directions :, :, :]
        last_layer_h = last_layer_h.transpose(0, 1).reshape(x.size(0), -1)
        return self.fc(last_layer_h)


# -----------------------------
# Mock Broker for Backtesting
# -----------------------------
@dataclass
class BacktestPosition:
    symbol: str
    qty: int
    entry_price: float
    entry_time: datetime
    side: str  # 'long' or 'short'
    stop_loss: float
    take_profit: float


@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: datetime
    symbol: str
    qty: int
    entry_price: float
    exit_price: float
    side: str
    pnl: float
    pnl_pct: float
    exit_reason: str


class MockBroker:
    """Mock broker for backtesting - simulates Alpaca/OANDA API"""
    
    def __init__(self, initial_equity: float = 100_000.0):
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.cash = initial_equity
        self.positions: Dict[str, BacktestPosition] = {}
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict] = []
        self._last_fill_times: Dict[Tuple[str, str], datetime] = {}
    
    def get_account_info(self) -> dict:
        return {
            "equity": self.equity,
            "cash": self.cash,
            "balance": self.cash,
        }
    
    def get_positions(self) -> List[dict]:
        return [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "side": p.side,
                "avg_entry_price": p.entry_price,
                "current_price": p.entry_price,  # Updated externally
            }
            for p in self.positions.values()
        ]
    
    def get_position(self, symbol: str) -> Optional[dict]:
        p = self.positions.get(symbol)
        if not p:
            return None
        return {
            "symbol": p.symbol,
            "qty": p.qty,
            "side": p.side,
            "avg_entry_price": p.entry_price,
            "current_price": p.entry_price,
        }
    
    def get_last_fill_time(self, symbol: str, side: str) -> Optional[datetime]:
        return self._last_fill_times.get((symbol, side))
    
    def open_position(
        self, 
        symbol: str, 
        qty: int, 
        price: float, 
        time: datetime,
        side: str = "long",
        stop_loss: float = 0.0,
        take_profit: float = 0.0
    ) -> bool:
        if symbol in self.positions:
            return False
        
        cost = qty * price
        if cost > self.cash:
            return False
        
        self.positions[symbol] = BacktestPosition(
            symbol=symbol,
            qty=qty,
            entry_price=price,
            entry_time=time,
            side=side,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.cash -= cost
        self._last_fill_times[(symbol, "buy" if side == "long" else "sell")] = time
        return True
    
    def close_position(
        self, 
        symbol: str, 
        price: float, 
        time: datetime, 
        reason: str = "Manual"
    ) -> bool:
        p = self.positions.get(symbol)
        if not p:
            return False
        
        # Calculate PnL
        if p.side == "long":
            pnl = (price - p.entry_price) * p.qty
        else:
            pnl = (p.entry_price - price) * p.qty
        
        pnl_pct = pnl / (p.entry_price * p.qty)
        
        # Record trade
        self.trades.append(BacktestTrade(
            entry_time=p.entry_time,
            exit_time=time,
            symbol=p.symbol,
            qty=p.qty,
            entry_price=p.entry_price,
            exit_price=price,
            side=p.side,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        ))
        
        # Update cash
        self.cash += p.qty * price
        del self.positions[symbol]
        return True
    
    def update_equity(self, current_prices: Dict[str, float], time: datetime):
        """Update equity based on current prices"""
        positions_value = sum(
            p.qty * current_prices.get(p.symbol, p.entry_price)
            for p in self.positions.values()
        )
        self.equity = self.cash + positions_value
        self.equity_curve.append({
            "time": time,
            "equity": self.equity,
            "positions": len(self.positions),
        })
    
    def check_stops(self, symbol: str, high: float, low: float, time: datetime) -> Optional[str]:
        """Check if stop loss or take profit was hit"""
        p = self.positions.get(symbol)
        if not p:
            return None
        
        if p.side == "long":
            if low <= p.stop_loss:
                self.close_position(symbol, p.stop_loss, time, "StopLoss")
                return "StopLoss"
            if high >= p.take_profit:
                self.close_position(symbol, p.take_profit, time, "TakeProfit")
                return "TakeProfit"
        else:  # short
            if high >= p.stop_loss:
                self.close_position(symbol, p.stop_loss, time, "StopLoss")
                return "StopLoss"
            if low <= p.take_profit:
                self.close_position(symbol, p.take_profit, time, "TakeProfit")
                return "TakeProfit"
        
        return None


# -----------------------------
# Data Loading & Features
# -----------------------------
def load_feature_list(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing feature list: {path}")
    feats = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                feats.append(s)
    return feats


def download_market_data(ticker: str, days: int = 7) -> pd.DataFrame:
    """Download market data from yfinance"""
    print(f"[DATA] Downloading {days}d of 1m for {ticker} via yfinance...")
    df = yf.download(
        ticker, 
        period=f"{days}d", 
        interval="1m", 
        auto_adjust=True, 
        prepost=False, 
        progress=False
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Build all features from raw OHLCV data"""
    df = df_raw.copy()
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["timestamp"] = df.index
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

    ema_periods = params["DATA_PREP"]["EMA_PERIODS"]
    slope_periods = params["DATA_PREP"]["SLOPE_PERIODS"]

    builder = FeatureBuilder(
        df=df,
        ema_windows=ema_periods,
        return_windows=slope_periods,
        price_col="vwap",
        timestamp_col="timestamp",
    )
    df_feat = builder.build_features_before_split()
    
    if "avg_volume_per_trade" not in df_feat.columns:
        df_feat["avg_volume_per_trade"] = df_feat["volume"] / 100.0

    pd.set_option('future.no_silent_downcasting', True)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()

    if df_feat.empty:
        raise RuntimeError("All features NaN after rolling windows.")

    return df_feat


def add_news_features(df_feat: pd.DataFrame, use_news: bool = True) -> pd.DataFrame:
    """Add news sentiment features (or neutral if unavailable)"""
    if use_news and NEWS_AVAILABLE:
        try:
            news_provider = NewsFeatureProvider(decay_lambda=0.001, cache_minutes=5)
            df_news = news_provider.fetch_news_df_once(tickers=["QQQ"])
            
            if not df_news.empty:
                df_news = df_news.sort_values("timestamp")
                df_news_idx = df_news.set_index("timestamp").sort_index()
                
                merged = pd.merge_asof(
                    df_feat.sort_index(), 
                    df_news_idx[["sentiment_score"]], 
                    left_index=True, 
                    right_index=True, 
                    direction='backward'
                )
                merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
                
                df_news_idx["pts"] = df_news_idx.index
                merged_ts = pd.merge_asof(
                    df_feat.sort_index(),
                    df_news_idx[["pts"]],
                    left_index=True,
                    right_index=True,
                    direction='backward'
                )
                
                age_s = (merged_ts.index - merged_ts["pts"]).dt.total_seconds() / 60.0
                age_s = age_s.fillna(999999.0)
                
                sent_s = merged["sentiment_score"]
                eff_s = sent_s * np.exp(-0.001 * age_s)
                
                df_feat = df_feat.copy()
                df_feat["last_news_sentiment"] = sent_s.values
                df_feat["news_age_minutes"] = age_s.values
                df_feat["effective_sentiment_t"] = eff_s.values
                
                print("[NEWS] Historical news aligned successfully")
                return df_feat
        except Exception as e:
            print(f"[NEWS] Failed to load: {e}")
    
    # Use neutral news features
    df_feat = df_feat.copy()
    df_feat["last_news_sentiment"] = 0.0
    df_feat["news_age_minutes"] = 0.0
    df_feat["effective_sentiment_t"] = 0.0
    return df_feat


def load_lstm_model() -> Tuple[LSTMModel, object, object]:
    """Load trained LSTM model and scalers"""
    model_path = os.path.join(MODELS_DIR, "best_lstm_model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(SCALER_Y_PATH):
        raise FileNotFoundError(f"Scaler Y not found: {SCALER_Y_PATH}")
    if not os.path.exists(SCALER_X_PATH):
        raise FileNotFoundError(f"Scaler X not found: {SCALER_X_PATH}")

    scaler_y = joblib.load(SCALER_Y_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)

    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        bidirectional=False,
        dropout=DROPOUT,
    ).to(DEVICE)

    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler_y, scaler_X


# -----------------------------
# Strategy Factory
# -----------------------------
def get_strategy_instance(config: StrategyConfig):
    """Get the appropriate strategy class instance"""
    mapping = {
        "long_only": LongOnlyMomentumStrategy,
        "short_only": ShortOnlyStrategy,
        "cfd_leveraged": CFDLeveragedStrategy,
        "long_short": CFDLeveragedStrategy,
    }
    strategy_class = mapping.get(config.strategy_type, LongOnlyMomentumStrategy)
    return strategy_class(config)


# -----------------------------
# Backtest Engine
# -----------------------------
class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, strategy_config: StrategyConfig, initial_equity: float = 100_000.0):
        self.config = strategy_config
        self.strategy = get_strategy_instance(strategy_config)
        self.broker = MockBroker(initial_equity)
        self.ticker = strategy_config.ticker if strategy_config.ticker != "NAS100_USD" else "QQQ"
        
    def run(self, days: int = 7, use_news: bool = True, verbose: bool = False) -> Dict:
        """Run the backtest"""
        print("=" * 70)
        print(f"LSTM Strategy Backtest")
        print(f"Strategy: {self.config.name} | Type: {self.config.strategy_type}")
        print(f"Ticker: {self.ticker} | Leverage: {self.config.leverage}x")
        print("=" * 70)
        
        # Load model
        model, scaler_y, scaler_X = load_lstm_model()
        feature_list = load_feature_list(FEATURE_LIST_PATH)
        
        if len(feature_list) != INPUT_SIZE:
            raise ValueError(f"Feature mismatch: {len(feature_list)} vs {INPUT_SIZE}")
        
        # Download data
        df_raw = download_market_data(self.ticker, days)
        if df_raw.empty:
            raise RuntimeError("No market data available")
        
        print(f"[DATA] Downloaded {len(df_raw)} bars from {df_raw.index[0]} to {df_raw.index[-1]}")
        
        # Build features
        df_feat = build_features(df_raw)
        df_feat = add_news_features(df_feat, use_news)
        
        # Build feature matrix
        X_list = []
        for feat in feature_list:
            if feat in df_feat.columns:
                X_list.append(df_feat[feat].values.astype(np.float32))
            else:
                print(f"[WARN] Missing feature '{feat}', filling with 0")
                X_list.append(np.zeros(len(df_feat), dtype=np.float32))
        
        X_raw = np.column_stack(X_list).astype(np.float32)
        X_df = pd.DataFrame(X_raw, columns=feature_list)
        X_scaled = scaler_X.transform(X_df)
        
        # Get prices aligned with features
        close_prices = df_raw["Close"].reindex(df_feat.index).astype(float)
        open_prices = df_raw["Open"].reindex(df_feat.index).astype(float)
        high_prices = df_raw["High"].reindex(df_feat.index).astype(float)
        low_prices = df_raw["Low"].reindex(df_feat.index).astype(float)
        
        print(f"[BACKTEST] Running on {len(df_feat)} bars...")
        
        # Initialize signal statistics
        self._signal_stats = {
            'max': float('-inf'),
            'min': float('inf'),
            'sum': 0.0,
            'count': 0,
            'above_threshold': 0,
            'r3_positive': 0,
        }
        
        # Main simulation loop
        for i in range(SEQUENCE_LENGTH, len(df_feat) - 1):
            # 1. Get sequence and predict
            x_seq = X_scaled[i - SEQUENCE_LENGTH + 1 : i + 1]
            
            with torch.no_grad():
                x_tensor = torch.from_numpy(x_seq).float().unsqueeze(0).to(DEVICE)
                pred_scaled = model(x_tensor).cpu().numpy()[0]
            
            pred = scaler_y.inverse_transform([pred_scaled])[0]
            pred = pred / 100.0  # Convert to decimal
            
            prediction_dict = {
                '1m': pred[0],
                '3m': pred[1],
                '5m': pred[2],
                '10m': pred[3],
                '15m': pred[4]
            }
            
            # Current bar info
            current_time = df_feat.index[i]
            next_time = df_feat.index[i + 1]
            current_price = float(close_prices.iloc[i])
            next_open = float(open_prices.iloc[i + 1])
            next_high = float(high_prices.iloc[i + 1])
            next_low = float(low_prices.iloc[i + 1])
            
            # Calculate volatility (annualized)
            recent_returns = close_prices.iloc[max(0, i-20):i+1].pct_change().dropna()
            volatility = float(recent_returns.std() * np.sqrt(252 * 390)) if len(recent_returns) > 1 else 0.01
            
            market_data = {
                'current_price': current_price,
                'volatility': volatility,
                'volume': float(df_raw['Volume'].iloc[i]) if 'Volume' in df_raw.columns else 0
            }
            
            # 2. Calculate signal using strategy
            signal, metadata = self.strategy.calc_signal(prediction_dict, market_data)
            
            # Update signal statistics
            r3 = metadata.get('r3', 0)
            self._signal_stats['max'] = max(self._signal_stats['max'], signal)
            self._signal_stats['min'] = min(self._signal_stats['min'], signal)
            self._signal_stats['sum'] += signal
            self._signal_stats['count'] += 1
            if signal > self.config.entry_threshold:
                self._signal_stats['above_threshold'] += 1
            if r3 > 0:
                self._signal_stats['r3_positive'] += 1
            
            # Debug output - show signals periodically
            bar_num = i - SEQUENCE_LENGTH
            if verbose and bar_num % 50 == 0:
                r3 = metadata.get('r3', 0)
                print(f"[DEBUG] Bar {bar_num}: signal={signal:.6f} r3={r3:.6f} threshold={self.config.entry_threshold:.6f} | {current_time}")
            
            # 3. Check for stop/take profit hits on next bar
            stop_result = self.broker.check_stops(self.ticker, next_high, next_low, next_time)
            
            # 4. Trading logic (if position wasn't closed by stops)
            if self.ticker not in self.broker.positions:
                # No position - check entry
                can_enter, reason = self.strategy.can_enter(
                    self.ticker, signal, metadata, self.broker
                )
                
                # Track all signals above threshold for debugging
                if verbose and signal > self.config.entry_threshold * 0.5:
                    r3 = metadata.get('r3', 0)
                    print(f"[SIGNAL] signal={signal:.6f} r3={r3:.6f} | can_enter={can_enter} | {reason}")
                
                if can_enter:
                    acct = self.broker.get_account_info()
                    equity = float(acct.get("equity", 0))
                    qty = self.strategy.calculate_position_size(
                        self.ticker, signal, equity, next_open
                    )
                    
                    if qty > 0:
                        order_params = self.strategy.get_order_params(
                            self.ticker, qty, "buy" if signal > 0 else "sell", next_open
                        )
                        
                        side = "long" if signal > 0 else "short"
                        sl = float(order_params['stop_loss']['stop_price'])
                        tp = float(order_params['take_profit']['limit_price'])
                        
                        self.broker.open_position(
                            self.ticker, qty, next_open, next_time, side, sl, tp
                        )
                        
                        if verbose:
                            print(f"[ENTRY] {next_time} {side.upper()} {qty} @ ${next_open:.2f} | signal={signal:.5f}")
            else:
                # Have position - check exit
                position = self.broker.get_position(self.ticker)
                should_exit, reason = self.strategy.should_exit(
                    self.ticker, position, signal, metadata, self.broker
                )
                
                if should_exit:
                    self.broker.close_position(self.ticker, next_open, next_time, reason)
                    if verbose:
                        print(f"[EXIT] {next_time} @ ${next_open:.2f} | {reason}")
            
            # Update equity
            self.broker.update_equity({self.ticker: next_open}, next_time)
        
        # Close any remaining position
        if self.ticker in self.broker.positions:
            final_price = float(close_prices.iloc[-1])
            self.broker.close_position(self.ticker, final_price, df_feat.index[-1], "EndOfBacktest")
        
        # Print signal statistics
        if hasattr(self, '_signal_stats'):
            stats = self._signal_stats
            print(f"\n[SIGNAL STATS]")
            print(f"  Max signal:  {stats['max']:.6f}")
            print(f"  Min signal:  {stats['min']:.6f}")
            print(f"  Avg signal:  {stats['sum']/stats['count']:.6f}")
            print(f"  Entry threshold: {self.config.entry_threshold:.6f}")
            print(f"  Signals above threshold: {stats['above_threshold']}")
            print(f"  Signals with r3 > 0: {stats['r3_positive']}")
        
        return self._calculate_results()
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest performance metrics"""
        trades = self.broker.trades
        equity_curve = pd.DataFrame(self.broker.equity_curve)
        
        results = {
            "strategy_name": self.config.name,
            "strategy_type": self.config.strategy_type,
            "ticker": self.ticker,
            "leverage": self.config.leverage,
            "initial_equity": self.broker.initial_equity,
            "final_equity": self.broker.equity,
            "total_return": (self.broker.equity - self.broker.initial_equity) / self.broker.initial_equity,
            "total_trades": len(trades),
        }
        
        if trades:
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl <= 0]
            
            results["winning_trades"] = len(wins)
            results["losing_trades"] = len(losses)
            results["win_rate"] = len(wins) / len(trades) if trades else 0
            results["total_pnl"] = sum(t.pnl for t in trades)
            results["avg_pnl"] = results["total_pnl"] / len(trades)
            results["avg_win"] = sum(t.pnl for t in wins) / len(wins) if wins else 0
            results["avg_loss"] = sum(t.pnl for t in losses) / len(losses) if losses else 0
            results["profit_factor"] = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float('inf')
            
            # Exit reasons
            exit_reasons = {}
            for t in trades:
                exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
            results["exit_reasons"] = exit_reasons
            
            # Calculate max drawdown
            if not equity_curve.empty:
                equity_series = equity_curve["equity"]
                running_max = equity_series.cummax()
                drawdown = (equity_series - running_max) / running_max
                results["max_drawdown"] = float(drawdown.min())
            else:
                results["max_drawdown"] = 0
        else:
            results["winning_trades"] = 0
            results["losing_trades"] = 0
            results["win_rate"] = 0
            results["total_pnl"] = 0
            results["avg_pnl"] = 0
            results["max_drawdown"] = 0
        
        results["equity_curve"] = equity_curve
        results["trades"] = trades
        
        return results


def print_results(results: Dict):
    """Print backtest results"""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Strategy:       {results['strategy_name']}")
    print(f"Type:           {results['strategy_type']}")
    print(f"Ticker:         {results['ticker']}")
    print(f"Leverage:       {results['leverage']}x")
    print("-" * 70)
    print(f"Initial Equity: ${results['initial_equity']:,.2f}")
    print(f"Final Equity:   ${results['final_equity']:,.2f}")
    print(f"Total Return:   {results['total_return']*100:.2f}%")
    print(f"Max Drawdown:   {results.get('max_drawdown', 0)*100:.2f}%")
    print("-" * 70)
    print(f"Total Trades:   {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Losing Trades:  {results['losing_trades']}")
    print(f"Win Rate:       {results['win_rate']*100:.1f}%")
    print("-" * 70)
    print(f"Total PnL:      ${results.get('total_pnl', 0):,.2f}")
    print(f"Avg Trade PnL:  ${results.get('avg_pnl', 0):,.2f}")
    print(f"Avg Win:        ${results.get('avg_win', 0):,.2f}")
    print(f"Avg Loss:       ${results.get('avg_loss', 0):,.2f}")
    print(f"Profit Factor:  {results.get('profit_factor', 0):.2f}")
    print("-" * 70)
    
    if "exit_reasons" in results and results["exit_reasons"]:
        print("Exit Reasons:")
        for reason, count in sorted(results["exit_reasons"].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    
    print("=" * 70)
    
    # Print last 10 trades
    trades = results.get("trades", [])
    if trades:
        print("\nLast 10 Trades:")
        print("-" * 90)
        print(f"{'Entry Time':<20} {'Exit Time':<20} {'Side':<6} {'Qty':<6} {'PnL':>10} {'Reason':<15}")
        print("-" * 90)
        for t in trades[-10:]:
            entry_str = t.entry_time.strftime("%Y-%m-%d %H:%M")
            exit_str = t.exit_time.strftime("%Y-%m-%d %H:%M")
            print(f"{entry_str:<20} {exit_str:<20} {t.side:<6} {t.qty:<6} ${t.pnl:>9,.2f} {t.exit_reason:<15}")


def plot_results(results: Dict, save_path: Optional[str] = None):
    """Plot equity curve and trade distribution"""
    equity_curve = results.get("equity_curve")
    trades = results.get("trades", [])
    
    if equity_curve is None or equity_curve.empty:
        print("[WARN] No equity curve data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Backtest: {results['strategy_name']} ({results['ticker']})", fontsize=14, fontweight='bold')
    
    # 1. Equity curve
    ax1 = axes[0, 0]
    ax1.plot(equity_curve["time"], equity_curve["equity"], label="Equity", color="blue")
    ax1.axhline(y=results["initial_equity"], color="gray", linestyle="--", label="Initial")
    ax1.set_title("Equity Curve")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Equity ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = axes[0, 1]
    equity_series = equity_curve["equity"]
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max * 100
    ax2.fill_between(equity_curve["time"], drawdown, 0, alpha=0.3, color="red")
    ax2.plot(equity_curve["time"], drawdown, color="red", linewidth=0.5)
    ax2.set_title("Drawdown (%)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, alpha=0.3)
    
    # 3. Trade PnL distribution
    ax3 = axes[1, 0]
    if trades:
        pnls = [t.pnl for t in trades]
        colors = ["green" if p > 0 else "red" for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax3.axhline(y=0, color="black", linewidth=0.5)
        ax3.set_title("Trade PnL")
        ax3.set_xlabel("Trade #")
        ax3.set_ylabel("PnL ($)")
    else:
        ax3.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax3.transAxes)
    ax3.grid(True, alpha=0.3)
    
    # 4. Exit reasons pie chart
    ax4 = axes[1, 1]
    if "exit_reasons" in results and results["exit_reasons"]:
        labels = list(results["exit_reasons"].keys())
        sizes = list(results["exit_reasons"].values())
        ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax4.set_title("Exit Reasons")
    else:
        ax4.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax4.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved to {save_path}")
    else:
        plt.show()


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LSTM Strategy Backtesting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lstm_strategy_backtest.py --list-strategies
  python lstm_strategy_backtest.py --strategy conservative_long --days 7
  python lstm_strategy_backtest.py --strategy aggressive_long --days 30 --plot
        """
    )
    parser.add_argument("--list-strategies", action="store_true", help="List all available strategies")
    parser.add_argument("--strategy", "-s", type=str, help="Strategy ID to backtest")
    parser.add_argument("--days", "-d", type=int, default=7, help="Number of days to backtest (default: 7)")
    parser.add_argument("--initial-equity", type=float, default=100000.0, help="Initial equity (default: 100000)")
    parser.add_argument("--no-news", action="store_true", help="Disable news features")
    parser.add_argument("--plot", action="store_true", help="Show equity curve plot")
    parser.add_argument("--save-plot", type=str, help="Save plot to file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # List strategies
    if args.list_strategies:
        list_available_strategies()
        return
    
    # Require strategy
    if not args.strategy:
        print("[ERROR] No strategy specified. Use --strategy <id> or --list-strategies")
        list_available_strategies()
        return
    
    # Load strategy
    all_strategies = load_all_strategies()
    if args.strategy not in all_strategies:
        print(f"[ERROR] Unknown strategy: {args.strategy}")
        list_available_strategies()
        return
    
    config = all_strategies[args.strategy]
    
    # Run backtest
    engine = BacktestEngine(config, initial_equity=args.initial_equity)
    results = engine.run(days=args.days, use_news=not args.no_news, verbose=args.verbose)
    
    # Print results
    print_results(results)
    
    # Plot
    if args.plot or args.save_plot:
        save_path = args.save_plot
        if not save_path and args.plot:
            os.makedirs(IMAGES_DIR, exist_ok=True)
            save_path = os.path.join(IMAGES_DIR, f"backtest_{args.strategy}_{args.days}d.png")
        plot_results(results, save_path)


if __name__ == "__main__":
    main()
