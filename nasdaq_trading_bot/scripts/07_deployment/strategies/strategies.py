"""
Example Trading Strategy Implementations
=========================================
Various strategy implementations for different trading styles.
"""

from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta, timezone
from .strategy_config import TradingStrategy, StrategyConfig


class LongOnlyMomentumStrategy(TradingStrategy):
    """Simple long-only momentum strategy"""

    def calc_signal(self, prediction: dict, market_data: dict) -> Tuple[float, dict]:
        """Calculate signal based on predicted returns"""
        r1 = prediction.get('1m', 0.0)
        r3 = prediction.get('3m', 0.0)
        r5 = prediction.get('5m', 0.0)

        # Weighted average of predictions
        signal = 0.3 * r1 + 0.4 * r3 + 0.3 * r5

        metadata = {
            'r1': r1,
            'r3': r3,
            'r5': r5,
            'timestamp': datetime.now(timezone.utc)
        }

        return signal, metadata

    def can_enter(self, symbol: str, signal: float, metadata: dict,
                  broker) -> Tuple[bool, str]:
        """Check if we can enter a long position"""

        # Check signal threshold
        if signal <= self.config.entry_threshold:
            return False, f"Signal too weak: {signal:.6f}"

        # Check short-term prediction is positive
        if metadata['r3'] <= 0:
            return False, "3m prediction negative"

        # Check cooldown
        last_trade = broker.get_last_fill_time(symbol, 'buy')
        if last_trade:
            elapsed = (datetime.now(timezone.utc) - last_trade).total_seconds() / 60
            if elapsed < self.config.cooldown_minutes:
                return False, f"Cooldown: {self.config.cooldown_minutes - elapsed:.1f}m left"

        # Check max positions
        positions = broker.get_positions()
        if len(positions) >= self.config.max_positions:
            return False, f"Max positions reached ({self.config.max_positions})"

        return True, "Entry conditions met"

    def should_exit(self, symbol: str, position: dict, signal: float,
                    metadata: dict, broker) -> Tuple[bool, str]:
        """Check if we should exit the position"""

        entry_time = broker.get_last_fill_time(symbol, 'buy')
        if not entry_time:
            return False, "Cannot determine entry time"

        hold_time = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60

        # Max hold time reached
        if hold_time >= self.config.max_hold_minutes:
            return True, f"Max hold time: {hold_time:.1f}m"

        # Wait for min hold time
        if hold_time < self.config.min_hold_minutes:
            return False, f"Min hold not met: {hold_time:.1f}m"

        # Exit on negative signal
        if signal < self.config.exit_threshold:
            return True, f"Signal turned negative: {signal:.6f}"

        # Exit on negative short-term prediction
        if metadata['r3'] < 0:
            return True, "3m prediction negative"

        return False, "Hold position"

    def calculate_position_size(self, symbol: str, signal: float,
                                account_equity: float, current_price: float) -> int:
        """Calculate position size as % of equity with leverage"""
        target_value = account_equity * self.config.position_size_pct * self.config.leverage
        qty = int(target_value / current_price)
        return max(1, qty)

    def get_order_params(self, symbol: str, qty: int, side: str,
                         current_price: float) -> dict:
        """Get order params for Alpaca bracket order"""
        sl_price = current_price * (1 + self.config.stop_loss_pct)
        tp_price = current_price * (1 + self.config.take_profit_pct)

        return {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'type': 'market',
            'time_in_force': 'day',
            'order_class': 'bracket',
            'take_profit': {'limit_price': f"{tp_price:.2f}"},
            'stop_loss': {'stop_price': f"{sl_price:.2f}"}
        }


class ShortOnlyStrategy(TradingStrategy):
    """Short-only strategy for bearish predictions"""

    def calc_signal(self, prediction: dict, market_data: dict) -> Tuple[float, dict]:
        """Calculate signal - looking for negative predictions"""
        r3 = prediction.get('3m', 0.0)
        r5 = prediction.get('5m', 0.0)
        r10 = prediction.get('10m', 0.0)

        # Negative signal for shorts (flip sign)
        signal = -(0.4 * r3 + 0.35 * r5 + 0.25 * r10)

        metadata = {
            'r3': r3,
            'r5': r5,
            'r10': r10,
            'timestamp': datetime.now(timezone.utc)
        }

        return signal, metadata

    def can_enter(self, symbol: str, signal: float, metadata: dict,
                  broker) -> Tuple[bool, str]:
        """Check if we can enter a short position"""

        if signal <= self.config.entry_threshold:
            return False, f"Signal too weak: {signal:.6f}"

        # Require negative predictions (original values)
        if metadata['r3'] >= 0:
            return False, "3m prediction not negative enough"

        positions = broker.get_positions()
        if len(positions) >= self.config.max_positions:
            return False, "Max positions reached"

        return True, "Short entry conditions met"

    def should_exit(self, symbol: str, position: dict, signal: float,
                    metadata: dict, broker) -> Tuple[bool, str]:
        """Check if we should cover the short"""

        entry_time = broker.get_last_fill_time(symbol, 'sell')
        if not entry_time:
            return False, "Cannot determine entry time"

        hold_time = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60

        if hold_time >= self.config.max_hold_minutes:
            return True, f"Max hold time: {hold_time:.1f}m"

        if hold_time < self.config.min_hold_minutes:
            return False, f"Min hold: {hold_time:.1f}m"

        # Cover if prediction turns positive
        if metadata['r3'] > 0:
            return True, "Prediction turned positive"

        return False, "Hold short"

    def calculate_position_size(self, symbol: str, signal: float,
                                account_equity: float, current_price: float) -> int:
        target_value = account_equity * self.config.position_size_pct
        return max(1, int(target_value / current_price))

    def get_order_params(self, symbol: str, qty: int, side: str,
                         current_price: float) -> dict:
        """Get short order params"""
        # For shorts, stop loss is ABOVE current price
        sl_price = current_price * (1 - self.config.stop_loss_pct)
        tp_price = current_price * (1 - self.config.take_profit_pct)

        return {
            'symbol': symbol,
            'qty': qty,
            'side': 'sell',  # short
            'type': 'market',
            'time_in_force': 'day',
            'order_class': 'bracket',
            'take_profit': {'limit_price': f"{tp_price:.2f}"},
            'stop_loss': {'stop_price': f"{sl_price:.2f}"}
        }


class CFDLeveragedStrategy(TradingStrategy):
    """Leveraged CFD strategy (long/short with leverage)"""

    def calc_signal(self, prediction: dict, market_data: dict) -> Tuple[float, dict]:
        """Calculate directional signal with volatility adjustment"""
        r1 = prediction.get('1m', 0.0)
        r3 = prediction.get('3m', 0.0)
        r5 = prediction.get('5m', 0.0)

        raw_signal = 0.3 * r1 + 0.4 * r3 + 0.3 * r5

        # Adjust for volatility (reduce leverage in high vol)
        volatility = market_data.get('volatility', 0.01)
        vol_adjustment = min(1.0, 0.01 / max(volatility, 0.005))

        signal = raw_signal * vol_adjustment

        metadata = {
            'r1': r1,
            'r3': r3,
            'r5': r5,
            'raw_signal': raw_signal,
            'volatility': volatility,
            'vol_adjustment': vol_adjustment,
            'timestamp': datetime.now(timezone.utc)
        }

        return signal, metadata

    def can_enter(self, symbol: str, signal: float, metadata: dict,
                  broker) -> Tuple[bool, str]:
        """Check entry for leveraged position (long or short)"""

        if abs(signal) <= self.config.entry_threshold:
            return False, f"Signal too weak: {signal:.6f}"

        # Check volatility limits
        vol = metadata['volatility']
        if self.config.min_volatility and vol < self.config.min_volatility:
            return False, f"Volatility too low: {vol:.4f}"
        if self.config.max_volatility and vol > self.config.max_volatility:
            return False, f"Volatility too high: {vol:.4f}"

        positions = broker.get_positions()
        if len(positions) >= self.config.max_positions:
            return False, "Max positions reached"

        side = 'long' if signal > 0 else 'short'
        return True, f"CFD entry ({side}) conditions met"

    def should_exit(self, symbol: str, position: dict, signal: float,
                    metadata: dict, broker) -> Tuple[bool, str]:
        """Exit logic for leveraged positions"""

        side = position.get('side', 'long')
        entry_time = broker.get_last_fill_time(symbol, 'buy' if side == 'long' else 'sell')

        if not entry_time:
            return False, "Cannot determine entry time"

        hold_time = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60

        # Tighter time limits for leveraged trading
        if hold_time >= self.config.max_hold_minutes:
            return True, f"Max hold: {hold_time:.1f}m"

        if hold_time < self.config.min_hold_minutes:
            return False, f"Min hold: {hold_time:.1f}m"

        # Exit if signal reverses
        if side == 'long' and signal < -self.config.exit_threshold:
            return True, "Signal reversed (long -> short)"
        if side == 'short' and signal > self.config.exit_threshold:
            return True, "Signal reversed (short -> long)"

        return False, "Hold leveraged position"

    def calculate_position_size(self, symbol: str, signal: float,
                                account_equity: float, current_price: float) -> int:
        """Calculate size with leverage factor"""
        base_value = account_equity * self.config.position_size_pct
        leveraged_value = base_value * self.config.leverage

        # Reduce size in extreme signals (risk management)
        signal_strength = min(abs(signal), 1.0)
        adjusted_value = leveraged_value * signal_strength

        qty = int(adjusted_value / current_price)
        return max(1, qty)

    def get_order_params(self, symbol: str, qty: int, side: str,
                         current_price: float) -> dict:
        """Order params for CFD broker"""

        if side == 'buy' or side == 'long':
            sl_price = current_price * (1 + self.config.stop_loss_pct)
            tp_price = current_price * (1 + self.config.take_profit_pct)
            order_side = 'buy'
        else:
            sl_price = current_price * (1 - self.config.stop_loss_pct)
            tp_price = current_price * (1 - self.config.take_profit_pct)
            order_side = 'sell'

        params = {
            'symbol': symbol,
            'qty': qty,
            'side': order_side,
            'type': 'market',
            'time_in_force': 'day',
            'order_class': 'bracket',
            'take_profit': {'limit_price': f"{tp_price:.2f}"},
            'stop_loss': {'stop_price': f"{sl_price:.2f}"},
            'leverage': self.config.leverage
        }

        if self.config.use_trailing_stop and self.config.trailing_stop_pct:
            params['trailing_stop'] = {
                'trail_percent': self.config.trailing_stop_pct
            }

        return params