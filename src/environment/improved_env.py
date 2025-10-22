"""
Improved Multi-Asset Deep Hedging Environment with better reward shaping
Key improvements:
1. Sharpe ratio based rewards
2. Better position sizing
3. Drawdown penalties
4. Improved normalization
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ImprovedMultiAssetEnv(gym.Env):
    """
    Improved environment with Sharpe ratio optimization and better risk management.
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        spot_data: pd.DataFrame,
        option_chain: pd.DataFrame,
        asset_universe: List[Dict[str, Any]],
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.0005,  # Reduced from 0.001
        max_position_size: float = 0.2,  # Max 20% per asset
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        episode_length: int = 200,
        lookback_window: int = 20  # Increased from 10
    ):
        super().__init__()
        
        self.spot_data = spot_data.reset_index(drop=True)
        self.option_chain = option_chain.reset_index(drop=True)
        self.asset_universe = asset_universe
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate
        self.episode_length = episode_length
        self.lookback_window = lookback_window
        
        assert len(self.asset_universe) == 20, f"Expected 20 options, got {len(self.asset_universe)}"
        
        # Action space: 21 positions (normalized to [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(21,),
            dtype=np.float32
        )
        
        # Observation space: 107 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(107,),
            dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.current_date_idx = 0
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = np.zeros(21)
        self.done = False
        
        # Track returns for Sharpe calculation
        self.episode_returns = []
        self.peak_value = initial_capital
        self.total_trades = 0
        
        logger.info(f"Initialized ImprovedMultiAssetEnv with {len(spot_data)} days of data")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Random start within valid range
        max_start = len(self.spot_data) - self.episode_length - self.lookback_window
        if max_start > self.lookback_window:
            self.current_date_idx = np.random.randint(self.lookback_window, max_start)
        else:
            self.current_date_idx = self.lookback_window
        
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = np.zeros(21)
        self.done = False
        self.episode_returns = []
        self.peak_value = self.initial_capital
        self.total_trades = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Get current prices
        current_prices = self._get_current_prices()
        prev_value = self._calculate_portfolio_value(current_prices)
        
        # Scale actions to actual position sizes (% of portfolio)
        # Action in [-1, 1] maps to [-max_position_size, +max_position_size]
        scaled_actions = action * self.max_position_size
        
        # Calculate target dollar amounts for each position
        target_dollar_amounts = scaled_actions * prev_value
        
        # Calculate number of contracts/shares
        target_positions = target_dollar_amounts / (current_prices + 1e-8)
        
        # Calculate position changes
        position_changes = target_positions - self.positions
        trade_value = np.sum(np.abs(position_changes * current_prices))
        trade_costs = trade_value * self.transaction_cost
        
        # Update positions and cash
        self.positions = target_positions
        self.cash -= trade_costs
        self.total_trades += np.sum(np.abs(position_changes) > 1e-6)
        
        # Move to next timestep
        self.current_step += 1
        self.current_date_idx += 1
        
        # Check termination
        if self.current_step >= self.episode_length or self.current_date_idx >= len(self.spot_data):
            self.done = True
            terminated = True
        else:
            terminated = False
        
        # Get new prices and portfolio value
        next_prices = self._get_current_prices()
        new_value = self._calculate_portfolio_value(next_prices)
        
        # Calculate step return
        step_return = (new_value - prev_value) / max(prev_value, 1.0)
        self.episode_returns.append(step_return)
        
        # Update peak for drawdown calculation
        if new_value > self.peak_value:
            self.peak_value = new_value
        
        # Calculate reward
        reward = self._calculate_reward(
            prev_value, new_value, trade_costs, step_return
        )
        
        self.portfolio_value = new_value
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _calculate_reward(
        self,
        prev_value: float,
        curr_value: float,
        trade_costs: float,
        step_return: float
    ) -> float:
        """
        Improved reward function focusing on risk-adjusted returns.
        """
        # 1. Return component (main signal)
        return_reward = step_return * 100.0  # Scale up for better learning
        
        # 2. Sharpe ratio component (risk-adjusted performance)
        if len(self.episode_returns) > 5:
            returns_array = np.array(self.episode_returns[-20:])  # Last 20 steps
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array) + 1e-8
            sharpe_ratio = (mean_return - self.risk_free_rate/252) / std_return
            sharpe_reward = sharpe_ratio * 10.0  # Scale for impact
        else:
            sharpe_reward = 0.0
        
        # 3. Drawdown penalty
        drawdown = (self.peak_value - curr_value) / max(self.peak_value, 1.0)
        drawdown_penalty = -drawdown * 50.0  # Penalize drawdowns
        
        # 4. Transaction cost penalty (scaled)
        cost_penalty = -(trade_costs / max(prev_value, 1.0)) * 100.0
        
        # 5. Position diversity bonus (encourage diversification)
        active_positions = np.sum(np.abs(self.positions) > 1e-6)
        diversity_bonus = (active_positions / 21.0) * 5.0  # Reward using more instruments
        
        # 6. Extreme position penalty (discourage concentration)
        position_values = np.abs(self.positions * self._get_current_prices())
        max_position_pct = np.max(position_values) / max(curr_value, 1.0)
        if max_position_pct > 0.3:  # More than 30% in single position
            concentration_penalty = -(max_position_pct - 0.3) * 20.0
        else:
            concentration_penalty = 0.0
        
        # Combined reward
        reward = (
            return_reward +
            sharpe_reward * 0.3 +  # 30% weight on Sharpe
            drawdown_penalty * 0.2 +  # 20% weight on drawdown
            cost_penalty +
            diversity_bonus * 0.1 +  # 10% weight on diversity
            concentration_penalty
        )
        
        # Clip to prevent extreme values
        reward = np.clip(reward, -100.0, 100.0)
        reward = np.nan_to_num(reward, nan=0.0, posinf=100.0, neginf=-100.0)
        
        return float(reward)
    
    def _get_observation(self) -> np.ndarray:
        """Enhanced observation with better feature engineering."""
        if self.current_date_idx >= len(self.spot_data):
            return np.zeros(107, dtype=np.float32)
        
        spot_row = self.spot_data.iloc[self.current_date_idx]
        
        # Market features
        spot_price = float(spot_row['close'])
        spot_return = float(spot_row.get('return', 0.0))
        realized_vol = float(spot_row.get('realized_vol_10d', 0.2))
        
        # Momentum features
        momentum_5d = float(spot_row.get('momentum_5d', 0.0))
        momentum_20d = float(spot_row.get('momentum_20d', 0.0))
        
        # Portfolio features
        portfolio_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio (trailing)
        if len(self.episode_returns) > 5:
            returns_array = np.array(self.episode_returns[-20:])
            sharpe = (np.mean(returns_array) - self.risk_free_rate/252) / (np.std(returns_array) + 1e-8)
        else:
            sharpe = 0.0
        
        # Drawdown
        drawdown = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1.0)
        
        base_features = np.array([
            np.log(spot_price / 4000.0),  # Log-normalized price
            spot_return * 100.0,  # Percentage returns
            realized_vol,
            portfolio_return,
            sharpe,
            drawdown,
            self.current_step / self.episode_length  # Progress indicator
        ], dtype=np.float32)
        
        # Option features (same as before but with better handling)
        option_features = []
        current_date = spot_row['date']
        
        for asset_config in self.asset_universe:
            option_data = self._get_option_data(
                current_date,
                asset_config['strike_offset'],
                asset_config['expiry_days'],
                asset_config['type']
            )
            
            if option_data is not None:
                bid = float(option_data.get('bid', 0.0))
                ask = float(option_data.get('ask', 0.0))
                mid_price = (bid + ask) / 2.0
                spread = ask - bid
                
                iv = float(option_data.get('iv', 0.2))
                moneyness = float(option_data.get('moneyness', 0.0))
                oi = float(option_data.get('open_interest', 0.0))
                
                # Normalized features
                option_features.extend([
                    np.log(mid_price + 1.0) / 10.0,  # Log-normalized price
                    spread / (mid_price + 1.0),  # Relative spread
                    iv,
                    moneyness,
                    np.log(oi + 1.0) / 10.0  # Log-normalized OI
                ])
            else:
                option_features.extend([0.0, 0.0, 0.2, 0.0, 0.0])
        
        option_features = np.array(option_features, dtype=np.float32)
        observation = np.concatenate([base_features, option_features])
        observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return observation
    
    def _get_current_prices(self) -> np.ndarray:
        """Get current prices for all 21 assets."""
        if self.current_date_idx >= len(self.spot_data):
            return np.ones(21)
        
        prices = []
        spot_row = self.spot_data.iloc[self.current_date_idx]
        prices.append(float(spot_row['close']))
        
        current_date = spot_row['date']
        for asset_config in self.asset_universe:
            option_data = self._get_option_data(
                current_date,
                asset_config['strike_offset'],
                asset_config['expiry_days'],
                asset_config['type']
            )
            
            if option_data is not None:
                bid = float(option_data.get('bid', 0.0))
                ask = float(option_data.get('ask', 0.0))
                mid_price = (bid + ask) / 2.0
                prices.append(max(mid_price, 0.01))
            else:
                prices.append(0.01)
        
        return np.array(prices, dtype=np.float32)
    
    def _get_option_data(
        self, date, strike_offset, expiry_days, option_type
    ) -> Optional[Dict[str, Any]]:
        """Retrieve option data for specific contract."""
        mask = (
            (self.option_chain['date'] == date) &
            (self.option_chain['strike_offset'] == strike_offset) &
            (self.option_chain['expiry_days'] == expiry_days) &
            (self.option_chain['type'] == option_type)
        )
        matching = self.option_chain[mask]
        return matching.iloc[0].to_dict() if len(matching) > 0 else None
    
    def _calculate_portfolio_value(self, prices: np.ndarray) -> float:
        """Calculate total portfolio value."""
        holdings_value = np.sum(self.positions * prices)
        return float(self.cash + holdings_value)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        if len(self.episode_returns) > 0:
            cumulative_return = np.prod([1 + r for r in self.episode_returns]) - 1
            sharpe = (np.mean(self.episode_returns) - self.risk_free_rate/252) / (np.std(self.episode_returns) + 1e-8)
        else:
            cumulative_return = 0.0
            sharpe = 0.0
        
        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe,
            'total_trades': self.total_trades,
            'peak_value': self.peak_value,
            'drawdown': (self.peak_value - self.portfolio_value) / max(self.peak_value, 1.0)
        }
    
    def render(self):
        pass
    
    def close(self):
        pass
