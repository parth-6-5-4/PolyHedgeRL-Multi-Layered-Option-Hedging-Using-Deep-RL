"""
Multi-Asset Deep Hedging Environment for PolyHedgeRL
Gymnasium-compatible RL environment for option portfolio management.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MultiAsset21DeepHedgingEnv(gym.Env):
    """
    Custom Gymnasium environment for multi-asset option hedging.
    
    Supports 21 tradable instruments:
    - 1 spot asset
    - 20 options (combinations of strikes, expiries, and call/put types)
    
    State space: 107-dimensional feature vector
    Action space: 21-dimensional continuous allocation vector
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        spot_data: pd.DataFrame,
        option_chain: pd.DataFrame,
        asset_universe: List[Dict[str, Any]],
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        risk_penalty: float = 0.01,
        episode_length: int = 200,
        lookback_window: int = 10
    ):
        """
        Initialize the trading environment.
        
        Args:
            spot_data: DataFrame with spot price data (columns: date, close, return, realized_vol_10d, etc.)
            option_chain: DataFrame with option data
            asset_universe: List of dicts defining the 20 options
            initial_capital: Starting cash amount
            transaction_cost: Transaction cost as fraction of trade value
            risk_penalty: Penalty coefficient for portfolio variance
            episode_length: Maximum steps per episode
            lookback_window: Number of historical periods for features
        """
        super().__init__()
        
        self.spot_data = spot_data.reset_index(drop=True)
        self.option_chain = option_chain.reset_index(drop=True)
        self.asset_universe = asset_universe
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty
        self.episode_length = episode_length
        self.lookback_window = lookback_window
        
        # Validate asset universe
        assert len(self.asset_universe) == 20, f"Expected 20 options, got {len(self.asset_universe)}"
        
        # Define action space: 21 positions (1 spot + 20 options)
        # Each position is a weight in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(21,),
            dtype=np.float32
        )
        
        # Define observation space: 107-dimensional feature vector
        # 7 base features + 20 options × 5 features each
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
        self.positions = np.zeros(21)  # Current positions
        self.done = False
        
        logger.info(f"Initialized MultiAsset21DeepHedgingEnv with {len(spot_data)} days of data")
        logger.info(f"Action space: {self.action_space.shape}, Observation space: {self.observation_space.shape}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial state observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Reset state variables
        self.current_step = 0
        
        # Start from a random valid position in the data
        max_start_idx = len(self.spot_data) - self.episode_length - self.lookback_window
        if max_start_idx > 0:
            self.current_date_idx = self.np_random.integers(self.lookback_window, max_start_idx)
        else:
            self.current_date_idx = self.lookback_window
        
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = np.zeros(21)
        self.done = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step within the environment.
        
        Args:
            action: Array of 21 position weights
            
        Returns:
            observation: Next state observation
            reward: Reward for this step
            terminated: Whether episode has ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call reset() to start a new episode.")
        
        # Clip and normalize action
        action = np.clip(action, -1.0, 1.0)
        
        # Calculate current asset prices
        current_prices = self._get_current_prices()
        
        # Calculate current portfolio value before action
        current_value = self._calculate_portfolio_value(current_prices)
        
        # Calculate target positions based on action
        target_positions = action * (current_value / (current_prices + 1e-8))
        
        # Execute trades and calculate transaction costs
        position_changes = target_positions - self.positions
        trade_costs = np.sum(np.abs(position_changes * current_prices)) * self.transaction_cost
        
        # Update positions
        self.positions = target_positions
        self.cash -= trade_costs
        
        # Move to next time step
        self.current_step += 1
        self.current_date_idx += 1
        
        # Check if episode is done
        if self.current_step >= self.episode_length or self.current_date_idx >= len(self.spot_data):
            self.done = True
            terminated = True
        else:
            terminated = False
        
        # Calculate new portfolio value after price changes
        next_prices = self._get_current_prices()
        new_value = self._calculate_portfolio_value(next_prices)
        
        # Calculate reward
        reward = self._calculate_reward(current_value, new_value, trade_costs)
        
        # Update portfolio value
        self.portfolio_value = new_value
        
        # Get next observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation vector (107 dimensions).
        
        Returns:
            107-dimensional feature vector
        """
        if self.current_date_idx >= len(self.spot_data):
            # Return zero observation if we're past the data
            return np.zeros(107, dtype=np.float32)
        
        # Base features (7 dimensions)
        spot_row = self.spot_data.iloc[self.current_date_idx]
        
        spot_price = float(spot_row['close'])
        spot_return = float(spot_row.get('return', 0.0))
        realized_vol = float(spot_row.get('realized_vol_10d', 0.2))
        
        # Portfolio features
        portfolio_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        cash_ratio = self.cash / (self.portfolio_value + 1e-8)
        
        # Calculate position concentration (Herfindahl index)
        abs_positions = np.abs(self.positions)
        position_sum = np.sum(abs_positions) + 1e-8
        normalized_positions = abs_positions / position_sum
        concentration = np.sum(normalized_positions ** 2)
        
        # Net exposure (long vs short)
        net_exposure = np.sum(self.positions)
        
        base_features = np.array([
            spot_price / 1000.0,  # Normalize
            spot_return,
            realized_vol,
            portfolio_return,
            cash_ratio,
            concentration,
            net_exposure / 1000.0
        ], dtype=np.float32)
        
        # Option features (100 dimensions = 20 options × 5 features)
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
                # 5 features per option: bid, ask, iv, moneyness, open_interest
                bid = float(option_data.get('bid', 0.0)) / 100.0  # Normalize
                ask = float(option_data.get('ask', 0.0)) / 100.0
                iv = float(option_data.get('iv', 0.2))
                moneyness = float(option_data.get('moneyness', 0.0))
                oi = float(option_data.get('open_interest', 0.0)) / 1000.0  # Normalize
                
                option_features.extend([bid, ask, iv, moneyness, oi])
            else:
                # Missing data - use zeros
                option_features.extend([0.0, 0.0, 0.2, 0.0, 0.0])
        
        option_features = np.array(option_features, dtype=np.float32)
        
        # Combine all features
        observation = np.concatenate([base_features, option_features])
        
        # Replace any NaN or inf values with zeros for stability
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure correct shape
        assert observation.shape == (107,), f"Observation shape mismatch: {observation.shape}"
        
        return observation
    
    def _get_current_prices(self) -> np.ndarray:
        """
        Get current prices for all 21 assets.
        
        Returns:
            Array of 21 prices (1 spot + 20 options)
        """
        if self.current_date_idx >= len(self.spot_data):
            return np.ones(21)
        
        prices = []
        
        # Spot price
        spot_row = self.spot_data.iloc[self.current_date_idx]
        spot_price = float(spot_row['close'])
        prices.append(spot_price)
        
        # Option prices (mid-price between bid and ask)
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
                prices.append(max(mid_price, 0.01))  # Avoid zero prices
            else:
                prices.append(0.01)  # Default small price for missing data
        
        return np.array(prices, dtype=np.float32)
    
    def _get_option_data(
        self,
        date: pd.Timestamp,
        strike_offset: int,
        expiry_days: int,
        option_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve option data for specific contract.
        
        Args:
            date: Current date
            strike_offset: Strike price offset from spot
            expiry_days: Days until expiration
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with option data or None if not found
        """
        # Filter option chain for matching contract
        mask = (
            (self.option_chain['date'] == date) &
            (self.option_chain['strike_offset'] == strike_offset) &
            (self.option_chain['expiry_days'] == expiry_days) &
            (self.option_chain['type'] == option_type)
        )
        
        matching = self.option_chain[mask]
        
        if len(matching) > 0:
            return matching.iloc[0].to_dict()
        else:
            return None
    
    def _calculate_portfolio_value(self, prices: np.ndarray) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            prices: Current asset prices
            
        Returns:
            Total portfolio value
        """
        holdings_value = np.sum(self.positions * prices)
        total_value = self.cash + holdings_value
        return float(total_value)
    
    def _calculate_reward(
        self,
        prev_value: float,
        curr_value: float,
        trade_costs: float
    ) -> float:
        """
        Calculate reward for current step.
        
        Reward components:
        1. Portfolio return (main signal)
        2. Transaction costs (penalty)
        3. Risk penalty (variance of positions)
        
        Args:
            prev_value: Portfolio value before action
            curr_value: Portfolio value after action
            trade_costs: Transaction costs incurred
            
        Returns:
            Reward value
        """
        # Portfolio return component
        portfolio_return = (curr_value - prev_value) / max(prev_value, 1.0)
        
        # Risk penalty based on position variance
        position_variance = np.var(self.positions)
        risk_penalty_term = -self.risk_penalty * position_variance
        
        # Transaction cost penalty
        cost_penalty = -trade_costs / max(prev_value, 1.0)
        
        # Combined reward
        reward = portfolio_return + risk_penalty_term + cost_penalty
        
        # Replace any NaN or inf values
        reward = np.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Scale reward to reasonable range
        reward = np.clip(reward, -10.0, 10.0)
        
        return float(reward)
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about current state.
        
        Returns:
            Dictionary with diagnostic information
        """
        return {
            'step': self.current_step,
            'date_idx': self.current_date_idx,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'total_positions': np.sum(np.abs(self.positions)),
            'net_exposure': np.sum(self.positions)
        }
    
    def render(self):
        """Render the environment (not implemented)."""
        pass
    
    def close(self):
        """Clean up environment resources."""
        pass
