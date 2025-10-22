"""
Synthetic Option Pricing for PolyHedgeRL
Generate realistic option chains for backtesting and simulation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SyntheticOptionPricer:
    """
    Class for generating synthetic option chains with realistic pricing.
    """
    
    def __init__(
        self,
        strike_offsets: List[int] = None,
        expiry_days: List[int] = None,
        option_types: List[str] = None,
        min_iv: float = 0.16,
        max_iv: float = 0.5,
        iv_noise_std: float = 0.04,
        min_bid: float = 2.0,
        bid_ask_spread_range: Tuple[float, float] = (0.5, 3.0),
        oi_range: Tuple[int, int] = (100, 3500)
    ):
        """
        Initialize the synthetic option pricer.
        
        Args:
            strike_offsets: List of strike price offsets from spot
            expiry_days: List of expiration days
            option_types: List of option types ('call', 'put')
            min_iv: Minimum implied volatility
            max_iv: Maximum implied volatility
            iv_noise_std: Standard deviation of IV noise
            min_bid: Minimum bid price
            bid_ask_spread_range: Range for bid-ask spreads
            oi_range: Range for open interest values
        """
        self.strike_offsets = strike_offsets or [-100, -50, 0, 50, 100]
        self.expiry_days = expiry_days or [30, 60]
        self.option_types = option_types or ['call', 'put']
        self.min_iv = min_iv
        self.max_iv = max_iv
        self.iv_noise_std = iv_noise_std
        self.min_bid = min_bid
        self.bid_ask_spread_range = bid_ask_spread_range
        self.oi_range = oi_range
        
        # Risk-free rate (assumed constant for simplicity)
        self.risk_free_rate = 0.02
        
        logger.info(f"Initialized SyntheticOptionPricer with {len(self.strike_offsets)} strikes, "
                   f"{len(self.expiry_days)} expiries, {len(self.option_types)} types")

    def black_scholes_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate Black-Scholes option price.
        
        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        if time_to_expiry <= 0:
            # At expiration
            if option_type == 'call':
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)
        
        # Black-Scholes calculation
        d1 = (np.log(spot / strike) + (self.risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        if option_type == 'call':
            price = spot * norm.cdf(d1) - strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:
            price = strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        return max(0, price)

    def generate_implied_volatility(
        self,
        spot: float,
        strike: float,
        realized_vol: float,
        option_type: str,
        time_to_expiry: float
    ) -> float:
        """
        Generate realistic implied volatility with smile/skew effects.
        
        Args:
            spot: Current spot price
            strike: Strike price
            realized_vol: Historical realized volatility
            option_type: 'call' or 'put'
            time_to_expiry: Time to expiration in years
            
        Returns:
            Implied volatility
        """
        # Base IV from realized volatility
        base_iv = realized_vol
        
        # Moneyness effect (volatility smile/skew)
        moneyness = np.log(spot / strike)
        
        # Typical equity index skew: puts more expensive than calls
        if option_type == 'put':
            skew_adjustment = -0.02 * moneyness  # Put skew
        else:
            skew_adjustment = -0.01 * moneyness  # Call skew (less pronounced)
        
        # Term structure effect
        term_adjustment = 0.02 * np.exp(-2 * time_to_expiry)  # Short-term vol higher
        
        # Add random noise
        noise = np.random.normal(0, self.iv_noise_std)
        
        # Combine effects
        iv = base_iv + skew_adjustment + term_adjustment + noise
        
        # Clamp to reasonable bounds
        iv = np.clip(iv, self.min_iv, self.max_iv)
        
        return iv

    def generate_option_chain(
        self,
        spot_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate complete option chain for all dates in spot data.
        
        Args:
            spot_data: DataFrame with columns ['date', 'close', 'realized_vol_10d']
            
        Returns:
            DataFrame with synthetic option chain
        """
        logger.info(f"Generating option chain for {len(spot_data)} trading days")
        
        option_chains = []
        
        for idx, row in spot_data.iterrows():
            date = row['date']
            spot = float(row['close'])
            realized_vol = float(row.get('realized_vol_10d', 0.2))
            
            # Generate options for each expiry
            for expiry_days in self.expiry_days:
                expiry_date = date + timedelta(days=expiry_days)
                time_to_expiry = expiry_days / 365.25
                
                # Generate options for each strike
                for strike_offset in self.strike_offsets:
                    strike = round(spot + strike_offset)
                    
                    # Generate both calls and puts
                    for option_type in self.option_types:
                        # Generate implied volatility
                        iv = self.generate_implied_volatility(
                            spot, strike, realized_vol, option_type, time_to_expiry
                        )
                        
                        # Calculate theoretical price using Black-Scholes
                        theoretical_price = self.black_scholes_price(
                            spot, strike, time_to_expiry, iv, option_type
                        )
                        
                        # Add market microstructure effects
                        bid, ask = self.generate_bid_ask_spread(theoretical_price, iv, time_to_expiry)
                        
                        # Generate open interest
                        open_interest = self.generate_open_interest(spot, strike, option_type, time_to_expiry)
                        
                        # Calculate moneyness
                        moneyness = spot / strike if strike > 0 else 1.0
                        
                        option_chains.append({
                            'date': date,
                            'expiry': expiry_date,
                            'expiry_days': expiry_days,
                            'strike': strike,
                            'strike_offset': strike_offset,
                            'type': option_type,
                            'spot': spot,
                            'theoretical_price': theoretical_price,
                            'bid': bid,
                            'ask': ask,
                            'mid': (bid + ask) / 2,
                            'iv': iv,
                            'moneyness': moneyness,
                            'open_interest': open_interest,
                            'time_to_expiry': time_to_expiry
                        })
        
        option_chain_df = pd.DataFrame(option_chains)
        logger.info(f"Generated {len(option_chain_df)} option records")
        
        return option_chain_df

    def generate_bid_ask_spread(
        self,
        theoretical_price: float,
        iv: float,
        time_to_expiry: float
    ) -> Tuple[float, float]:
        """
        Generate realistic bid-ask spread around theoretical price.
        
        Args:
            theoretical_price: Black-Scholes theoretical price
            iv: Implied volatility
            time_to_expiry: Time to expiration
            
        Returns:
            Tuple of (bid, ask) prices
        """
        # Base spread as percentage of price
        base_spread_pct = 0.02  # 2% base spread
        
        # Increase spread for high IV and short time to expiry
        iv_adjustment = max(0, (iv - 0.2) * 0.5)  # Higher IV = wider spread
        time_adjustment = max(0, (0.1 - time_to_expiry) * 0.3)  # Short time = wider spread
        
        # Total spread percentage
        spread_pct = base_spread_pct + iv_adjustment + time_adjustment
        spread_pct = min(spread_pct, 0.15)  # Cap at 15%
        
        # Calculate absolute spread
        spread = max(
            theoretical_price * spread_pct,
            np.random.uniform(*self.bid_ask_spread_range)
        )
        
        # Generate bid and ask
        mid_adjustment = np.random.uniform(-0.3, 0.3) * spread  # Random mid adjustment
        mid_price = max(self.min_bid, theoretical_price + mid_adjustment)
        
        bid = max(self.min_bid, mid_price - spread / 2)
        ask = mid_price + spread / 2
        
        return float(bid), float(ask)

    def generate_open_interest(
        self,
        spot: float,
        strike: float,
        option_type: str,
        time_to_expiry: float
    ) -> int:
        """
        Generate realistic open interest values.
        
        Args:
            spot: Current spot price
            strike: Strike price
            option_type: 'call' or 'put'
            time_to_expiry: Time to expiration
            
        Returns:
            Open interest value
        """
        # ATM options have highest open interest
        moneyness = abs(np.log(spot / strike))
        atm_factor = np.exp(-5 * moneyness)  # Exponential decay from ATM
        
        # Longer-dated options have higher open interest
        time_factor = 0.5 + 0.5 * time_to_expiry / (1/12)  # Normalize to monthly
        time_factor = min(time_factor, 1.5)
        
        # Put-call asymmetry (index puts more popular)
        type_factor = 1.2 if option_type == 'put' else 1.0
        
        # Base open interest with random component
        base_oi = np.random.uniform(*self.oi_range)
        adjusted_oi = base_oi * atm_factor * time_factor * type_factor
        
        return int(max(100, adjusted_oi))

    def update_option_prices(
        self,
        option_chain: pd.DataFrame,
        new_spot: float,
        new_realized_vol: float,
        current_date: datetime
    ) -> pd.DataFrame:
        """
        Update option prices for a new spot price and date.
        
        Args:
            option_chain: Existing option chain
            new_spot: New spot price
            new_realized_vol: New realized volatility
            current_date: Current date
            
        Returns:
            Updated option chain
        """
        updated_chain = option_chain.copy()
        
        for idx, row in updated_chain.iterrows():
            # Calculate new time to expiry
            time_to_expiry = max(0, (row['expiry'] - current_date).days / 365.25)
            
            if time_to_expiry <= 0:
                # Option expired
                if row['type'] == 'call':
                    price = max(0, new_spot - row['strike'])
                else:
                    price = max(0, row['strike'] - new_spot)
                updated_chain.loc[idx, 'bid'] = price
                updated_chain.loc[idx, 'ask'] = price
                updated_chain.loc[idx, 'mid'] = price
            else:
                # Update IV and price
                iv = self.generate_implied_volatility(
                    new_spot, row['strike'], new_realized_vol, row['type'], time_to_expiry
                )
                
                theoretical_price = self.black_scholes_price(
                    new_spot, row['strike'], time_to_expiry, iv, row['type']
                )
                
                bid, ask = self.generate_bid_ask_spread(theoretical_price, iv, time_to_expiry)
                
                updated_chain.loc[idx, 'theoretical_price'] = theoretical_price
                updated_chain.loc[idx, 'bid'] = bid
                updated_chain.loc[idx, 'ask'] = ask
                updated_chain.loc[idx, 'mid'] = (bid + ask) / 2
                updated_chain.loc[idx, 'iv'] = iv
                updated_chain.loc[idx, 'time_to_expiry'] = time_to_expiry
        
        updated_chain['spot'] = new_spot
        
        return updated_chain

def create_synthetic_option_chain(
    spot_data: pd.DataFrame,
    config: Dict = None
) -> pd.DataFrame:
    """
    Convenience function to create synthetic option chain.
    
    Args:
        spot_data: DataFrame with spot price data
        config: Configuration dictionary
        
    Returns:
        Synthetic option chain DataFrame
    """
    if config is None:
        config = {}
    
    pricer = SyntheticOptionPricer(**config)
    option_chain = pricer.generate_option_chain(spot_data)
    
    return option_chain

def extract_option_features(
    option_chain: pd.DataFrame,
    date: datetime,
    spot_price: float,
    asset_universe: List[Dict]
) -> List[float]:
    """
    Extract option features for a specific date and asset universe.
    
    Args:
        option_chain: Complete option chain DataFrame
        date: Date to extract features for
        spot_price: Current spot price
        asset_universe: List of option specifications
        
    Returns:
        List of option features
    """
    features = []
    day_chain = option_chain[option_chain['date'] == date]
    
    for asset in asset_universe:
        # Filter for specific option
        option_filter = (
            (day_chain['type'] == asset['type']) &
            (day_chain['expiry_days'] == asset['expiry_days']) &
            (day_chain['strike_offset'] == asset['strike_offset'])
        )
        
        matching_options = day_chain[option_filter]
        
        if not matching_options.empty:
            option = matching_options.iloc[0]
            option_features = [
                float(option['bid']),
                float(option['ask']),
                float(option['iv']),
                float(spot_price / option['strike']),  # Moneyness
                float(option['open_interest'])
            ]
        else:
            # Default values if option not found
            option_features = [0.0, 0.0, 0.2, 1.0, 1000.0]
        
        features.extend(option_features)
    
    return features

def calculate_option_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.02,
    option_type: str = 'call'
) -> Dict[str, float]:
    """
    Calculate option Greeks using Black-Scholes model.
    
    Args:
        spot: Current spot price
        strike: Strike price
        time_to_expiry: Time to expiration in years
        volatility: Implied volatility
        risk_free_rate: Risk-free interest rate
        option_type: 'call' or 'put'
        
    Returns:
        Dictionary with Greeks values
    """
    if time_to_expiry <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))
    
    # Theta
    if option_type == 'call':
        theta = (-(spot * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry))
                - risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)) / 365
    else:
        theta = (-(spot * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry))
                + risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)) / 365
    
    # Vega (same for calls and puts)
    vega = spot * np.sqrt(time_to_expiry) * norm.pdf(d1) / 100
    
    # Rho
    if option_type == 'call':
        rho = strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100
    else:
        rho = -strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }