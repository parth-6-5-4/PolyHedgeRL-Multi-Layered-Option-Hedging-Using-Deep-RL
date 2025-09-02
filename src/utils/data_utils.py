"""
Data Utilities for PolyHedgeRL
Comprehensive data manipulation and preprocessing utilities.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)

def safe_float(x) -> float:
    """
    Safely convert input to float, handling pandas Series and edge cases.
    
    Args:
        x: Input value to convert
        
    Returns:
        Float value, 0.0 if conversion fails
    """
    try:
        if isinstance(x, pd.Series):
            val = float(x.iloc[0]) if not x.empty else 0.0
        else:
            val = float(x)
        if not np.isfinite(val):
            return 0.0
        return val
    except (ValueError, TypeError, IndexError):
        return 0.0

def safe_mean(series: pd.Series) -> float:
    """
    Safely calculate mean of pandas Series.
    
    Args:
        series: Input pandas Series
        
    Returns:
        Mean value or 0.0 if calculation fails
    """
    try:
        if isinstance(series, pd.Series) and not series.empty:
            val = series.mean()
        else:
            val = float(series) if series is not None else 0.0
        return float(val) if np.isfinite(val) else 0.0
    except (ValueError, TypeError):
        return 0.0

def download_market_data(
    symbol: str = "^GSPC",
    start_date: str = "2010-01-01",
    end_date: str = "2025-01-01",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Download and preprocess market data from Yahoo Finance.
    
    Args:
        symbol: Market symbol to download
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval (1d, 1h, etc.)
        
    Returns:
        Processed DataFrame with market data
    """
    logger.info(f"Downloading {symbol} data from {start_date} to {end_date}")
    
    try:
        # Download data
        df = yf.download(
            symbol, 
            start=start_date, 
            end=end_date, 
            interval=interval, 
            auto_adjust=True,
            progress=False
        )
        
        if df.empty:
            raise ValueError(f"No data downloaded for {symbol}")
        
        # Reset index and clean columns
        df = df.reset_index()
        
        # Standardize column names
        column_mapping = {
            'Date': 'date',
            'Close': 'close',
            'Open': 'open', 
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        df = df[['date', 'close']].copy()  # Keep only essential columns
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate returns
        df['return'] = df['close'].pct_change()
        
        # Calculate rolling volatility
        df['realized_vol_10d'] = df['return'].rolling(window=10).std() * np.sqrt(252)
        df['realized_vol_20d'] = df['return'].rolling(window=20).std() * np.sqrt(252)
        
        # Additional technical indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Price momentum indicators
        df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
        
        # Remove rows with NaN values
        df = df.dropna().reset_index(drop=True)
        
        logger.info(f"Successfully processed {len(df)} data points")
        return df
        
    except Exception as e:
        logger.error(f"Error downloading market data: {str(e)}")
        raise

def create_time_series_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create time-series aware train/validation/test splits.
    
    Args:
        df: Input DataFrame with time series data
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def create_regime_splits(
    df: pd.DataFrame,
    regime_periods: Dict[str, Tuple[str, str]]
) -> Dict[str, pd.DataFrame]:
    """
    Split data into different market regime periods.
    
    Args:
        df: Input DataFrame with date column
        regime_periods: Dictionary mapping regime names to (start, end) date tuples
        
    Returns:
        Dictionary mapping regime names to DataFrames
    """
    regime_data = {}
    
    for regime_name, (start_date, end_date) in regime_periods.items():
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
        regime_df = df[mask].copy().reset_index(drop=True)
        
        if len(regime_df) > 0:
            regime_data[regime_name] = regime_df
            logger.info(f"Regime '{regime_name}': {len(regime_df)} data points from {start_date} to {end_date}")
        else:
            logger.warning(f"No data found for regime '{regime_name}' in period {start_date} to {end_date}")
    
    return regime_data

def calculate_rolling_metrics(
    df: pd.DataFrame,
    price_col: str = 'close',
    return_col: str = 'return'
) -> pd.DataFrame:
    """
    Calculate comprehensive rolling metrics for market data.
    
    Args:
        df: Input DataFrame
        price_col: Name of price column
        return_col: Name of returns column
        
    Returns:
        DataFrame with additional rolling metrics
    """
    df = df.copy()
    
    # Rolling volatility metrics
    for window in [5, 10, 20, 50]:
        df[f'vol_{window}d'] = df[return_col].rolling(window).std() * np.sqrt(252)
        df[f'mean_return_{window}d'] = df[return_col].rolling(window).mean() * 252
    
    # Rolling price metrics
    for window in [5, 10, 20, 50, 200]:
        df[f'sma_{window}'] = df[price_col].rolling(window).mean()
        df[f'price_ratio_{window}'] = df[price_col] / df[f'sma_{window}']
    
    # Volatility-adjusted returns (Sharpe-like)
    df['vol_adj_return_10d'] = df['mean_return_10d'] / (df['vol_10d'] + 1e-8)
    df['vol_adj_return_20d'] = df['mean_return_20d'] / (df['vol_20d'] + 1e-8)
    
    # Trend indicators
    df['trend_5_20'] = df['sma_5'] / df['sma_20'] - 1
    df['trend_20_50'] = df['sma_20'] / df['sma_50'] - 1
    
    return df

def validate_data_quality(df: pd.DataFrame, critical_columns: List[str]) -> bool:
    """
    Validate data quality and completeness.
    
    Args:
        df: DataFrame to validate
        critical_columns: List of columns that must be present and non-null
        
    Returns:
        True if data passes validation, False otherwise
    """
    # Check if DataFrame is empty
    if df.empty:
        logger.error("DataFrame is empty")
        return False
    
    # Check for missing critical columns
    missing_cols = [col for col in critical_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing critical columns: {missing_cols}")
        return False
    
    # Check for excessive NaN values
    for col in critical_columns:
        nan_ratio = df[col].isna().sum() / len(df)
        if nan_ratio > 0.1:  # More than 10% NaN
            logger.warning(f"Column '{col}' has {nan_ratio:.1%} NaN values")
    
    # Check for data consistency
    if 'date' in df.columns:
        # Check for duplicate dates
        if df['date'].duplicated().any():
            logger.warning("Duplicate dates found in data")
        
        # Check for proper chronological order
        if not df['date'].is_monotonic_increasing:
            logger.warning("Dates are not in chronological order")
    
    # Check for extreme values in price/return columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['return']:
            # Check for extreme returns (>50% in a day)
            extreme_returns = df[abs(df[col]) > 0.5]
            if len(extreme_returns) > 0:
                logger.warning(f"Found {len(extreme_returns)} extreme returns in column '{col}'")
    
    logger.info("Data quality validation completed")
    return True

def resample_data(
    df: pd.DataFrame,
    frequency: str = 'D',
    agg_method: str = 'last'
) -> pd.DataFrame:
    """
    Resample time series data to different frequencies.
    
    Args:
        df: Input DataFrame with date column
        frequency: Target frequency ('D', 'W', 'M', etc.)
        agg_method: Aggregation method ('last', 'mean', 'sum')
        
    Returns:
        Resampled DataFrame
    """
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain 'date' column")
    
    df = df.copy()
    df = df.set_index('date')
    
    agg_functions = {
        'last': 'last',
        'mean': 'mean',
        'sum': 'sum',
        'first': 'first'
    }
    
    if agg_method not in agg_functions:
        raise ValueError(f"Unknown aggregation method: {agg_method}")
    
    # Resample data
    resampled = df.resample(frequency).agg(agg_functions[agg_method])
    resampled = resampled.reset_index()
    
    # Recalculate returns if price data exists
    if 'close' in resampled.columns:
        resampled['return'] = resampled['close'].pct_change()
    
    logger.info(f"Resampled data from {len(df)} to {len(resampled)} periods at {frequency} frequency")
    
    return resampled

def create_features_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Raw market data DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Starting feature engineering pipeline")
    
    df = df.copy()
    
    # Basic price and return features
    df = calculate_rolling_metrics(df)
    
    # Market microstructure features
    df['high_low_ratio'] = df.get('high', df['close']) / df.get('low', df['close'])
    df['volume_sma_ratio'] = df.get('volume', 1) / df.get('volume', 1).rolling(20).mean().fillna(1)
    
    # Volatility features
    df['vol_10_20_ratio'] = df['vol_10d'] / (df['vol_20d'] + 1e-8)
    df['vol_change'] = df['vol_10d'] - df['vol_10d'].shift(1)
    
    # Momentum and mean reversion features
    df['rsi_14'] = calculate_rsi(df['close'], window=14)
    df['bb_position'] = calculate_bollinger_position(df['close'], window=20, std_mult=2)
    
    # Remove rows with excessive NaN values
    df = df.dropna()
    
    logger.info(f"Feature engineering completed. Final shape: {df.shape}")
    
    return df

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_position(
    prices: pd.Series, 
    window: int = 20, 
    std_mult: float = 2
) -> pd.Series:
    """Calculate position within Bollinger Bands."""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * std_mult)
    lower_band = sma - (std * std_mult)
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    return bb_position.clip(0, 1)

def save_processed_data(df: pd.DataFrame, filename: str, directory: str = 'data/processed') -> None:
    """
    Save processed data to CSV file.
    
    Args:
        df: DataFrame to save
        filename: Output filename
        directory: Output directory
    """
    import os
    os.makedirs(directory, exist_ok=True)
    
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath, index=False)
    logger.info(f"Data saved to {filepath}")

def load_processed_data(filename: str, directory: str = 'data/processed') -> pd.DataFrame:
    """
    Load processed data from CSV file.
    
    Args:
        filename: Input filename
        directory: Input directory
        
    Returns:
        Loaded DataFrame
    """
    import os
    filepath = os.path.join(directory, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Data loaded from {filepath}, shape: {df.shape}")
    return df