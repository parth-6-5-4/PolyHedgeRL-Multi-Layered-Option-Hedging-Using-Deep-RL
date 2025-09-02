# market_data.py
"""
Market Data Processing Module for PolyHedgeRL
Handles downloading, loading, and preprocessing of spot market data.
"""

import pandas as pd
import os
from typing import Tuple
from src.utils.data_utils import download_market_data, create_time_series_split, create_regime_splits
from src.config.settings import get_config
import logging

logger = logging.getLogger(__name__)

class MarketDataHandler:
    """
    Manages downloading, saving, and splitting market data.
    """
    def __init__(self, config_name: str = 'data'):
        self.config = get_config(config_name)
        self.data_dir = get_config('paths')['processed_data_dir']
        os.makedirs(self.data_dir, exist_ok=True)

    def load_or_download(self) -> pd.DataFrame:
        """
        Load processed data if available, otherwise download and preprocess.
        """
        filename = f"{self.config['symbol']}_processed.csv"
        filepath = os.path.join(self.data_dir, filename)

        if os.path.exists(filepath):
            logger.info(f"Loading processed data from {filepath}")
            df = pd.read_csv(filepath, parse_dates=['date'])
        else:
            logger.info("No processed file found, downloading market data...")
            df = download_market_data(
                symbol=self.config['symbol'],
                start_date=self.config['start_date'],
                end_date=self.config['end_date'],
                interval=self.config['interval']
            )
            df.to_csv(filepath, index=False)
            logger.info(f"Processed data saved to {filepath}")
        return df

    def create_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train, validation, and test splits.
        """
        return create_time_series_split(
            df,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )

    def get_regime_data(self, df: pd.DataFrame) -> dict:
        """
        Generate data slices for each market regime.
        """
        regime_periods = get_config('validation')['regime_periods']
        return create_regime_splits(df, regime_periods)
