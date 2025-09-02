# tests/test_environment.py
"""
Unit tests for the MultiAsset21DeepHedgingEnv environment.
"""
import pytest
import pandas as pd
import numpy as np
from src.environment.multi_asset_env import MultiAsset21DeepHedgingEnv
from src.environment.option_pricing import create_synthetic_option_chain
from src.utils.data_utils import download_market_data
from src.config.settings import get_config

def test_env_initialization():
    cfg = get_config('data')
    df = download_market_data(
        symbol=cfg['symbol'],
        start_date=cfg['start_date'],
        end_date=cfg['end_date'],
        interval=cfg['interval']
    )
    option_chain = create_synthetic_option_chain(df, get_config('option'))

    strikes = get_config('option')['strike_offsets']
    expiries = get_config('option')['expiry_days']
    types_ = get_config('option')['option_types']
    asset_universe = [{'strike_offset': s, 'expiry_days': e, 'type': t}
                      for e in expiries for s in strikes for t in types_]
    env = MultiAsset21DeepHedgingEnv(df, option_chain, asset_universe)

    # Check spaces
    assert env.action_space.shape == (env.n_assets,)
    assert env.observation_space.shape == (env.obs_dim,)

    # Reset and step
    state, _ = env.reset()
    action = env.action_space.sample()
    next_state, reward, done, trunc, info = env.step(action)

    assert isinstance(next_state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert next_state.shape == (env.obs_dim,)

if __name__ == '__main__':
    pytest.main([__file__])