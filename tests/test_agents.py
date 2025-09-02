# tests/test_agents.py
"""
Unit tests for PPOAgent functionality.
"""
import pytest
from src.environment.multi_asset_env import MultiAsset21DeepHedgingEnv
from src.environment.option_pricing import create_synthetic_option_chain
from src.utils.data_utils import download_market_data
from src.config.settings import get_config
from src.agents.ppo_agent import PPOAgent

def test_agent_creation_and_training():
    # Prepare minimal data
    cfg = get_config('data')
    df = download_market_data(
        symbol=cfg['symbol'],
        start_date=cfg['start_date'],
        end_date=cfg['end_date'],
        interval=cfg['interval']
    )
    option_chain = create_synthetic_option_chain(df.iloc[:50], get_config('option'))

    strikes = get_config('option')['strike_offsets']
    expiries = get_config('option')['expiry_days']
    types_ = get_config('option')['option_types']
    asset_universe = [{'strike_offset': s, 'expiry_days': e, 'type': t}
                      for e in expiries for s in strikes for t in types_]

    env = MultiAsset21DeepHedgingEnv(df.iloc[:100], option_chain, asset_universe)
    agent = PPOAgent(env)
    model = agent.create_model()

    # Train for a small number of timesteps
    model.learn(total_timesteps=100)

    # Check model has been trained/saved attributes
    assert model is not None
    assert hasattr(model, 'policy')

if __name__ == '__main__':
    pytest.main([__file__])