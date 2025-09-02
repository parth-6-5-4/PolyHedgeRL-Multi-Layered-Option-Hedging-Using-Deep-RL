# scripts/train_agent.py
"""
Usage: python scripts/train_agent.py --timesteps <num_timesteps>
Train PPO agent in MultiAsset21DeepHedgingEnv and save checkpoints.
"""
import argparse
import os
import logging
from src.config.settings import get_config, create_directories
from src.utils.data_utils import download_market_data
from src.environment.option_pricing import create_synthetic_option_chain
from src.environment.multi_asset_env import MultiAsset21DeepHedgingEnv
from src.agents.ppo_agent import PPOAgent

# Setup logging
logging.basicConfig(level=get_config('logging')['log_level'])
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train PPO agent for PolyHedgeRL')
    parser.add_argument('--timesteps', type=int, default=50000, help='Total timesteps to train')
    args = parser.parse_args()

    # Prepare directories
    create_directories()

    # Load data
    data_cfg = get_config('data')
    df = download_market_data(
        symbol=data_cfg['symbol'],
        start_date=data_cfg['start_date'],
        end_date=data_cfg['end_date'],
        interval=data_cfg['interval']
    )
    option_chain = create_synthetic_option_chain(df, get_config('option'))

    # Asset universe
    strikes = get_config('option')['strike_offsets']
    expiries = get_config('option')['expiry_days']
    types_ = get_config('option')['option_types']
    asset_universe = [{'strike_offset': s, 'expiry_days': e, 'type': t}
                      for e in expiries for s in strikes for t in types_]

    # Initialize environment and agent
    env = MultiAsset21DeepHedgingEnv(df, option_chain, asset_universe)
    agent = PPOAgent(env)

    # Create and train model
    agent.create_model()
    agent.train(total_timesteps=args.timesteps)

if __name__ == '__main__':
    main()
