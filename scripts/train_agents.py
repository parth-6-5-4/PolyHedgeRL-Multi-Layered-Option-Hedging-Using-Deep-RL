#!/usr/bin/env python3
"""
Train PPO agent in MultiAsset21DeepHedgingEnv and save checkpoints.
Automatically uses MPS (Apple Silicon), CUDA (NVIDIA), or CPU.

Usage: 
    python scripts/train_agents.py --timesteps 50000
    python scripts/train_agents.py --timesteps 100000 --eval_freq 5000
    python scripts/train_agents.py --timesteps 50000 --device mps
"""
import argparse
import os
import sys
import logging
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.settings import get_config, create_directories
from src.utils.data_utils import download_market_data
from src.environment.option_pricing import create_synthetic_option_chain
from src.environment.Multi_asset_env import MultiAsset21DeepHedgingEnv
from src.agents.ppo_agent import PPOAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_device_info():
    """Print information about available compute devices."""
    logger.info("=" * 60)
    logger.info("DEVICE INFORMATION")
    logger.info("=" * 60)
    
    # Check MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        logger.info("✓ MPS (Apple Silicon GPU) is AVAILABLE")
        logger.info("  → Will use Metal Performance Shaders for acceleration")
    else:
        logger.info("✗ MPS not available")
    
    # Check CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA is AVAILABLE (Device: {torch.cuda.get_device_name(0)})")
    else:
        logger.info("✗ CUDA not available")
    
    # CPU is always available
    logger.info(f"✓ CPU: {torch.get_num_threads()} threads available")
    logger.info("=" * 60)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PPO agent for PolyHedgeRL')
    parser.add_argument('--timesteps', type=int, default=50000, 
                        help='Total timesteps to train (default: 50000)')
    parser.add_argument('--eval_freq', type=int, default=5000,
                        help='Evaluation frequency (default: 5000)')
    parser.add_argument('--save_freq', type=int, default=10000,
                        help='Model save frequency (default: 10000)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'mps', 'cuda', 'cpu'],
                        help='Device to use for training (default: auto-detect)')
    args = parser.parse_args()

    logger.info("Starting PolyHedgeRL training pipeline")
    logger.info(f"Training for {args.timesteps} timesteps")
    
    # Print device information
    print_device_info()

    # Prepare directories
    create_directories()
    logger.info("Directories created/verified")

    # Load data
    logger.info("Downloading/loading market data...")
    data_cfg = get_config('data')
    df = download_market_data(
        symbol=data_cfg['symbol'],
        start_date=data_cfg['start_date'],
        end_date=data_cfg['end_date'],
        interval=data_cfg['interval']
    )
    logger.info(f"Loaded {len(df)} days of market data")

    # Generate synthetic option chain
    logger.info("Generating synthetic option chain...")
    option_config = get_config('option')
    option_chain = create_synthetic_option_chain(df, option_config)
    logger.info(f"Generated {len(option_chain)} option contracts")

    # Build asset universe (20 options)
    strikes = option_config['strike_offsets']
    expiries = option_config['expiry_days']
    types_ = option_config['option_types']
    
    asset_universe = [
        {'strike_offset': s, 'expiry_days': e, 'type': t}
        for e in expiries for s in strikes for t in types_
    ]
    logger.info(f"Asset universe: {len(asset_universe)} options")
    
    # Verify we have exactly 20 options
    assert len(asset_universe) == 20, f"Expected 20 options, got {len(asset_universe)}"

    # Initialize environment
    logger.info("Initializing training environment...")
    env_config = get_config('env')
    env = MultiAsset21DeepHedgingEnv(
        spot_data=df,
        option_chain=option_chain,
        asset_universe=asset_universe,
        transaction_cost=env_config['transaction_cost'],
        risk_penalty=env_config['risk_penalty'],
        episode_length=env_config['episode_length']
    )
    logger.info(f"Environment initialized: obs_space={env.observation_space.shape}, action_space={env.action_space.shape}")

    # Create evaluation environment (optional)
    eval_env = MultiAsset21DeepHedgingEnv(
        spot_data=df,
        option_chain=option_chain,
        asset_universe=asset_universe,
        transaction_cost=env_config['transaction_cost'],
        risk_penalty=env_config['risk_penalty'],
        episode_length=env_config['episode_length']
    )

    # Initialize PPO agent with device selection
    logger.info("Creating PPO agent...")
    agent = PPOAgent(env, device=args.device)
    agent.create_model()
    logger.info(f"PPO model created successfully on device: {agent.device}")

    # Train the model
    logger.info("Starting training...")
    logger.info(f"Device: {agent.device.upper()}")
    try:
        agent.train(
            total_timesteps=args.timesteps,
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq
        )
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info(f"Model saved to {agent.model_dir}")
    logger.info("Training pipeline complete!")


if __name__ == '__main__':
    main()
