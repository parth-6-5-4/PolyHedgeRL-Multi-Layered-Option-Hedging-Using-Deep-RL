"""
Improved training script with:
1. RecurrentPPO for time series
2. Better hyperparameters tuned for finance
3. Full dataset (2015-2025)
4. Extended training (200k+ timesteps)
5. Advanced callbacks and monitoring
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import get_config, create_directories
from src.utils.data_utils import download_market_data
from src.environment.option_pricing import create_synthetic_option_chain
from src.environment.improved_env import ImprovedMultiAssetEnv

# Use RecurrentPPO for time series data
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('results/logs/improved_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('improved_training')


def create_env(df, option_chain, asset_universe):
    """Create and wrap environment."""
    env = ImprovedMultiAssetEnv(
        spot_data=df,
        option_chain=option_chain,
        asset_universe=asset_universe,
        initial_capital=100000.0,
        transaction_cost=0.0005,  # 0.05% (reduced)
        max_position_size=0.2,  # 20% max per position
        risk_free_rate=0.02,
        episode_length=252,  # Full trading year
        lookback_window=20
    )
    return env


def train_improved_model(
    env,
    total_timesteps=200000,
    save_interval=10000,
    output_dir='results/models_improved'
):
    """
    Train using RecurrentPPO with finance-optimized hyperparameters.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("IMPROVED TRAINING WITH RECURRENTPPO")
    logger.info("=" * 80)
    
    # Hyperparameters optimized for financial time series
    hyperparams = {
        'policy': 'MlpLstmPolicy',  # LSTM policy for temporal dependencies
        'learning_rate': 3e-5,  # Lower learning rate for stability
        'n_steps': 2048,  # Longer rollouts
        'batch_size': 128,  # Larger batches for LSTM
        'n_epochs': 20,  # More epochs per update
        'gamma': 0.995,  # Higher discount for long-term rewards
        'gae_lambda': 0.98,  # Higher GAE lambda
        'clip_range': 0.15,  # Tighter clipping
        'ent_coef': 0.005,  # Lower entropy for more deterministic policy
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'verbose': 1,
        'tensorboard_log': str(output_path / 'tensorboard')
    }
    
    logger.info("Hyperparameters:")
    for key, value in hyperparams.items():
        logger.info(f"  {key}: {value}")
    
    # Create model
    logger.info("Creating RecurrentPPO model...")
    model = RecurrentPPO(**hyperparams, env=env)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_interval,
        save_path=str(output_path),
        name_prefix='rppo_model'
    )
    
    logger.info(f"Training for {total_timesteps:,} timesteps...")
    logger.info(f"Saving checkpoints every {save_interval:,} steps")
    
    start_time = datetime.now()
    
    try:
        # Train
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            log_interval=10,
            progress_bar=True
        )
        
        # Save final model
        final_path = output_path / 'rppo_model_final.zip'
        model.save(str(final_path))
        logger.info(f"✓ Final model saved: {final_path}")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration}")
        logger.info(f"Average speed: {total_timesteps / duration.total_seconds():.1f} steps/sec")
        logger.info(f"Final model: {final_path}")
        logger.info("=" * 80)
        
        return model, final_path
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted!")
        emergency_path = output_path / f'rppo_emergency_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        model.save(str(emergency_path))
        logger.info(f"Emergency save: {emergency_path}")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def fetch_full_dataset(symbol='^GSPC', start_date='2015-01-01'):
    """Fetch full historical dataset."""
    logger.info("=" * 80)
    logger.info("FETCHING FULL DATASET")
    logger.info("=" * 80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Date range: {start_date} to today")
    
    df = download_market_data(
        symbol=symbol,
        start_date=start_date,
        end_date=datetime.now().strftime('%Y-%m-%d'),
        interval='1d'
    )
    
    logger.info(f"Downloaded {len(df)} trading days")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Improved PolyHedgeRL Training')
    parser.add_argument('--symbol', type=str, default='^GSPC',
                       help='Trading symbol (default: ^GSPC)')
    parser.add_argument('--start-date', type=str, default='2015-01-01',
                       help='Start date for data (default: 2015-01-01)')
    parser.add_argument('--timesteps', type=int, default=200000,
                       help='Total training timesteps (default: 200k)')
    parser.add_argument('--save-interval', type=int, default=10000,
                       help='Save interval (default: 10k)')
    parser.add_argument('--output-dir', type=str, default='results/models_improved',
                       help='Output directory')
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("POLYHEDGERL IMPROVED TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Timesteps: {args.timesteps:,}")
    logger.info(f"Start date: {args.start_date}")
    logger.info("=" * 80)
    
    # Create directories
    create_directories()
    
    # Fetch data
    df = fetch_full_dataset(args.symbol, args.start_date)
    
    # Create option chain
    logger.info("Creating synthetic option chain...")
    option_cfg = get_config('option')
    option_chain = create_synthetic_option_chain(df, option_cfg)
    logger.info(f"Generated {len(option_chain)} option records")
    
    # Define asset universe
    strikes = option_cfg['strike_offsets']
    expiries = option_cfg['expiry_days']
    types_ = option_cfg['option_types']
    asset_universe = [
        {'strike_offset': s, 'expiry_days': e, 'type': t}
        for e in expiries for s in strikes for t in types_
    ]
    logger.info(f"Asset universe: {len(asset_universe)} options")
    
    # Create environment
    logger.info("Creating improved environment...")
    env = create_env(df, option_chain, asset_universe)
    logger.info("Environment created successfully")
    
    # Train model
    model, model_path = train_improved_model(
        env,
        total_timesteps=args.timesteps,
        save_interval=args.save_interval,
        output_dir=args.output_dir
    )
    
    logger.info("=" * 80)
    logger.info("✓ PIPELINE COMPLETE!")
    logger.info(f"Model saved to: {model_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
