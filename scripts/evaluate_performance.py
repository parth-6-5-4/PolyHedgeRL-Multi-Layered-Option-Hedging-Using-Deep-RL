# scripts/evaluate_performance.py
"""
Evaluate a trained PPO agent on test data.
Usage: python scripts/evaluate_performance.py [--model results/models/ppo_agent_final.zip] [--episodes 10]
"""
import argparse
import logging
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_config, create_directories
from src.utils.data_utils import download_market_data
from src.environment.option_pricing import create_synthetic_option_chain
from src.environment.Multi_asset_env import MultiAsset21DeepHedgingEnv
from stable_baselines3 import PPO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('evaluation')

def evaluate_model(model, env, episodes=10):
    """Evaluate model over multiple episodes."""
    logger.info(f"Evaluating model over {episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    final_portfolio_values = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Get final portfolio value if available
        if hasattr(env, 'portfolio_value'):
            final_portfolio_values.append(env.portfolio_value)
        
        logger.info(f"Episode {episode+1}/{episodes}: Reward={episode_reward:.2f}, Steps={steps}")
    
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths)
    }
    
    if final_portfolio_values:
        metrics['mean_final_value'] = np.mean(final_portfolio_values)
        metrics['std_final_value'] = np.std(final_portfolio_values)
    
    return metrics, episode_rewards

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained PPO agent')
    parser.add_argument('--model', type=str, default='results/models/ppo_agent_final.zip',
                       help='Path to trained model (default: results/models/ppo_agent_final.zip)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                       help='Start date for evaluation data (default: 2023-01-01)')
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("POLYHEDGERL MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Evaluation data start: {args.start_date}")
    logger.info("=" * 80)
    
    # Create directories
    create_directories()
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        logger.info("Available models:")
        models_dir = Path('results/models')
        if models_dir.exists():
            for model_file in sorted(models_dir.glob('*.zip')):
                logger.info(f"  - {model_file}")
        return
    
    # Load test data
    logger.info("Loading test data...")
    data_cfg = get_config('data')
    df = download_market_data(
        symbol=data_cfg['symbol'],
        start_date=args.start_date,
        end_date='2025-10-19',  # Use recent data for testing
        interval=data_cfg['interval']
    )
    logger.info(f"Loaded {len(df)} days of test data")
    
    # Create environment
    logger.info("Creating test environment...")
    option_cfg = get_config('option')
    option_chain = create_synthetic_option_chain(df, option_cfg)
    
    strikes = option_cfg['strike_offsets']
    expiries = option_cfg['expiry_days']
    types_ = option_cfg['option_types']
    asset_universe = [
        {'strike_offset': s, 'expiry_days': e, 'type': t}
        for e in expiries for s in strikes for t in types_
    ]
    
    env = MultiAsset21DeepHedgingEnv(df, option_chain, asset_universe)
    logger.info(f"Environment created with {len(df)} timesteps")
    
    # Load model
    logger.info("Loading trained model...")
    model = PPO.load(args.model)
    logger.info("Model loaded successfully")
    
    # Evaluate
    logger.info("=" * 80)
    logger.info("RUNNING EVALUATION")
    logger.info("=" * 80)
    
    metrics, episode_rewards = evaluate_model(model, env, episodes=args.episodes)
    
    # Print results
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Mean Reward:      {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    logger.info(f"Min Reward:       {metrics['min_reward']:.2f}")
    logger.info(f"Max Reward:       {metrics['max_reward']:.2f}")
    logger.info(f"Avg Episode Len:  {metrics['mean_episode_length']:.0f} steps")
    
    if 'mean_final_value' in metrics:
        logger.info(f"Final Portfolio:  ${metrics['mean_final_value']:.2f} ± ${metrics['std_final_value']:.2f}")
    
    # Save results
    results_path = Path(get_config('paths')['reports_dir'])
    results_path.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame({
        'episode': range(1, len(episode_rewards) + 1),
        'reward': episode_rewards
    })
    
    output_file = results_path / 'evaluation_results.csv'
    results_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")
    
    # Save summary
    summary_file = results_path / 'evaluation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("PolyHedgeRL Model Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"Test Data: {args.start_date} to 2025-10-19\n\n")
        f.write("Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value}\n")
    
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("=" * 80)
    logger.info("✓ Evaluation complete!")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
