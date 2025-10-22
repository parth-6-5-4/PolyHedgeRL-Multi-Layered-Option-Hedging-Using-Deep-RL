"""
Evaluate trained RecurrentPPO models on test data.
Calculates comprehensive financial metrics: Sharpe, Sortino, Calmar, Max Drawdown, Win Rate.
"""
import argparse
import logging
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_config, create_directories
from src.utils.data_utils import download_market_data
from src.environment.option_pricing import create_synthetic_option_chain
from src.environment.improved_env import ImprovedMultiAssetEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('evaluation')


def calculate_financial_metrics(returns, portfolio_values, initial_capital=100000):
    """Calculate comprehensive financial performance metrics."""
    returns = np.array(returns)
    portfolio_values = np.array(portfolio_values)
    
    # Basic metrics
    total_return = (portfolio_values[-1] - initial_capital) / initial_capital
    annualized_return = total_return * (252 / len(returns))  # Assuming daily returns
    
    # Sharpe Ratio (annualized)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
    else:
        sortino_ratio = 0.0
    
    # Maximum Drawdown
    cumulative = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - cumulative) / cumulative
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
    
    # Calmar Ratio
    if max_drawdown < 0:
        calmar_ratio = annualized_return / abs(max_drawdown)
    else:
        calmar_ratio = 0.0
    
    # Win Rate
    win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.0
    
    # Volatility (annualized)
    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'win_rate': win_rate,
        'final_portfolio_value': portfolio_values[-1],
        'total_trades': len(returns)
    }


def evaluate_model(model, env, episodes=10):
    """Evaluate RecurrentPPO model over multiple episodes."""
    logger.info(f"Evaluating model over {episodes} episodes...")
    
    episode_rewards = []
    episode_returns = []
    all_portfolio_values = []
    all_step_returns = []
    
    for episode in range(episodes):
        obs = env.reset()
        
        # Initialize LSTM states
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        
        done = False
        episode_reward = 0
        steps = 0
        portfolio_history = []
        step_returns = []
        
        while not done:
            # RecurrentPPO prediction with LSTM states
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )
            
            obs, reward, dones, info = env.step(action)
            
            episode_reward += reward[0]
            steps += 1
            
            # Track portfolio value
            if 'portfolio_value' in info[0]:
                portfolio_history.append(info[0]['portfolio_value'])
            
            # Track step returns
            if len(portfolio_history) > 1:
                step_return = (portfolio_history[-1] - portfolio_history[-2]) / portfolio_history[-2]
                step_returns.append(step_return)
            
            episode_starts = dones
            done = dones[0]
        
        episode_rewards.append(episode_reward)
        
        if len(portfolio_history) > 0:
            initial_value = portfolio_history[0]
            final_value = portfolio_history[-1]
            episode_return = (final_value - initial_value) / initial_value
            episode_returns.append(episode_return)
            all_portfolio_values.extend(portfolio_history)
            all_step_returns.extend(step_returns)
            
            logger.info(
                f"Episode {episode+1}/{episodes}: "
                f"Reward={episode_reward:.2f}, "
                f"Return={episode_return*100:.2f}%, "
                f"Final Value=${final_value:,.2f}, "
                f"Steps={steps}"
            )
        else:
            logger.info(
                f"Episode {episode+1}/{episodes}: "
                f"Reward={episode_reward:.2f}, "
                f"Steps={steps}"
            )
    
    # Calculate comprehensive metrics
    if len(all_step_returns) > 0 and len(all_portfolio_values) > 0:
        financial_metrics = calculate_financial_metrics(
            all_step_returns,
            all_portfolio_values
        )
    else:
        financial_metrics = {}
    
    basic_metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_return': np.mean(episode_returns) if episode_returns else 0.0,
        'std_return': np.std(episode_returns) if episode_returns else 0.0
    }
    
    # Combine all metrics
    all_metrics = {**basic_metrics, **financial_metrics}
    
    return all_metrics, episode_rewards, episode_returns


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RecurrentPPO agent')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (e.g., results/models_improved/rppo_model_80000_steps.zip)')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of evaluation episodes (default: 20)')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='Start date for evaluation data (default: 2024-01-01)')
    parser.add_argument('--end-date', type=str, default='2025-10-20',
                       help='End date for evaluation data (default: 2025-10-20)')
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("POLYHEDGERL RECURRENTPPO MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Test Period: {args.start_date} to {args.end_date}")
    logger.info("=" * 80)
    
    # Create directories
    create_directories()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {args.model}")
        logger.info("\nAvailable models in results/models_improved/:")
        models_dir = Path('results/models_improved')
        if models_dir.exists():
            for model_file in sorted(models_dir.glob('*.zip'), key=lambda x: x.stat().st_mtime, reverse=True):
                size_mb = model_file.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                logger.info(f"  - {model_file.name} ({size_mb:.1f}MB, {mtime})")
        return
    
    # Load test data
    logger.info("Loading test data...")
    df = download_market_data(
        symbol='^GSPC',
        start_date=args.start_date,
        end_date=args.end_date,
        interval='1d'
    )
    logger.info(f"Loaded {len(df)} days of test data")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Create environment
    logger.info("Creating test environment...")
    option_cfg = get_config('option')
    option_chain = create_synthetic_option_chain(df, option_cfg)
    logger.info(f"Generated {len(option_chain)} option records")
    
    strikes = option_cfg['strike_offsets']
    expiries = option_cfg['expiry_days']
    types_ = option_cfg['option_types']
    asset_universe = [
        {'strike_offset': s, 'expiry_days': e, 'type': t}
        for e in expiries for s in strikes for t in types_
    ]
    
    env = ImprovedMultiAssetEnv(
        spot_data=df,
        option_chain=option_chain,
        asset_universe=asset_universe,
        initial_capital=100000.0,
        lookback_window=60
    )
    
    # Wrap in DummyVecEnv for RecurrentPPO
    env = DummyVecEnv([lambda: env])
    logger.info("Environment created successfully")
    
    # Load model
    logger.info("Loading trained RecurrentPPO model...")
    try:
        model = RecurrentPPO.load(args.model)
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Evaluate
    logger.info("=" * 80)
    logger.info("RUNNING EVALUATION")
    logger.info("=" * 80)
    
    metrics, episode_rewards, episode_returns = evaluate_model(
        model, env, episodes=args.episodes
    )
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info("\n📊 BASIC METRICS:")
    logger.info(f"  Mean Reward:        {metrics['mean_reward']:>12.2f} ± {metrics['std_reward']:.2f}")
    logger.info(f"  Min Reward:         {metrics['min_reward']:>12.2f}")
    logger.info(f"  Max Reward:         {metrics['max_reward']:>12.2f}")
    logger.info(f"  Mean Return:        {metrics['mean_return']*100:>11.2f}% ± {metrics['std_return']*100:.2f}%")
    
    if 'sharpe_ratio' in metrics:
        logger.info("\n💰 FINANCIAL METRICS:")
        logger.info(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>12.2f}")
        logger.info(f"  Sortino Ratio:      {metrics['sortino_ratio']:>12.2f}")
        logger.info(f"  Calmar Ratio:       {metrics['calmar_ratio']:>12.2f}")
        logger.info(f"  Max Drawdown:       {metrics['max_drawdown']*100:>11.2f}%")
        logger.info(f"  Volatility (ann.):  {metrics['volatility']*100:>11.2f}%")
        logger.info(f"  Win Rate:           {metrics['win_rate']*100:>11.2f}%")
        logger.info(f"  Annual Return:      {metrics['annualized_return']*100:>11.2f}%")
        logger.info(f"  Final Portfolio:    ${metrics['final_portfolio_value']:>11,.2f}")
        
        # Performance assessment
        logger.info("\n🎯 PERFORMANCE ASSESSMENT:")
        sharpe = metrics['sharpe_ratio']
        annual_return = metrics['annualized_return'] * 100
        max_dd = abs(metrics['max_drawdown'] * 100)
        
        if sharpe >= 2.0:
            logger.info("  Rating: ⭐⭐⭐⭐⭐ WORLD-CLASS (Sharpe ≥ 2.0)")
        elif sharpe >= 1.5:
            logger.info("  Rating: ⭐⭐⭐⭐ EXCELLENT (Sharpe ≥ 1.5)")
        elif sharpe >= 1.2:
            logger.info("  Rating: ⭐⭐⭐ PROFESSIONAL (Sharpe ≥ 1.2)")
        elif sharpe >= 0.8:
            logger.info("  Rating: ⭐⭐ GOOD / MVP READY (Sharpe ≥ 0.8)")
        elif sharpe >= 0.5:
            logger.info("  Rating: ⭐ ACCEPTABLE (Sharpe ≥ 0.5)")
        else:
            logger.info("  Rating: ⚠️  NEEDS IMPROVEMENT (Sharpe < 0.5)")
        
        logger.info(f"\n  Target Metrics:")
        logger.info(f"    Sharpe > 0.8:     {'✅ PASS' if sharpe > 0.8 else '❌ FAIL'}")
        logger.info(f"    Return > 12%:     {'✅ PASS' if annual_return > 12 else '❌ FAIL'}")
        logger.info(f"    Max DD < 25%:     {'✅ PASS' if max_dd < 25 else '❌ FAIL'}")
        logger.info(f"    Win Rate > 52%:   {'✅ PASS' if metrics['win_rate'] > 0.52 else '❌ FAIL'}")
    
    logger.info("=" * 80)
    
    # Save results
    results_path = Path('results/reports')
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Save episode details
    results_df = pd.DataFrame({
        'episode': range(1, len(episode_rewards) + 1),
        'reward': episode_rewards,
        'return': episode_returns if episode_returns else [0] * len(episode_rewards)
    })
    
    model_name = model_path.stem
    output_file = results_path / f'evaluation_{model_name}.csv'
    results_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Episode results saved to: {output_file}")
    
    # Save summary
    summary_file = results_path / f'evaluation_{model_name}_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("PolyHedgeRL RecurrentPPO Model Evaluation\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"Test Period: {args.start_date} to {args.end_date}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Metrics:\n")
        f.write("-" * 80 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'ratio' in key.lower() or 'return' in key.lower() or 'rate' in key.lower():
                    f.write(f"  {key:.<30} {value:>12.4f}\n")
                else:
                    f.write(f"  {key:.<30} {value:>12.2f}\n")
            else:
                f.write(f"  {key:.<30} {value:>12}\n")
    
    logger.info(f"✓ Summary saved to: {summary_file}")
    logger.info("\n" + "=" * 80)
    logger.info("✓ EVALUATION COMPLETE!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
