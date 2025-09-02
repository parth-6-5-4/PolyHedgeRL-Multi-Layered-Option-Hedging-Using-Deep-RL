# scripts/evaluate_performance.py
"""
Usage: python scripts/evaluate_performance.py --model_path <path> --episodes <n>
Load a trained agent and run evaluation/backtest episodes.
"""
import argparse
import logging
from src.config.settings import get_config
from src.utils.data_utils import download_market_data
from src.environment.option_pricing import create_synthetic_option_chain
from src.environment.multi_asset_env import MultiAsset21DeepHedgingEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.evaluation import evaluate_model, backtest_series
import pandas as pd

# Setup logging
logging.basicConfig(level=get_config('logging')['log_level'])
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO agent for PolyHedgeRL')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model .zip file')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes for evaluation')
    args = parser.parse_args()

    # Load data and environment
    data_cfg = get_config('data')
    df = download_market_data(
        symbol=data_cfg['symbol'],
        start_date=data_cfg['start_date'],
        end_date=data_cfg['end_date'],
        interval=data_cfg['interval']
    )
    option_chain = create_synthetic_option_chain(df, get_config('option'))
    strikes = get_config('option')['strike_offsets']
    expiries = get_config('option')['expiry_days']
    types_ = get_config('option')['option_types']
    asset_universe = [{'strike_offset': s, 'expiry_days': e, 'type': t}
                      for e in expiries for s in strikes for t in types_]

    env = MultiAsset21DeepHedgingEnv(df, option_chain, asset_universe)
    agent = PPOAgent(env)
    model = agent.load(args.model_path)

    # Evaluate
    metrics = evaluate_model(model, env, episodes=args.episodes)
    logger.info(f"Evaluation metrics: {metrics}")

    # Backtest series
    df_bt = backtest_series(model, env, steps=get_config('env')['episode_length'])
    results_path = get_config('paths')['reports_dir']
    df_bt.to_csv(f"{results_path}/backtest_series.csv", index=False)
    logger.info(f"Backtest series saved to {results_path}/backtest_series.csv")

if __name__ == '__main__':
    main()
