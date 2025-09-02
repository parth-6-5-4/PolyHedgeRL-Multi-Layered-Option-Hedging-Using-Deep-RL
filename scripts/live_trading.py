# scripts/live_trading.py
"""
Usage: python scripts/live_trading.py --steps <n>
Simulate live trading using trained agent and real-time data (or synthetic for demo).
"""
import argparse
import logging
import time
import pandas as pd
import numpy as np
from src.config.settings import get_config
from src.environment.multi_asset_env import MultiAsset21DeepHedgingEnv
from src.environment.option_pricing import create_synthetic_option_chain
from stable_baselines3 import PPO

# Setup logging
logging.basicConfig(level=get_config('logging')['log_level'])
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Simulate live trading with PPO agent')
    parser.add_argument('--steps', type=int, default=100, help='Number of live simulation steps')
    args = parser.parse_args()

    # Load data for demo (or replace with live API)
    data_cfg = get_config('data')
    df_live = pd.read_csv(f"{get_config('paths')['synthetic_data_dir']}/{data_cfg['symbol']}_processed.csv", parse_dates=['date'])
    option_chain_live = create_synthetic_option_chain(df_live, get_config('option'))

    strikes = get_config('option')['strike_offsets']
    expiries = get_config('option')['expiry_days']
    types_ = get_config('option')['option_types']
    asset_universe = [{'strike_offset': s, 'expiry_days': e, 'type': t}
                      for e in expiries for s in strikes for t in types_]

    env = MultiAsset21DeepHedgingEnv(df_live, option_chain_live, asset_universe)

    # Load model
    model_path = f"{get_config('paths')['models_dir']}/{get_config('model')['best_model_name']}"
    model = PPO.load(model_path, env=env)
    logger.info(f"Loaded model from {model_path}")

    obs, _ = env.reset()
    records = []

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        cumulative = records[-1]['cumulative_reward'] + reward if records else reward
        records.append({'step': step, 'action': action.tolist(), 'reward': reward, 'cumulative_reward': cumulative})
        logger.info(f"Step {step}: Reward={reward:.2f}, Cumulative={cumulative:.2f}")
        if done:
            break
        time.sleep(get_config('live')['update_frequency'])

    df_records = pd.DataFrame(records)
    out_path = get_config('paths')['logs_dir']
    df_records.to_csv(f"{out_path}/live_trading_log.csv", index=False)
    logger.info(f"Live trading log saved to {out_path}/live_trading_log.csv")

if __name__ == '__main__':
    main()
