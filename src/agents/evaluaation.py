# agents/evaluation.py
"""
Performance Evaluation Module for PolyHedgeRL
Provides functions for backtesting, metrics calculation, and result aggregation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from src.utils.data_utils import safe_mean
import logging

eval_logger = logging.getLogger(__name__)

def evaluate_model(
    model,
    env,
    episodes: int = 100,
    deterministic: bool = True
) -> Dict[str, Any]:
    """
    Run multiple episodes to evaluate agent performance.
    
    Args:
        model: Trained RL model
        env: Gym environment for evaluation
        episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        
    Returns:
        Dictionary of performance metrics
    """
    rewards_per_episode = []
    lengths = []

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        length = 0

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            length += 1
            if done:
                break

        rewards_per_episode.append(total_reward)
        lengths.append(length)

    rewards = np.array(rewards_per_episode)
    metrics = {
        'mean_reward': rewards.mean(),
        'std_reward': rewards.std(),
        'sharpe_ratio': rewards.mean() / (rewards.std() + 1e-8),
        'max_drawdown': np.min(np.minimum.accumulate(rewards) - rewards)
    }
    eval_logger.info(f"Evaluation results over {episodes} episodes: {metrics}")
    return metrics


def backtest_series(
    model,
    env,
    steps: int = 200,
    deterministic: bool = True
) -> pd.DataFrame:
    """
    Single-episode backtest producing time series of P&L and positions.
    
    Args:
        model: Trained RL model
        env: Gym environment
        steps: Maximum number of steps
        deterministic: Whether to use deterministic actions
        
    Returns:
        DataFrame with columns ['step','action','reward','cumulative_reward']
    """
    obs, _ = env.reset()
    records = []
    cumulative_reward = 0.0

    for step in range(steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, _, _ = env.step(action)
        cumulative_reward += reward
        records.append({
            'step': step,
            'action': action.tolist(),
            'reward': reward,
            'cumulative_reward': cumulative_reward
        })
        if done:
            break

    df = pd.DataFrame(records)
    eval_logger.info(f"Backtest completed: final cumulative reward {cumulative_reward}")
    return df


def summary_to_dataframe(results: list) -> pd.DataFrame:
    """
    Convert a list of evaluation results into a summary DataFrame.
    
    Args:
        results: List of dicts with keys ['tx_cost','risk_pen','final_pnl','avg_reward','sharpe']
        
    Returns:
        Pandas DataFrame structured for reporting
    """
    df = pd.DataFrame(results)
    # Ensure correct ordering
    df = df[['tx_cost','risk_pen','final_pnl','avg_reward','sharpe']]
    return df