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
    final_values = []

    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0.0
        length = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            length += 1
            if done or truncated:
                break
        
        rewards_per_episode.append(total_reward)
        lengths.append(length)
        final_values.append(info.get('portfolio_value', env.initial_capital))

    rewards = np.array(rewards_per_episode)
    final_vals = np.array(final_values)
    
    # Calculate additional metrics
    total_returns = (final_vals - env.initial_capital) / env.initial_capital
    
    metrics = {
        'mean_reward': float(rewards.mean()),
        'std_reward': float(rewards.std()),
        'median_reward': float(np.median(rewards)),
        'min_reward': float(rewards.min()),
        'max_reward': float(rewards.max()),
        'sharpe_ratio': float(rewards.mean() / (rewards.std() + 1e-8)),
        'mean_return': float(total_returns.mean()),
        'std_return': float(total_returns.std()),
        'win_rate': float(np.sum(total_returns > 0) / len(total_returns)),
        'avg_episode_length': float(np.mean(lengths)),
        'max_drawdown': float(calculate_max_drawdown(rewards))
    }
    eval_logger.info(f"Evaluation results over {episodes} episodes: {metrics}")
    return metrics


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from returns series.
    
    Args:
        returns: Array of returns
        
    Returns:
        Maximum drawdown value (negative)
    """
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = np.min(drawdown)
    return max_dd


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
    # Ensure correct ordering for available columns
    available_cols = [col for col in ['tx_cost','risk_pen','final_pnl','avg_reward','sharpe'] if col in df.columns]
    df = df[available_cols]
    return df


def calculate_performance_metrics(returns: np.ndarray, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics from returns.
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary of performance metrics
    """
    if len(returns) == 0:
        return {}
    
    # Annualization factor (assuming daily returns)
    ann_factor = 252
    
    # Basic statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Sharpe ratio
    excess_return = mean_return - risk_free_rate / ann_factor
    sharpe = (excess_return / std_return) * np.sqrt(ann_factor) if std_return > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_return
    sortino = (excess_return / downside_std) * np.sqrt(ann_factor) if downside_std > 0 else 0
    
    # Maximum drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = np.min(drawdown)
    
    # Calmar ratio
    annualized_return = mean_return * ann_factor
    calmar = -annualized_return / max_dd if max_dd < 0 else 0
    
    # Win rate
    win_rate = np.sum(returns > 0) / len(returns)
    
    # Average win/loss
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    
    # Profit factor
    total_wins = np.sum(wins) if len(wins) > 0 else 0
    total_losses = abs(np.sum(losses)) if len(losses) > 0 else 1e-8
    profit_factor = total_wins / total_losses
    
    return {
        'mean_return': float(mean_return),
        'std_return': float(std_return),
        'annualized_return': float(annualized_return),
        'annualized_volatility': float(std_return * np.sqrt(ann_factor)),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'max_drawdown': float(max_dd),
        'calmar_ratio': float(calmar),
        'win_rate': float(win_rate),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'profit_factor': float(profit_factor)
    }