"""
Agents Module

This module contains reinforcement learning agents for option hedging,
including PPO implementation and evaluation utilities.
"""

from src.agents.ppo_agent import PPOAgent
from src.agents.evaluation import evaluate_model, calculate_performance_metrics

__all__ = [
    "PPOAgent",
    "evaluate_model",
    "calculate_performance_metrics",
]
