"""
PolyHedgeRL - Multi-Layered Option Hedging Using Deep Reinforcement Learning

A professional framework for training RL agents to hedge multi-asset derivative portfolios
using deep hedging techniques.

Author: Parth Dambhare
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Parth Dambhare"
__email__ = "your.email@example.com"

from src.environment.Multi_asset_env import MultiAsset21DeepHedgingEnv
from src.agents.ppo_agent import PPOAgent
from src.config.settings import get_config

__all__ = [
    "MultiAsset21DeepHedgingEnv",
    "PPOAgent",
    "get_config",
]
