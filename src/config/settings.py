"""
PolyHedgeRL Configuration Settings
Global configuration parameters for the reinforcement learning trading system.
"""

import os
from typing import Dict, List, Any

# Data Configuration
DATA_CONFIG = {
    'symbol': '^GSPC',
    'start_date': '2010-01-01',
    'end_date': '2025-01-01',
    'interval': '1d',
    'lookback_window': 10,
    'min_training_days': 100,
    'test_split_date': '2023-01-01'
}

# Environment Configuration
ENV_CONFIG = {
    'n_assets': 21,  # 1 spot + 20 options
    'option_features': 5,  # bid, ask, iv, moneyness, oi
    'base_features': 7,
    'obs_dim': 107,  # 7 + 20*5
    'action_bounds': (-1.0, 1.0),
    'transaction_cost': 0.001,
    'risk_penalty': 0.01,
    'episode_length': 200
}

# Option Chain Configuration
OPTION_CONFIG = {
    'strike_offsets': [-100, -50, 0, 50, 100],
    'expiry_days': [30, 60],
    'option_types': ['call', 'put'],
    'min_iv': 0.16,
    'max_iv': 0.5,
    'iv_noise_std': 0.04,
    'min_bid': 2.0,
    'bid_ask_spread_range': (0.5, 3.0),
    'oi_range': (100, 3500)
}

# PPO Training Configuration
PPO_CONFIG = {
    'policy': 'MlpPolicy',
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'verbose': 1
}

# Validation Configuration
VALIDATION_CONFIG = {
    'train_years': 3,
    'test_years': 1,
    'regime_periods': {
        'bull': ('2017-01-01', '2019-12-31'),
        'covid': ('2020-02-01', '2020-06-01'),
        'volatile': ('2022-01-01', '2023-01-01'),
        'recent': ('2023-01-02', '2025-01-01')
    },
    'parameter_sweep': {
        'transaction_costs': [0.0001, 0.001, 0.005],
        'risk_penalties': [0.005, 0.01, 0.05]
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'polyhedge_rl.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# File Paths
PATHS = {
    'data_dir': 'data',
    'raw_data_dir': 'data/raw',
    'processed_data_dir': 'data/processed',
    'synthetic_data_dir': 'data/synthetic',
    'results_dir': 'results',
    'models_dir': 'results/models',
    'plots_dir': 'results/plots',
    'logs_dir': 'results/logs',
    'reports_dir': 'results/reports'
}

# Ensure directories exist
def create_directories():
    """Create all necessary directories if they don't exist."""
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    'save_frequency': 10000,
    'eval_frequency': 5000,
    'checkpoint_prefix': 'polyhedge_model',
    'best_model_name': 'best_model.zip',
    'final_model_name': 'final_model.zip'
}

# Live Trading Configuration
LIVE_CONFIG = {
    'api_provider': 'yfinance',  # 'yfinance', 'alpaca', 'ibkr'
    'update_frequency': 60,  # seconds
    'max_live_steps': 1000,
    'paper_trading': True,
    'position_limits': {
        'max_single_position': 0.2,
        'max_total_exposure': 1.0
    }
}

# Risk Management
RISK_CONFIG = {
    'max_drawdown_threshold': 0.15,
    'volatility_lookback': 20,
    'position_sizing_method': 'kelly',
    'rebalance_threshold': 0.05,
    'emergency_stop_loss': 0.20
}

def get_config(config_name: str) -> Dict[str, Any]:
    """
    Get configuration dictionary by name.
    
    Args:
        config_name: Name of configuration ('data', 'env', 'ppo', etc.)
    
    Returns:
        Configuration dictionary
    """
    config_map = {
        'data': DATA_CONFIG,
        'env': ENV_CONFIG,
        'option': OPTION_CONFIG,
        'ppo': PPO_CONFIG,
        'validation': VALIDATION_CONFIG,
        'logging': LOGGING_CONFIG,
        'paths': PATHS,
        'model': MODEL_CONFIG,
        'live': LIVE_CONFIG,
        'risk': RISK_CONFIG
    }
    
    return config_map.get(config_name, {})

def update_config(config_name: str, updates: Dict[str, Any]) -> None:
    """
    Update configuration with new values.
    
    Args:
        config_name: Name of configuration to update
        updates: Dictionary of updates to apply
    """
    config_map = {
        'data': DATA_CONFIG,
        'env': ENV_CONFIG,
        'option': OPTION_CONFIG,
        'ppo': PPO_CONFIG,
        'validation': VALIDATION_CONFIG,
        'live': LIVE_CONFIG,
        'risk': RISK_CONFIG
    }
    
    if config_name in config_map:
        config_map[config_name].update(updates)

# Initialize directories on import
create_directories()