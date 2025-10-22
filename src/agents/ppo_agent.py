# agents/ppo_agent.py
"""
Proximal Policy Optimization Agent Module for PolyHedgeRL
Handles model creation, training, and checkpointing.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.config.settings import get_config
import os
import logging
import torch

logger = logging.getLogger(__name__)

def get_device():
    """
    Automatically detect and return the best available device.
    Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    """
    if torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon GPU) detected and will be used for training")
        return "mps"
    elif torch.cuda.is_available():
        logger.info("CUDA GPU detected and will be used for training")
        return "cuda"
    else:
        logger.info("No GPU detected, using CPU for training")
        return "cpu"

class PPOAgent:
    """
    Wrapper for creating and training a PPO agent with specified configurations.
    Automatically uses MPS acceleration on Apple Silicon Macs.
    """
    def __init__(self, env, model_dir: str = None, config_name: str = 'ppo', device: str = 'auto'):
        self.env = env
        self.config = get_config(config_name)
        self.model_dir = model_dir or get_config('paths')['models_dir']
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Auto-detect device if not specified
        if device == 'auto':
            self.device = get_device()
        else:
            self.device = device
            
        logger.info(f"PPOAgent initialized with device: {self.device}")
        self.model = None

    def create_model(self, env=None):
        """
        Instantiate a PPO model with MPS/CUDA/CPU device support.
        """
        if env is None:
            env = self.env
        self.model = PPO(
            policy=self.config['policy'],
            env=env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            verbose=self.config['verbose'],
            device=self.device  # Enable MPS/CUDA/CPU
        )
        logger.info(f"PPO model created on device: {self.model.device}")
        return self.model

    def train(
        self,
        total_timesteps: int,
        eval_env=None,
        eval_freq: int = None,
        save_freq: int = None
    ):
        """
        Train the PPO model with optional evaluation and checkpoint callbacks.
        """
        if self.model is None:
            self.create_model()

        callbacks = []
        
        # Checkpoint callback
        save_freq = save_freq or get_config('model')['save_frequency']
        checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=checkpoint_dir,
            name_prefix=get_config('model')['checkpoint_prefix']
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        if eval_env is not None:
            eval_freq = eval_freq or get_config('model')['eval_frequency']
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=self.model_dir,
                log_path=self.model_dir,
                eval_freq=eval_freq,
                deterministic=True,
                verbose=1
            )
            callbacks.append(eval_callback)

        logger.info(f"Starting training for {total_timesteps} timesteps")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )
        
        final_path = os.path.join(self.model_dir, get_config('model')['final_model_name'])
        self.model.save(final_path)
        logger.info(f"Training complete. Model saved to {final_path}")
        return self.model

    def load(self, model_path: str):
        """
        Load a pre-trained PPO model.
        """
        self.model = PPO.load(model_path, env=self.env)
        logger.info(f"Loaded model from {model_path}")
        return self.model
