"""
Command-Line Interface for PolyHedgeRL

Professional CLI for training, evaluation, and live trading operations.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger

console = Console()
logger = setup_logger("cli", level="INFO")


@click.group()
@click.version_option(version="1.0.0", prog_name="PolyHedgeRL")
def cli():
    """
    PolyHedgeRL - Multi-Layered Option Hedging Using Deep Reinforcement Learning
    
    A professional framework for training RL agents to hedge derivative portfolios.
    """
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration YAML file",
)
@click.option(
    "--timesteps",
    "-t",
    type=int,
    default=100_000,
    help="Total training timesteps",
)
@click.option(
    "--save-freq",
    type=int,
    default=10_000,
    help="Save model every N timesteps",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="results/models/ppo_agent.zip",
    help="Output path for trained model",
)
@click.option(
    "--log-interval",
    type=int,
    default=100,
    help="Log training info every N episodes",
)
@click.option(
    "--gpu/--no-gpu",
    default=False,
    help="Use GPU acceleration",
)
def train(
    config: Optional[str],
    timesteps: int,
    save_freq: int,
    output: str,
    log_interval: int,
    gpu: bool,
):
    """Train a new PPO agent for option hedging."""
    from scripts.train_agents import main as train_main
    
    console.print("\n[bold green] Starting Training[/bold green]\n")
    
    # Display configuration
    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Config File", config or "default")
    table.add_row("Total Timesteps", f"{timesteps:,}")
    table.add_row("Save Frequency", f"{save_freq:,}")
    table.add_row("Output Path", output)
    table.add_row("Log Interval", str(log_interval))
    table.add_row("GPU Enabled", "Yes" if gpu else "No")
    
    console.print(table)
    console.print()
    
    logger.info(f"Starting training with {timesteps:,} timesteps")
    
    try:
        train_main(
            config_path=config,
            total_timesteps=timesteps,
            save_freq=save_freq,
            output_path=output,
            log_interval=log_interval,
            use_gpu=gpu,
        )
        console.print("\n[bold green] Training completed successfully![/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red] Training failed: {e}[/bold red]\n")
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--episodes",
    "-e",
    type=int,
    default=10,
    help="Number of episodes to evaluate",
)
@click.option(
    "--render/--no-render",
    default=False,
    help="Render environment during evaluation",
)
@click.option(
    "--save-results/--no-save-results",
    default=True,
    help="Save evaluation results",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="results/evaluation",
    help="Output directory for results",
)
def evaluate(
    model_path: str,
    episodes: int,
    render: bool,
    save_results: bool,
    output_dir: str,
):
    """Evaluate a trained agent."""
    from scripts.evaluate_performance import main as eval_main
    
    console.print("\n[bold blue] Starting Evaluation[/bold blue]\n")
    
    logger.info(f"Evaluating model: {model_path}")
    
    try:
        eval_main(
            model_path=model_path,
            n_episodes=episodes,
            render=render,
            save_results=save_results,
            output_dir=output_dir,
        )
        console.print("\n[bold green] Evaluation completed![/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red] Evaluation failed: {e}[/bold red]\n")
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--paper-trading/--live-trading",
    default=True,
    help="Use paper trading mode",
)
@click.option(
    "--duration",
    "-d",
    type=int,
    help="Trading duration in hours (optional)",
)
@click.option(
    "--initial-capital",
    type=float,
    default=100_000.0,
    help="Initial capital",
)
@click.option(
    "--log-trades/--no-log-trades",
    default=True,
    help="Log all trades",
)
def live(
    model_path: str,
    paper_trading: bool,
    duration: Optional[int],
    initial_capital: float,
    log_trades: bool,
):
    """Run live trading simulation."""
    from scripts.live_trading import main as live_main
    
    mode = "Paper Trading" if paper_trading else "LIVE TRADING"
    console.print(f"\n[bold yellow] Starting {mode}[/bold yellow]\n")
    
    if not paper_trading:
        confirm = click.confirm(
            "️  You are about to start LIVE TRADING with real money. Continue?",
            abort=True,
        )
    
    logger.info(f"Starting live trading: {model_path}")
    
    try:
        live_main(
            model_path=model_path,
            paper_trading=paper_trading,
            duration_hours=duration,
            initial_capital=initial_capital,
            log_trades=log_trades,
        )
        console.print("\n[bold green] Trading session completed![/bold green]\n")
    except KeyboardInterrupt:
        console.print("\n[bold yellow]️  Trading interrupted by user[/bold yellow]\n")
        logger.warning("Trading interrupted by user")
    except Exception as e:
        console.print(f"\n[bold red] Trading failed: {e}[/bold red]\n")
        logger.error(f"Trading failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--check-data/--no-check-data",
    default=True,
    help="Check data availability",
)
@click.option(
    "--check-models/--no-check-models",
    default=True,
    help="Check trained models",
)
@click.option(
    "--check-config/--no-check-config",
    default=True,
    help="Check configuration",
)
def status(check_data: bool, check_models: bool, check_config: bool):
    """Check project status and health."""
    console.print("\n[bold cyan] Project Status[/bold cyan]\n")
    
    from pathlib import Path
    
    # Check data
    if check_data:
        data_dir = Path("data")
        table = Table(title="Data Status")
        table.add_column("Directory", style="cyan")
        table.add_column("Files", style="magenta")
        table.add_column("Status", style="green")
        
        for subdir in ["raw", "processed", "synthetic"]:
            path = data_dir / subdir
            if path.exists():
                num_files = len(list(path.glob("*")))
                status = "" if num_files > 0 else "️ Empty"
                table.add_row(subdir, str(num_files), status)
            else:
                table.add_row(subdir, "0", " Missing")
        
        console.print(table)
        console.print()
    
    # Check models
    if check_models:
        models_dir = Path("results/models")
        table = Table(title="Models Status")
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="magenta")
        table.add_column("Modified", style="green")
        
        if models_dir.exists():
            for model_file in models_dir.glob("*.zip"):
                size = model_file.stat().st_size / 1024 / 1024  # MB
                mtime = model_file.stat().st_mtime
                from datetime import datetime
                mod_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                table.add_row(model_file.name, f"{size:.2f} MB", mod_time)
        
        console.print(table)
        console.print()
    
    # Check configuration
    if check_config:
        console.print("[bold]Configuration:[/bold]")
        console.print("  Config file: config/settings.py")
        console.print("  Environment: .env")
        console.print()


@cli.command()
def info():
    """Display system and package information."""
    console.print("\n[bold cyan]ℹ️  System Information[/bold cyan]\n")
    
    import platform
    import sys
    
    table = Table()
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="magenta")
    
    table.add_row("Python", platform.python_version())
    table.add_row("Platform", platform.platform())
    table.add_row("Architecture", platform.machine())
    
    try:
        import gymnasium
        table.add_row("Gymnasium", gymnasium.__version__)
    except ImportError:
        table.add_row("Gymnasium", "Not installed")
    
    try:
        import stable_baselines3
        table.add_row("Stable-Baselines3", stable_baselines3.__version__)
    except ImportError:
        table.add_row("Stable-Baselines3", "Not installed")
    
    try:
        import torch
        table.add_row("PyTorch", torch.__version__)
        table.add_row("CUDA Available", "Yes" if torch.cuda.is_available() else "No")
    except ImportError:
        table.add_row("PyTorch", "Not installed")
    
    console.print(table)
    console.print()


if __name__ == "__main__":
    cli()
