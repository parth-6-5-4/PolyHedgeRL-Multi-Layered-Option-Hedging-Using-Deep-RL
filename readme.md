# PolyHedgeRL: Multi-Layered Option Hedging Using Deep Reinforcement Learning

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)
![Tests](https://img.shields.io/badge/tests-pytest-yellow.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

**A production-ready deep reinforcement learning framework for hedging complex derivative portfolios**

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Results](#results) • [Contributing](docs/CONTRIBUTING.md)

</div>

---

## Overview

PolyHedgeRL is a sophisticated framework that leverages deep reinforcement learning (Proximal Policy Optimization) to create optimal hedging strategies for multi-asset option portfolios. The system handles 21 correlated derivative instruments simultaneously, making it suitable for real-world trading scenarios.

### Key Highlights

- **Multi-Asset Environment**: Gymnasium-compliant RL environment for 21 option instruments
- **Advanced RL Agent**: PPO implementation with custom reward shaping
- **Comprehensive Backtesting**: Walk-forward validation with regime analysis
- **Production Ready**: Full logging, testing, CI/CD, and deployment setup
- **Professional Code**: Type hints, docstrings, and 80%+ test coverage
- **Experiment Tracking**: MLflow and Weights & Biases integration

---

## Features

### Core Functionality
- **Custom Gymnasium Environment** for multi-asset option hedging
- **PPO Agent** with hyperparameter optimization
- **Black-Scholes Pricing** with Greeks calculation
- **Market Regime Detection** (Bull/Bear/Sideways)
- **Walk-Forward Validation** for realistic backtesting
- **Risk Metrics**: Sharpe, Max Drawdown, VaR, CVaR

### Professional Tools
- **CLI Interface** for training and evaluation
- **Web Dashboard** for real-time monitoring
- **Docker Support** for easy deployment
- **CI/CD Pipeline** with GitHub Actions
- **Comprehensive Documentation** with Sphinx
- **Full Test Suite** with pytest

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA for GPU acceleration

### Quick Install

```bash
# Clone the repository
git clone https://github.com/parth-6-5-4/PolyHedgeRL.git
cd PolyHedgeRL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# (Optional) Install development tools
pip install -r requirements-dev.txt
```

### Docker Installation

```bash
# Build the Docker image
docker-compose build

# Run the container
docker-compose up
```

---

## Quick Start

### 1. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys and settings
nano .env
```

### 2. Train an Agent

```python
from polyhedgerl import MultiAsset21DeepHedgingEnv, PPOAgent, get_config
from polyhedgerl.utils import setup_logger

# Setup logging
logger = setup_logger("training", level="INFO")

# Load configuration
config = get_config()

# Create environment
env = MultiAsset21DeepHedgingEnv(
    market_data=df,
    option_chain=opt_chain,
    asset_universe=asset_universe
)

# Train agent
agent = PPOAgent(env, **config['ppo'])
agent.train(total_timesteps=100_000)
agent.save("results/models/ppo_agent.zip")
```

### 3. Using the CLI

```bash
# Train a new agent
python -m polyhedgerl train --config config/default.yaml --timesteps 100000

# Evaluate trained agent
python -m polyhedgerl evaluate --model results/models/ppo_agent.zip

# Run live trading simulation
python -m polyhedgerl live --model results/models/ppo_agent.zip --paper-trading
```

### 4. Run Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Navigate to notebooks/ and run:
# - 01_Data_Exploration.ipynb
# - 02_Environment_Development.ipynb
# - 03_Agent_Training.ipynb
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PolyHedgeRL System                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │ Market Data  │───▶│ Environment  │◀──▶│ PPO Agent│  │
│  │  (Yahoo/AV)  │    │  (Gymnasium) │    │ (SB3)    │  │
│  └──────────────┘    └──────────────┘    └──────────┘  │
│         │                    │                  │        │
│         ▼                    ▼                  ▼        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │Option Pricing│    │ State Space  │    │ Training │  │
│  │ (BS Model)   │    │ (126 dims)   │    │ Pipeline │  │
│  └──────────────┘    └──────────────┘    └──────────┘  │
│         │                    │                  │        │
│         ▼                    ▼                  ▼        │
│  ┌──────────────────────────────────────────────────┐  │
│  │          Evaluation & Backtesting               │  │
│  │  • Walk-Forward Validation                      │  │
│  │  • Regime Analysis                              │  │
│  │  • Risk Metrics                                 │  │
│  └──────────────────────────────────────────────────┘  │
│                           │                             │
│                           ▼                             │
│                  ┌─────────────────┐                    │
│                  │ Results & Plots │                    │
│                  └─────────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

### Environment Details

**State Space** (126 dimensions):
- 21 option positions
- 21 option prices
- 21 option deltas
- 21 option gammas
- 21 option vegas
- Spot price, volatility, time to expiry

**Action Space**:
- Box(-1, 1, shape=(21,)) - Continuous actions for each option

**Reward Function**:
- PnL maximization
- Transaction cost penalty
- Risk penalty (volatility)

---

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 1.85 |
| Max Drawdown | -12.3% |
| Win Rate | 64.2% |
| Annualized Return | 23.5% |
| VaR (95%) | -2.1% |

### Regime Performance

| Regime | Sharpe | Return | Max DD |
|--------|--------|--------|--------|
| Bull | 2.12 | +28.3% | -8.1% |
| Bear | 1.54 | +18.7% | -15.2% |
| Sideways | 1.71 | +20.1% | -10.5% |

*See `DEMO_RESULTS.md` for detailed results and visualizations*

---

## Project Structure

```
PolyHedgeRL/
├── src/
│   ├── agents/           # RL agents (PPO, evaluation)
│   ├── environment/      # Gymnasium environment, pricing
│   ├── config/           # Configuration management
│   └── utils/            # Utilities, logging, data tools
├── notebooks/            # Jupyter notebooks (8 analysis notebooks)
├── scripts/              # Training, evaluation, live trading
├── tests/                # Unit and integration tests
├── data/                 # Market data (raw, processed, synthetic)
├── results/              # Models, plots, reports, logs
├── docs/                 # Sphinx documentation
├── docker/               # Docker configuration
└── .github/              # CI/CD workflows
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_environment.py

# Run with verbose output
pytest -v -s
```

---

## Documentation

### User Guide
- [Installation Guide](docs/guides/QUICKSTART.md)
- [Quick Start Tutorial](docs/guides/QUICKSTART.md)
- [PDF Export Guide](docs/guides/PDF_EXPORT_README.md)
- [API Reference](docs/)

### Developer Guide
- [Contributing Guidelines](docs/CONTRIBUTING.md)
- [Code of Conduct](docs/CONTRIBUTING.md#code-of-conduct)
- [Architecture Overview](docs/README_PROFESSIONAL.md)
- [Development Setup](docs/CONTRIBUTING.md#development-setup)

### Research Papers
- [Deep Hedging Theory](docs/papers/deep_hedging.md)
- [PPO for Finance](docs/papers/ppo_finance.md)

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Format code (`black . && isort .`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Parth Dambhare**
- GitHub: [@parth-6-5-4](https://github.com/parth-6-5-4)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/parth-dambhare)


---

## Acknowledgments

- **Stable-Baselines3** for RL implementations
- **Gymnasium** for environment framework
- **Black-Scholes** option pricing model
- Research papers on deep hedging and reinforcement learning

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{polyhedgerl2024,
  author = {Dambhare, Parth},
  title = {PolyHedgeRL: Multi-Layered Option Hedging Using Deep RL},
  year = {2024},
  url = {https://github.com/parth-6-5-4/PolyHedgeRL}
}
```

---

## Related Projects

- [DeepHedging](https://github.com/hansbuehler/deephedging) - Original deep hedging implementation
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL) - RL for financial applications
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms

---

<div align="center">

**Star this repository if you find it helpful!**

Made by [Parth Dambhare](https://github.com/parth-6-5-4)

</div>
