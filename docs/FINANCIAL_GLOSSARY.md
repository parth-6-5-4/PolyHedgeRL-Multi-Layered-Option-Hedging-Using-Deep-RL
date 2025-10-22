# Financial Terms Glossary
## PolyHedgeRL Reference Guide

**Purpose**: Quick reference for all financial and technical terms used in the PolyHedgeRL project.

---

## Portfolio Metrics

### Sharpe Ratio
**Definition**: Risk-adjusted return metric  
**Formula**: `(Mean Return - Risk-Free Rate) / Standard Deviation of Returns`  
**Range**: -∞ to +∞ (typically -1 to 3)  
**Good Value**: > 1.0 (professional grade: > 1.5)  
**Example**: 20% return with 10% volatility = Sharpe of 2.0 (excellent)  
**Why It Matters**: Tells you return per unit of risk taken

### Sortino Ratio
**Definition**: Like Sharpe, but only penalizes downside volatility  
**Formula**: `(Mean Return - Risk-Free Rate) / Downside Deviation`  
**Difference from Sharpe**: Ignores upside volatility (which investors like)  
**Good Value**: > 1.5  
**Why It Matters**: More relevant for investors (we want upside volatility!)

### Maximum Drawdown (Max DD)
**Definition**: Largest peak-to-valley decline in portfolio value  
**Formula**: `max((Peak Value - Current Value) / Peak Value)`  
**Range**: 0% to 100%  
**Good Value**: < 15% (professional), < 20% (acceptable)  
**Example**: Peak at $150k, fall to $120k = 20% drawdown  
**Why It Matters**: Measures worst-case pain an investor experiences

### Calmar Ratio
**Definition**: Return per unit of drawdown risk  
**Formula**: `Annualized Return / Maximum Drawdown`  
**Good Value**: > 1.0 (return exceeds worst drawdown)  
**Example**: 20% return, 10% max DD = Calmar of 2.0  
**Why It Matters**: Balances greed (returns) with fear (losses)

### Annualized Return
**Definition**: Compound annual growth rate (CAGR)  
**Formula**: `(Final Value / Initial Value)^(1/years) - 1`  
**Benchmarks**: S&P 500 = ~10%, Good fund = 15%+  
**Example**: $100k → $150k in 3 years = 14.5% annualized  
**Why It Matters**: Standard measure for comparing investments

### Win Rate
**Definition**: Percentage of profitable periods  
**Formula**: `Winning Days / Total Days × 100%`  
**Range**: 0% to 100%  
**Good Value**: > 55% (> 60% is excellent)  
**Note**: High win rate ≠ profitable (one huge loss can wipe out many wins)  
**Why It Matters**: Measures consistency

### Alpha
**Definition**: Excess return vs. benchmark  
**Formula**: `Portfolio Return - Beta × Market Return`  
**Good Value**: > 0 (positive alpha = beating market)  
**Example**: If S&P 500 returns 10% and you return 15% with beta=1.0, alpha = 5%  
**Why It Matters**: Measures skill/edge vs. just market exposure

### Beta
**Definition**: Correlation/sensitivity to market moves  
**Formula**: `Covariance(Portfolio, Market) / Variance(Market)`  
**Range**: -∞ to +∞ (typically -1 to 2)  
**Values**: 
- β = 1.0: Moves with market
- β = 0.5: Half as volatile as market
- β = 0.0: Market-neutral
- β = -0.5: Inverse to market  
**Why It Matters**: Measures diversification benefit

### Volatility (Standard Deviation)
**Definition**: How much returns fluctuate  
**Formula**: `std(returns) × sqrt(252)` (annualized)  
**Benchmarks**: S&P 500 = ~18%, Bonds = ~5%  
**Good Value**: Depends on goals (lower if risk-averse)  
**Why It Matters**: Measures uncertainty/risk

### Value at Risk (VaR)
**Definition**: Maximum expected loss at confidence level  
**Example**: "95% VaR = $10k" means 95% of days lose < $10k  
**Confidence Levels**: Typically 95% or 99%  
**Limitation**: Doesn't capture tail risk beyond threshold  
**Why It Matters**: Regulatory risk reporting

### Herfindahl Index (HHI)
**Definition**: Concentration measure  
**Formula**: `Σ(weight_i)²`  
**Range**: 0 (diversified) to 1.0 (concentrated)  
**Example**: 
- Equal 20 positions: HHI = 0.05
- All in one: HHI = 1.0  
**Good Value**: < 0.10 (well diversified)  
**Why It Matters**: Diversification reduces portfolio-specific risk

---

## Option Terminology

### Call Option
**Definition**: Right (not obligation) to BUY underlying at strike price  
**Profit When**: Underlying price goes UP  
**Maximum Loss**: Premium paid (if long)  
**Example**: Buy $100 call for $5, stock goes to $110 → profit $5

### Put Option
**Definition**: Right (not obligation) to SELL underlying at strike price  
**Profit When**: Underlying price goes DOWN  
**Maximum Loss**: Premium paid (if long)  
**Example**: Buy $100 put for $5, stock goes to $90 → profit $5

### Strike Price (K)
**Definition**: The price at which option can be exercised  
**Example**: "$100 call" means right to buy at $100  
**At-the-Money (ATM)**: Strike ≈ Current Price  
**In-the-Money (ITM)**: Strike favorable (call: K < S, put: K > S)  
**Out-of-the-Money (OTM)**: Strike unfavorable

### Expiration (Expiry)
**Definition**: Date when option contract ends  
**Formats**: "30-day", "60-day", "Jan 2026"  
**Time Decay**: Options lose value as expiration approaches  
**Note**: American options exercisable anytime, European only at expiry

### Premium
**Definition**: Price paid to buy an option  
**Components**: Intrinsic Value + Time Value  
**Factors**: Strike, expiry, volatility, interest rates, dividends

### Intrinsic Value
**Definition**: Immediate exercise value  
**Call**: `max(Spot - Strike, 0)`  
**Put**: `max(Strike - Spot, 0)`  
**Example**: Stock at $110, $100 call has $10 intrinsic value

### Time Value (Extrinsic Value)
**Definition**: Premium above intrinsic value  
**Formula**: `Option Price - Intrinsic Value`  
**Cause**: Optionality (chance to profit before expiry)  
**Decay**: Accelerates as expiration approaches

### Moneyness
**Definition**: Relationship between strike and spot price  
**Formula**: `Strike / Spot Price`  
**Values**:
- 0.90: Deep ITM call / Deep OTM put
- 0.95: ITM call / OTM put
- 1.00: At-the-money (ATM)
- 1.05: OTM call / ITM put
- 1.10: Deep OTM call / Deep ITM put

---

## The Greeks

### Delta (Δ)
**Definition**: Change in option price per $1 change in underlying  
**Formula**: `∂Option_Price / ∂Spot_Price`  
**Range**: 
- Calls: 0 to +1
- Puts: -1 to 0  
**Example**: Delta = 0.50 call → stock up $1 → option up $0.50  
**At-the-Money**: ~0.50 (50% chance of ending ITM)  
**Uses**: Position sizing, hedging, probability estimation

### Gamma (Γ)
**Definition**: Change in delta per $1 change in underlying  
**Formula**: `∂Delta / ∂Spot_Price = ∂²Option_Price / ∂Spot_Price²`  
**Range**: 0 to ~0.10 (highest ATM)  
**Example**: Gamma = 0.05 → stock up $1 → delta increases by 0.05  
**Meaning**: 
- High gamma: Delta changes quickly (risk!)
- Low gamma: Delta stable  
**Why It Matters**: Gamma scalping, risk management

### Theta (Θ)
**Definition**: Change in option price per day (time decay)  
**Formula**: `∂Option_Price / ∂Time`  
**Sign**: Negative for long options (lose value over time)  
**Typical Values**: -$0.05 to -$0.15 per day  
**Accelerates**: As expiration approaches  
**Example**: Theta = -$0.10 → option loses $0.10 in value overnight  
**Strategy**: 
- Sell options → collect theta
- Buy options → fight theta

### Vega (ν)
**Definition**: Change in option price per 1% change in implied volatility  
**Formula**: `∂Option_Price / ∂Implied_Volatility`  
**Sign**: Positive for long options  
**Typical Values**: $0.05 to $0.30 per 1% IV change  
**Example**: Vega = $0.20, IV rises 10% → option up $2.00  
**Strategy**:
- Long options → long vega → profit from volatility spikes
- Short options → short vega → profit from calm markets

### Rho (ρ)
**Definition**: Change in option price per 1% change in interest rate  
**Formula**: `∂Option_Price / ∂Interest_Rate`  
**Impact**: Usually small (< $0.05)  
**Note**: More relevant for long-dated options  
**Why It Matters Less**: Interest rates change slowly

---

## Volatility Concepts

### Historical Volatility (HV)
**Definition**: Actual past price fluctuations  
**Formula**: `std(returns) × sqrt(252)`  
**Calculation**: Rolling window (e.g., 20-day, 60-day)  
**Uses**: Compare to implied volatility, risk measurement

### Implied Volatility (IV)
**Definition**: Market's expectation of future volatility  
**Source**: Derived from option prices (inverse Black-Scholes)  
**Normal Range**: 15-25% for S&P 500  
**High**: > 30% (fear, uncertainty)  
**Low**: < 15% (complacency)  
**Note**: Forward-looking, not historical

### VIX (Volatility Index)
**Definition**: S&P 500 30-day implied volatility  
**Nickname**: "Fear gauge"  
**Normal**: 12-20  
**High**: > 30 (nervous market)  
**Crisis**: > 50 (panic)  
**Record**: 82 (2008 crisis), 85 (2020 COVID)  
**Use**: Market sentiment, volatility trading

### IV Rank
**Definition**: Where current IV sits in 52-week range  
**Formula**: `(Current IV - IV_min) / (IV_max - IV_min) × 100`  
**Range**: 0% to 100%  
**Interpretation**:
- < 25%: Low (sell options more expensive)
- 25-50%: Average
- 50-75%: Elevated  
- \> 75%: High (buy options cheaper)  
**Use**: Relative value, strategy selection

### Volatility Smile/Skew
**Definition**: Pattern where IV varies by strike  
**Observation**: OTM puts often have higher IV than ATM  
**Cause**: Demand for downside protection (insurance)  
**Implication**: Market prices tail risk

---

## Option Strategies (Classical)

### Covered Call
**Setup**: Own stock + sell OTM call  
**Profit**: Stock appreciation (up to strike) + premium  
**Loss**: Unlimited downside (minus premium)  
**View**: Bullish to neutral  
**Use**: Income generation, slight upside

### Protective Put
**Setup**: Own stock + buy OTM put  
**Profit**: Unlimited upside (minus premium)  
**Loss**: Limited downside (put strike minus premium)  
**View**: Bullish with insurance  
**Use**: Downside protection

### Straddle (Long)
**Setup**: Buy ATM call + ATM put  
**Profit**: Large move in EITHER direction  
**Loss**: Limited to premiums paid  
**View**: Neutral (expect volatility)  
**Use**: Earnings, events, volatility plays

### Strangle (Long)
**Setup**: Buy OTM call + OTM put  
**Profit**: Large move (needs bigger move than straddle)  
**Loss**: Limited to premiums paid  
**View**: Neutral (cheaper than straddle)  
**Use**: Event plays with limited capital

### Iron Condor
**Setup**: Sell OTM call spread + sell OTM put spread  
**Profit**: Market stays in range (collect premiums)  
**Loss**: Limited (width of spread minus premiums)  
**View**: Neutral (low volatility expected)  
**Use**: Income in range-bound markets

### Butterfly Spread
**Setup**: Buy 1 low strike, sell 2 mid strikes, buy 1 high strike  
**Profit**: Underlying lands at middle strike  
**Loss**: Limited to net premium  
**View**: Neutral with specific target  
**Use**: Low-cost directional bet

### Calendar Spread
**Setup**: Sell near-term option, buy longer-term option (same strike)  
**Profit**: From time decay difference  
**Loss**: Limited  
**View**: Neutral, long volatility  
**Use**: Volatility trading, theta capture

---

## Reinforcement Learning Terms

### Agent
**Definition**: The AI that learns and makes decisions  
**In PolyHedgeRL**: The RecurrentPPO model  
**Actions**: Option position sizes  
**Goal**: Maximize cumulative reward

### Environment
**Definition**: The world the agent interacts with  
**In PolyHedgeRL**: Multi-asset option market simulation  
**State**: Market prices, portfolio, risk metrics  
**Reward**: Based on returns, Sharpe, drawdowns

### State (Observation)
**Definition**: What the agent sees  
**Components**: 
- Market data (prices, volume, volatility)
- Option data (strikes, Greeks, prices)
- Portfolio data (positions, P&L, risk)  
**Dimension**: 60+ features

### Action
**Definition**: What the agent does  
**In PolyHedgeRL**: 20-dim vector (position size per option)  
**Range**: -1.0 (max short) to +1.0 (max long)  
**Continuous**: Not discrete buy/sell, but sizing

### Reward
**Definition**: Feedback signal for learning  
**Components**: Returns, Sharpe, drawdown, costs  
**Goal**: Maximize cumulative discounted reward  
**Note**: Reward shaping is critical for success

### Policy (π)
**Definition**: Agent's strategy (state → action mapping)  
**Notation**: `π(a|s)` = probability of action a given state s  
**In PolyHedgeRL**: LSTM neural network  
**Learning**: Adjusts policy to maximize expected reward

### Value Function (V)
**Definition**: Expected future reward from state  
**Formula**: `V(s) = E[Σ γ^t r_t | s_0 = s]`  
**Use**: Estimates how good a state is  
**Critic**: Neural network that learns V

### Q-Function (Q)
**Definition**: Expected future reward from state-action pair  
**Formula**: `Q(s,a) = E[Σ γ^t r_t | s_0 = s, a_0 = a]`  
**Difference from V**: Conditions on specific action  
**Note**: Not used directly in PPO (policy-based)

### Advantage (A)
**Definition**: How much better action is than expected  
**Formula**: `A(s,a) = Q(s,a) - V(s)`  
**Positive**: Action better than average  
**Negative**: Action worse than average  
**Use**: Guides policy updates

### Discount Factor (γ)
**Definition**: How much to value future rewards  
**Range**: 0 to 1  
**Values**:
- γ = 0.99: Horizon ~100 steps
- γ = 0.995: Horizon ~200 steps (our choice)
- γ = 1.0: Infinite horizon (impractical)  
**Why High for Finance**: Options are long-term instruments

### Episode
**Definition**: One complete trajectory through environment  
**In PolyHedgeRL**: 252 timesteps (1 trading year)  
**Start**: Initial state (t=0)  
**End**: Terminal state (t=252) or portfolio blowup

### Timestep
**Definition**: One interaction (observe, act, receive reward)  
**In PolyHedgeRL**: 1 timestep = 1 trading day  
**Training**: 300,000 timesteps = ~1,190 episodes (years)

### Rollout
**Definition**: Collecting experience by running policy  
**In PPO**: Collect n_steps (2,048) before updating  
**Purpose**: Gather data for policy improvement

### Entropy
**Definition**: Randomness in policy  
**High Entropy**: Explores (tries different actions)  
**Low Entropy**: Exploits (deterministic, best known action)  
**Schedule**: Start high (explore), end low (exploit)  
**In PolyHedgeRL**: 0.005 (mostly deterministic)

### On-Policy vs Off-Policy
**On-Policy**: Learns from current policy's experience (PPO)  
**Off-Policy**: Learns from any past experience (Q-learning)  
**Trade-off**: On-policy more stable, off-policy more sample efficient

---

## Model Architecture Terms

### LSTM (Long Short-Term Memory)
**Type**: Recurrent neural network  
**Purpose**: Remembers long sequences  
**Components**: 
- Forget gate (what to discard)
- Input gate (what to store)
- Output gate (what to reveal)  
**In PolyHedgeRL**: 256 units × 2 layers  
**Why**: Financial time series have temporal dependencies

### Hidden State
**Definition**: LSTM's memory of past inputs  
**Dimension**: 256 in our model  
**Update**: At each timestep based on new observation  
**Use**: Carries context (yesterday's volatility, trends, etc.)

### Actor (Policy Network)
**Definition**: Neural network that outputs actions  
**Input**: State + hidden state  
**Output**: 20-dim action vector  
**Loss**: Policy gradient (maximize expected reward)

### Critic (Value Network)
**Definition**: Neural network that estimates state value  
**Input**: State + hidden state  
**Output**: Scalar value V(s)  
**Loss**: Mean squared error vs. actual returns  
**Use**: Baseline for advantage estimation

### Policy Gradient
**Definition**: RL algorithm that directly optimizes policy  
**Idea**: Increase probability of actions that led to high reward  
**Formula**: `∇_θ J(θ) = E[∇_θ log π_θ(a|s) A(s,a)]`  
**Vanilla PG**: High variance (unstable)  
**PPO**: Clipped version (stable)

### Proximal Policy Optimization (PPO)
**Type**: On-policy actor-critic algorithm  
**Key Innovation**: Clipped objective (prevents large policy changes)  
**Clip Range**: ±15% in our model  
**Advantages**: 
- Stable (won't catastrophically fail)
- Sample efficient (reuses data via mini-batches)
- Robust (works on many tasks)  
**Why for Finance**: Stability critical (can't risk blowing up)

### Generalized Advantage Estimation (GAE)
**Purpose**: Better advantage estimates (reduces variance)  
**Formula**: `A_t^GAE = Σ (γλ)^l δ_{t+l}`  
**Parameters**: 
- λ = 0.98 (bias-variance tradeoff)
- γ = 0.995 (discount)  
**Why**: More accurate gradient estimates → faster learning

---

## Training Concepts

### Hyperparameters
**Definition**: Settings that control learning process  
**Examples**: Learning rate, batch size, clip range  
**Not Learned**: Set before training (vs. model parameters)  
**Critical**: Huge impact on performance

### Learning Rate
**Definition**: Step size for parameter updates  
**In PolyHedgeRL**: 3×10⁻⁵ (very small)  
**Too High**: Unstable, diverges  
**Too Low**: Learns too slowly  
**Schedule**: Often decay over time

### Batch Size
**Definition**: Number of samples per gradient update  
**In PolyHedgeRL**: 128  
**Trade-off**: 
- Larger: More stable gradients, more RAM
- Smaller: Noisier gradients, less RAM  
**For LSTM**: Larger batches help

### Epoch
**Definition**: One pass through collected rollout data  
**In PolyHedgeRL**: 20 epochs  
**Why Multiple**: Squeeze more learning from expensive rollouts  
**Risk**: Overfitting to recent experience

### Gradient Clipping
**Definition**: Limit gradient magnitude (prevent explosion)  
**In PolyHedgeRL**: Max norm = 0.5  
**Why**: RNNs prone to exploding/vanishing gradients  
**Effect**: Prevents catastrophic updates

### Checkpoint
**Definition**: Saved model snapshot  
**Frequency**: Every 20,000 steps in our training  
**Use**: 
- Resume training if interrupted
- Evaluate intermediate models
- Rollback if performance degrades

### Tensorboard
**Definition**: Visualization tool for training metrics  
**Tracks**: Loss, rewards, learning rate, etc.  
**In PolyHedgeRL**: `results/models_improved/tensorboard/`  
**View**: `tensorboard --logdir [path]`

### Overfitting
**Definition**: Model memorizes training data, fails on new data  
**Symptoms**: Great backtest, poor live performance  
**Prevention**: 
- Out-of-sample testing
- Regularization (dropout, entropy)
- Early stopping  
**Critical in Finance**: Market patterns change

### Out-of-Sample Testing
**Definition**: Evaluate on data not used in training  
**Example**: Train on 2004-2023, test on 2024-2025  
**Gold Standard**: Walk-forward analysis  
**Purpose**: Detect overfitting

### Walk-Forward Analysis
**Definition**: Rolling out-of-sample testing  
**Process**: 
1. Train on years 1-5
2. Test on year 6
3. Train on years 2-6
4. Test on year 7
5. Repeat...  
**Most Realistic**: Simulates live deployment

---

## Market Regimes

### Bull Market
**Definition**: Rising prices, positive sentiment  
**Characteristics**: Low volatility, high confidence  
**Duration**: Months to years  
**Examples**: 2009-2019, 2020-2021  
**Strategy**: Long calls, covered calls

### Bear Market
**Definition**: Falling prices, negative sentiment  
**Characteristics**: High volatility, fear  
**Duration**: Weeks to months  
**Definition (Technical)**: -20% from peak  
**Examples**: 2008, 2020 Q1, 2022  
**Strategy**: Long puts, short calls

### Sideways (Range-Bound) Market
**Definition**: No clear trend, oscillates in range  
**Characteristics**: Low volatility, indecision  
**Duration**: Weeks to months  
**Strategy**: Iron condors, short straddles

### High Volatility Regime
**Definition**: Large daily swings (VIX > 30)  
**Causes**: Uncertainty, fear, crisis  
**Options**: More expensive (high IV)  
**Strategy**: Long options (bet on continued volatility)

### Low Volatility Regime
**Definition**: Small daily swings (VIX < 15)  
**Causes**: Calm, complacency, stability  
**Options**: Cheaper (low IV)  
**Strategy**: Short options (collect premium)

### Crisis/Crash
**Definition**: Rapid, severe decline  
**Characteristics**: Extreme volatility, panic  
**Examples**: 2008 GFC (-50%), 2020 COVID (-34%)  
**Options**: Massive IV spike, puts very expensive  
**Strategy**: Cash preservation, defensive positioning

---

## Risk Management

### Position Sizing
**Definition**: How much capital to allocate per trade  
**In PolyHedgeRL**: Maximum 20% per option  
**Methods**: 
- Fixed $ amount
- Fixed % of portfolio
- Kelly criterion  
**Goal**: Balance growth and survival

### Stop-Loss
**Definition**: Automatic exit at loss threshold  
**In PolyHedgeRL**: -5% daily, -20% total  
**Purpose**: Limit catastrophic losses  
**Trade-off**: May exit before recovery

### Diversification
**Definition**: Spreading risk across uncorrelated assets  
**In PolyHedgeRL**: 20 options, HHI < 0.15  
**Benefit**: Reduces portfolio-specific risk  
**Limit**: Can't diversify systematic (market) risk

### Correlation
**Definition**: How assets move together  
**Range**: -1 (inverse) to +1 (same direction)  
**Ideal**: Low or negative correlation  
**Reality**: Correlations spike in crashes (everything down)

### Leverage
**Definition**: Controlling more capital than you have  
**In Options**: Built-in (control $10k stock with $500 option)  
**Benefit**: Amplified returns  
**Risk**: Amplified losses  
**In PolyHedgeRL**: Implicit via option positions

### Circuit Breaker
**Definition**: Automatic trading halt on extreme moves  
**In PolyHedgeRL**: Stop trading if -5% daily loss  
**Purpose**: Prevent emotional decisions, limit losses  
**Resume**: Next day or manual override

---

## Performance Benchmarks

### S&P 500 (Benchmark)
**Historic Return**: ~10% annualized  
**Historic Volatility**: ~18%  
**Sharpe**: ~0.6  
**Use**: Standard benchmark for US equities

### Risk-Free Rate
**Definition**: Return on "safe" investment (US Treasury)  
**Current**: ~4.5% (10-year Treasury)  
**Use**: Baseline in Sharpe ratio, cost of capital

### Institutional Performance
**Mutual Funds**: 7-8% after fees  
**Hedge Funds**: 8-12% average  
**Top Quant Funds**: 15-25%  
**Our Target**: 15-20% (top tier)

---

**Last Updated**: October 20, 2025  
**Version**: 1.0  
**Use**: Reference while reading technical documentation
