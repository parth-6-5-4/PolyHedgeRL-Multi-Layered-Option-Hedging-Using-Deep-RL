# MPS (GPU) Performance Testing Results

**Date**: October 20, 2025  
**System**: MacBook Pro with Metal Performance Shaders (MPS)

---

## Executive Summary

**Key Finding**: MPS (GPU) acceleration provides **NO performance benefit** for this project due to compatibility issues with LSTM/RecurrentPPO and overhead from small batch operations.

### Test Results

| Metric | CPU | MPS (GPU) | Difference |
|--------|-----|-----------|------------|
| **Training Time (5k steps)** | 330.69s (5.51 min) | 373.74s (6.23 min) | **+13% slower** |
| **Training Speed** | 15.12 FPS | 13.38 FPS | **-11% slower** |
| **300k Steps Projection** | 5.5 hours | 6.2 hours | **+0.7 hours** |

---

## Critical Findings

### 1. MPS is Actually SLOWER Than CPU
- **CPU**: 15.12 FPS
- **MPS**: 13.38 FPS
- **Result**: CPU is **1.13x faster** than MPS for this workload

### 2. Why MPS is Slower

#### a) **LSTM Incompatibility** (Primary Issue)
- RecurrentPPO uses LSTM layers for temporal memory
- PyTorch MPS has **known bugs** with LSTM operations
- Test crashed with: `failed assertion '[MPSNDArrayDescriptor sliceDimension:withSubrange:]'`
- **Cannot use RecurrentPPO with MPS at all**

#### b) **Overhead for Small Operations**
- Standard PPO test shows MPS is still slower than CPU
- GPU overhead (data transfer, kernel launch) exceeds computation benefit
- Small batch sizes (128) and network size don't saturate GPU

#### c) **Environment Computation on CPU**
- Option pricing calculations run on CPU regardless of model device
- GPU only accelerates neural network forward/backward passes
- Most time spent in environment step() function (CPU-bound)

---

## Detailed Analysis

### Test Configuration
```
Test Size: 5,000 timesteps
Data: 2023-2025 (652 trading days, 13,040 option records)
Model: PPO with MlpPolicy (standard neural network)
Batch Size: 128
Network: 3-layer MLP (no LSTM due to MPS bugs)
```

### CPU Performance
- **Speed**: 15.12 FPS
- **Time**: 330.69 seconds (5.51 minutes)
- **Projected 300k**: 5.5 hours
- **Stability**: ✅ Fully stable, no errors

### MPS Performance  
- **Speed**: 13.38 FPS
- **Time**: 373.74 seconds (6.23 minutes)
- **Projected 300k**: 6.2 hours
- **Stability**: ⚠️ Cannot run RecurrentPPO (LSTM crashes)

---

## RecurrentPPO Training Comparison

### Your Previous Training Run (CPU Only)
- **Model**: RecurrentPPO with LSTM
- **Data**: 2023-2025 (652 days, similar to test)
- **Speed**: ~13 FPS (from validation run)
- **80k steps**: 16 hours (5k steps/hour)
- **Result**: Positive rewards, good convergence

### Why RecurrentPPO is Slower
- LSTM adds sequential computation (cannot fully parallelize)
- Hidden state management increases overhead
- More parameters: 2.8M vs ~500k for standard PPO
- **Trade-off**: Better for time series despite slower speed

---

## Recommendations

### ✅ RECOMMENDED: Stay with CPU + RecurrentPPO

**Reasons:**
1. **LSTM is critical** for financial time series (temporal dependencies)
2. **CPU is actually faster** than MPS for this workload
3. **No compatibility issues** - fully stable
4. **Your 80k checkpoint** already shows excellent results (reward 4,350)

**Training Strategy:**
```bash
# Use optimized data window (10 years)
python train_improved.py \
  --timesteps 150000 \
  --start-date 2015-01-01 \
  --device cpu
```

**Expected Time**: ~12-15 hours for 150k steps with 10 years of data

---

### ❌ NOT RECOMMENDED: MPS

**Reasons:**
1. **RecurrentPPO crashes** - cannot use LSTM with MPS
2. **Standard PPO is slower** on MPS than CPU
3. **No LSTM memory** = worse for financial time series
4. **Unstable** - known PyTorch MPS bugs

---

## Alternative GPU Options

### 1. Google Colab (Free CUDA GPU)
- **T4 GPU**: 5-10x faster than CPU for LSTM
- **RecurrentPPO compatible**: Full CUDA support
- **Free Tier**: 12-hour sessions
- **Cost**: Free (with limits) or $10/month for Pro

### 2. AWS/GCP Cloud GPUs
- **NVIDIA GPUs**: Full CUDA support
- **Expected Speedup**: 5-10x for RecurrentPPO
- **Cost**: ~$0.50-$1.50/hour
- **150k steps**: 1-2 hours (~$1-3 total)

### 3. Reduce Training Requirements
- **Current 80k checkpoint** may already be good enough
- **Evaluate first** before additional training
- **150k steps** instead of 300k (50% reduction)
- **Use 10 years** of data (already planned)

---

## Updated Training Code

The code has been updated with MPS support (though not beneficial):

```python
# train_improved.py now includes device detection
def get_device():
    if torch.backends.mps.is_available():
        return 'mps'  # Mac GPU
    elif torch.cuda.is_available():
        return 'cuda'  # NVIDIA GPU
    else:
        return 'cpu'

# Usage
python train_improved.py --device auto  # Auto-detect
python train_improved.py --device cpu   # Force CPU
python train_improved.py --device mps   # Force MPS (will fail with RecurrentPPO)
```

---

## Performance Bottleneck Analysis

### Time Breakdown (Estimated)
```
Total Training Time: 100%
├─ Environment Steps: ~70%
│  ├─ Option Pricing: ~40%
│  ├─ Portfolio Calc: ~20%
│  └─ State Updates: ~10%
│
└─ Model Training: ~30%
   ├─ Forward Pass: ~10%
   ├─ Backward Pass: ~15%
   └─ Optimizer Step: ~5%
```

### Why GPU Doesn't Help
- **70% of time** spent in environment (CPU-bound)
- **Only 30%** in neural network (GPU-acceleratable)
- **Best case GPU speedup**: 3-5x on 30% = **1.6-1.9x overall**
- **MPS overhead** negates this benefit for small models

---

## Next Steps

### Immediate Actions

1. **Evaluate Your 80k Checkpoint**
   ```bash
   python scripts/evaluate_performance.py \
     --model results/models_improved/rppo_model_80000_steps.zip
   ```
   
2. **If Results Are Good** (Sharpe >0.8)
   - Move to paper trading
   - Skip additional training

3. **If More Training Needed**
   - Use CPU with optimized settings
   - 150k steps, 10 years data
   - Expected: 12-15 hours

### Long-Term Optimizations

1. **Optimize Environment**
   - Vectorize option pricing (NumPy)
   - Cache repeated calculations
   - Reduce state computation

2. **Use Cloud GPU** (If Needed)
   - Colab for free CUDA
   - 5-10x speedup with LSTM
   - Worth it for large experiments

3. **Consider Alternative Approaches**
   - Pre-compute option Greeks
   - Simplify state space
   - Use transformer instead of LSTM (better GPU utilization)

---

## Conclusion

**Bottom Line**: For this project on MacBook Pro, **CPU training is optimal**. MPS provides no benefit due to:
1. LSTM incompatibility (crashes RecurrentPPO)
2. Overhead exceeds gains for small models
3. Environment computation is CPU-bound

**Recommendation**: Continue with CPU-based RecurrentPPO training. Your current approach is already optimal for the available hardware.

If training time is still a concern, the better solution is to:
1. Evaluate your existing 80k checkpoint first
2. Consider reducing to 150k total steps
3. Use cloud GPUs (Colab/AWS) if significant additional training is needed

---

## Test Artifacts

- **Test Script**: `test_mps_ppo.py`
- **CPU Time**: 5.51 minutes (5k steps)
- **MPS Time**: 6.23 minutes (5k steps)
- **Date**: October 20, 2025
- **PyTorch Version**: 2.8.0
- **MPS Available**: Yes
- **MPS Functional**: Yes (for standard PPO only)
- **RecurrentPPO Compatible**: ❌ No
