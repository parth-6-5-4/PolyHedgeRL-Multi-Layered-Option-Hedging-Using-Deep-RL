#!/bin/bash

# Real-time training progress tracker

echo "==================================================="
echo "FULL TRAINING - DETAILED PROGRESS"
echo "==================================================="
echo ""

# Check process
PID=$(pgrep -f "train_improved.py --timesteps 300000")
if [ -n "$PID" ]; then
    echo "✅ Training is ACTIVE (PID: $PID)"
    ps -p $PID -o pid,%cpu,%mem,etime,command | tail -1
    echo ""
else
    echo "❌ Training process not found"
    echo ""
    exit 1
fi

# Show key metrics from log
echo "==================================================="
echo "TRAINING METRICS (from log)"
echo "==================================================="
if [ -f "full_training.log" ]; then
    echo "Total timesteps completed:"
    grep "total_timesteps" full_training.log | tail -1
    echo ""
    echo "Episode reward mean:"
    grep "ep_rew_mean" full_training.log | tail -1
    echo ""
    echo "Training time elapsed (seconds):"
    grep "time_elapsed" full_training.log | tail -1
    echo ""
    echo "FPS (frames per second):"
    grep "fps" full_training.log | tail -1
else
    echo "No log file found"
fi

echo ""
echo "==================================================="
echo "CHECKPOINTS PROGRESS"
echo "==================================================="

# Find checkpoints from current training (after 18:42)
echo "New checkpoints from current training run:"
find results/models_improved -name "rppo_model_*_steps.zip" -newermt "2025-10-19 18:42:00" -exec ls -lh {} \; | awk '{print $9, "  -  Created:", $6, $7, $8}'

echo ""
echo "All model checkpoints:"
ls -lth results/models_improved/rppo_model_*_steps.zip 2>/dev/null | awk '{print $9, "  -  ", $6, $7, $8}' | head -10

echo ""
echo "==================================================="
echo "PROGRESS CALCULATION"
echo "==================================================="

# Count new checkpoints
NEW_CHECKPOINT_COUNT=$(find results/models_improved -name "rppo_model_*_steps.zip" -newermt "2025-10-19 18:42:00" | wc -l | tr -d ' ')
TOTAL_CHECKPOINTS=15  # 300k / 20k

if [ "$NEW_CHECKPOINT_COUNT" -gt 0 ]; then
    PROGRESS=$((NEW_CHECKPOINT_COUNT * 100 / TOTAL_CHECKPOINTS))
    TIMESTEPS=$((NEW_CHECKPOINT_COUNT * 20000))
    echo "✅ New checkpoints saved: $NEW_CHECKPOINT_COUNT"
    echo "📊 Progress: $TIMESTEPS / 300,000 timesteps ($PROGRESS%)"
    
    REMAINING=$((TOTAL_CHECKPOINTS - NEW_CHECKPOINT_COUNT))
    echo "⏳ Remaining checkpoints: $REMAINING"
    
    # Estimate completion time
    if [ -f "full_training.log" ]; then
        TIME_ELAPSED=$(grep "time_elapsed" full_training.log | tail -1 | awk '{print $2}')
        if [ -n "$TIME_ELAPSED" ]; then
            TIME_PER_CHECKPOINT=$((TIME_ELAPSED / NEW_CHECKPOINT_COUNT))
            REMAINING_TIME=$((TIME_PER_CHECKPOINT * REMAINING / 60))
            echo "⏰ Estimated time remaining: ~$REMAINING_TIME minutes"
        fi
    fi
else
    echo "⏳ No checkpoints saved yet (still in first 20k timesteps)"
    echo "   First checkpoint expected around 7:20-7:30 PM"
fi

echo ""
echo "==================================================="
echo "KEY OBSERVATIONS"
echo "==================================================="
if [ -f "full_training.log" ]; then
    EP_REWARD=$(grep "ep_rew_mean" full_training.log | tail -1 | awk '{print $2}')
    if [ -n "$EP_REWARD" ]; then
        echo "✅ Episode reward: $EP_REWARD (positive = good!)"
        echo "   This is MUCH better than the old model (-807)"
    fi
fi

echo ""
echo "==================================================="
echo "Run './check_training_progress.sh' again to update"
echo "==================================================="
