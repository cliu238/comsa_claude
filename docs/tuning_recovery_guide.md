# Hyperparameter Tuning Recovery Guide

## Overview

The hyperparameter tuning phase is computationally intensive and may take 2-4 hours to complete. This guide explains how to handle interruptions and resume the process.

## Running the Tuning Phase

### Option 1: Full Tuning Run (Original Script)

```bash
./run_tuning_phase.sh
```

- Runs 100 Bayesian optimization trials per model
- 4 models × 100 trials × 5 CV folds = 2000 model trainings per experiment
- 20 total experiments (different site combinations)
- Expected runtime: 2-4 hours

### Option 2: Monitored Run with Resume Capability (Recommended)

```bash
# First run
python run_tuning_with_monitoring.py

# Resume after interruption
python run_tuning_with_monitoring.py --resume --output-dir results/tuning_phase_YYYYMMDD_HHMMSS
```

Features:
- Progress monitoring with time estimates
- Automatic checkpointing every 2 experiments
- Graceful interruption handling (Ctrl+C)
- Resume capability from last checkpoint

### Option 3: Quick Test Run

```bash
./test_tuning_quick.sh
```

- Only 5 tuning trials (instead of 100)
- Only 2 models (xgboost, logistic_regression)
- 3-fold CV (instead of 5)
- Expected runtime: 5-10 minutes
- Use this to verify the pipeline works before full run

## Monitoring Progress

### Ray Dashboard

Visit http://localhost:8265 while the script is running to see:
- Active tasks and workers
- Resource utilization
- Task timeline

### Log Files

Monitor progress in real-time:
```bash
# Watch progress tracker log
tail -f logs/orchestration/progress_tracker_*.log

# Check for errors
tail -f logs/orchestration/__main__*.log
```

### Progress Indicators

The monitored script shows:
- Current experiment number (e.g., 5/20 experiments completed)
- Estimated time remaining
- Checkpoint save notifications

## Handling Interruptions

### Graceful Shutdown

1. Press Ctrl+C once
2. Wait for "Saving progress..." message
3. Note the resume command shown

### Force Shutdown (if needed)

```bash
# Kill Ray processes
pkill -f "ray::"
pkill -f "run_distributed_comparison.py"

# Clean up Ray session
ray stop
```

### Resume from Checkpoint

```bash
# Use the command shown after interruption
python run_tuning_with_monitoring.py --resume --output-dir results/tuning_phase_20250730_134305
```

## Checkpoint Structure

Checkpoints are saved in the output directory:
```
results/tuning_phase_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── progress.json          # Overall progress tracking
│   ├── checkpoint_batch_1.pkl # Completed experiment results
│   └── ...
├── processed_data/            # Preprocessed datasets
└── tuning_results/           # Best hyperparameters (when complete)
```

## Troubleshooting

### Out of Memory

If you see memory warnings:
1. Reduce `--tuning-max-concurrent-trials` (default: 4)
2. Reduce `--n-workers` (default: 8)
3. Increase `--memory-per-worker` (default: 2GB)

### Ray Dashboard Not Accessible

If http://localhost:8265 doesn't work:
1. Check if port is in use: `lsof -i :8265`
2. Use different port: `--ray-dashboard-port 8266`

### Slow Progress

Normal tuning speed:
- ~1-2 experiments per 10 minutes
- Each experiment involves multiple model × trial combinations

To speed up:
1. Reduce `--tuning-trials` (but may affect quality)
2. Increase `--n-workers` (if CPU available)
3. Use `--tuning-algorithm random` instead of `bayesian`

## Next Steps

After tuning completes:
1. Check results in `results/tuning_phase_*/tuning_results/`
2. Run evaluation phase: `./run_evaluation_phase.sh`
3. Review hyperparameter analysis in results directory