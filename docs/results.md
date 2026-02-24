# Results

This page documents experimental results, performance metrics, and key findings from the MARLIN model development pipeline.

## Results Summary

### Performance Overview

Results will be populated as experiments are completed. Each entry includes:
- Experiment identifier
- Configuration details
- Training metrics
- Test performance
- Key observations

---

## Baseline Results

### Exp 1: Base Multi-Head Model (No Positional Encoding)
**Status**: [Pending]

**Config**:
- Hidden dim: 128
- LSTM layers: 3
- Positional encoding: None
- Learning rate: 1e-3
- Batch size: 32

**Results**:
| Metric | EIR | Prevalence | Incidence |
|--------|-----|-----------|-----------|
| Test RMSE | — | — | — |
| Test MAE | — | — | — |
| Test R² | — | — | — |
| Train Time (hours) | — | — | — |

**Observations**:
- To be filled after experiment completion

---

## Enhanced Model Results

### Exp 2: With Fourier Positional Encoding
**Status**: [Pending]

**Config**:
- Positional encoding: Fourier (8 frequencies)
- Applied to: EIR encoder

**Results**:
| Metric | EIR | Prevalence | Incidence |
|--------|-----|-----------|-----------|
| Test RMSE | — | — | — |
| Test MAE | — | — | — |
| Test R² | — | — | — |
| Improvement vs Baseline | — | — | — |

---

### Exp 3: Hybrid Positional Encoding
**Status**: [Pending]

**Config**:
- Positional encoding: Fourier + Learned (Hybrid)
- Alpha: 1.0, Beta: 0.5
- Applied to: Both EIR and prevalence encoders

**Results**:
| Metric | EIR | Prevalence | Incidence |
|--------|-----|-----------|-----------|
| Test RMSE | — | — | — |
| Test MAE | — | — | — |
| Test R² | — | — | — |
| Improvement vs Baseline | — | — | — |

---

## Ablation Study Results

### Attention Mechanism
**Status**: [Pending]

| Configuration | Pool Type | EIR RMSE | Prevalence RMSE | Incidence RMSE |
|---|---|---|---|---|
| With Attention | Attention | — | — | — |
| Without Attention | Last Timestep | — | — | — |
| Difference | — | — | — | — |

---

### LSTM Layer Depth
**Status**: [Pending]

| Num Layers | EIR RMSE | Phi RMSE | Inc RMSE | Train Time (hrs) |
|---|---|---|---|---|
| 1 | — | — | — | — |
| 2 | — | — | — | — |
| 3 | — | — | — | — |
| 4 | — | — | — | — |

---

### Multi-Scale Convolution Branches
**Status**: [Pending]

| Branch Configuration | EIR RMSE | Phi RMSE | Inc RMSE |
|---|---|---|---|
| All branches (U+L+H) | — | — | — |
| Ultra + Low only | — | — | — |
| Low + High only | — | — | — |
| High only | — | — | — |

---

## Hyperparameter Optimization

### Learning Rate Sweep
**Status**: [Pending]

| Learning Rate | Best Val Loss | EIR RMSE | Phi RMSE | Inc RMSE |
|---|---|---|---|---|
| 1e-4 | — | — | — | — |
| 5e-4 | — | — | — | — |
| 1e-3 | — | — | — | — |
| 5e-3 | — | — | — | — |
| 1e-2 | — | — | — | — |

---

### Loss Weight Optimization
**Status**: [Pending]

Best performing loss weights:

```python
Best Config:
  w_eir = X.X
  w_phi = X.X
  w_inc = X.X
  
Validation Loss: X.XXX
Test Performance:
  EIR RMSE: X.XXX
  Phi RMSE: X.XXX
  Inc RMSE: X.XXX
```

---

## Prediction Quality Analysis

### Residual Analysis
**Status**: [Pending]

```
EIR Residuals:
  Mean: X.XXX
  Std: X.XXX
  Min: X.XXX
  Max: X.XXX
  
Prevalence Residuals:
  Mean: X.XXX
  Std: X.XXX
  
Incidence Residuals:
  Mean: X.XXX
  Std: X.XXX
```

---

### Uncertainty Estimation

With Monte Carlo Dropout enabled, confidence intervals estimated:

```python
Prediction ± 95% CI (EIR):    [X.XXX, X.XXX]
Prediction ± 95% CI (Phi):    [X.XXX, X.XXX]
Prediction ± 95% CI (Inc):    [X.XXX, X.XXX]

Calibration metrics:
  Prediction interval coverage: X.X%
  Effective CI width: X.XXX
```

---

## Computational Efficiency

### Training Metrics
**Status**: [Pending]

| Model Config | Batch Size | Epochs | Time per Epoch (s) | Total Time (hrs) | GPU Memory |
|---|---|---|---|---|---|
| Baseline | 32 | 100 | — | — | — |
| With PE | 32 | 100 | — | — | — |
| Hybrid PE | 32 | 100 | — | — | — |

### Inference Metrics
**Status**: [Pending]

| Model | Batch Size | Time per Batch (ms) | Throughput (samples/sec) |
|---|---|---|---|
| MultiHeadModel | 32 | — | — |
| MultiHeadModel | 64 | — | — |
| MultiHeadModel | 128 | — | — |

---

## Cross-Validation Results

**Status**: [Pending]

5-fold cross-validation results:

| Fold | EIR RMSE | Phi RMSE | Inc RMSE | Val Loss |
|---|---|---|---|---|
| Fold 1 | — | — | — | — |
| Fold 2 | — | — | — | — |
| Fold 3 | — | — | — | — |
| Fold 4 | — | — | — | — |
| Fold 5 | — | — | — | — |
| Mean ± Std | — ± — | — ± — | — ± — | — ± — |

---

## Benchmark Comparison

### Against Baseline Methods

| Method | EIR RMSE | Phi RMSE | Inc RMSE | Notes |
|---|---|---|---|---|
| Simple LSTM | — | — | — | Single-task baseline |
| CNN-LSTM | — | — | — | Without multi-scale |
| MARLIN (Baseline) | — | — | — | No PE |
| MARLIN (Optimized) | — | — | — | With Hybrid PE |

---

## Key Findings

✏️ **Findings to be updated as experiments progress**

### Effective Design Choices
- [ ] Multi-scale convolution benefits
- [ ] Attention mechanism contribution
- [ ] Positional encoding impact
- [ ] Task weighting strategy

### Optimal Hyperparameters
- [ ] Learning rate: X.XXX
- [ ] Batch size: XXX
- [ ] LSTM layers: X
- [ ] Conv channels: XX

### Limitations Identified
- [ ] Performance on edge cases
- [ ] Generalization to new data
- [ ] Computational constraints

---

## Model Checkpoints

| Experiment | Model File | Val Loss | Test R² | Notes |
|---|---|---|---|---|
| Exp 1 | `4_layers_model.pth` | — | — | Baseline |
| Exp 2 | — | — | — | Fourier PE |
| Exp 3 | — | — | — | Best overall |

---

## How to Update Results

1. Run experiment using configuration in [Experiments](experiments.html)
2. Record metrics in corresponding table above
3. Add observations and model checkpoint path
4. Update benchmark comparison if applicable

---

## Data Visualization

### Training Progress
*Graphs will be generated and linked here*

- Train/val loss curves
- Per-task metric progression
- Learning rate schedule visualization

### Error Analysis
*Analysis plots to be added*

- Prediction vs actual scatter plots
- Residual distributions
- Error by input magnitude

---

Last Updated: February 2026

[← Back to Experiments](experiments.html) | [Home](index.html)
