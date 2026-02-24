# Quick Reference

## Model Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                    MARLIN MultiHeadModel                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
        ┌───────▼─────┐  ┌────▼──────┐  ┌─────▼────────┐
        │ EIR Encoder │  │ Phi       │  │ Incidence    │
        │             │  │ Encoder   │  │ Head         │
        │ (3L LSTM)   │  │ (3L LSTM) │  │              │
        └───────┬─────┘  └────┬──────┘  └─────▼────────┘
                │             │              │
         ┌──────▼─┐     ┌─────▼──┐    ┌─────▼───────┐
         │ h_eir  │     │ h_phi  │    │ Incidence   │
         │ (128D) │     │ (128D) │    │ Prediction  │
         └──┬─────┘     └────┬───┘    └─────────────┘
            │                │
         ┌──▼─┐          ┌───▼──┐
         │EIR │          │ Phi  │
         │out │          │ out  │
         └────┘          └──────┘
```

---

## Key Specifications

### Model Parameters
- **Total trainable parameters**: ~450K
- **GPU memory (inference)**: ~200MB
- **GPU memory (training)**: ~800MB

### Multi-Scale Convolution
| Level | Dilation | Kernel | Receptive Field |
|-------|----------|--------|-----------------|
| Ultra-low | 1 | 3 | 5 |
| Low | 2 | 5 | ~13 |
| High | 8-16 | 7 | >100 |

### LSTM Configuration
- **Layers**: 3
- **Hidden size**: 128
- **Bidirectional**: No
- **Dropout**: 0.0 (can be added)

### Training Configuration
```python
Optimizer:     Adam
Learning rate: 1e-3 (configurable)
Batch size:    32 (configurable)
Epochs:        100-300
Scheduler:     StepLR (step=25, gamma=0.5)
Gradient clip: 1.0
Early stop:    patience=20
```

---

## Data Specifications

### Input Sequences
| Task | Window | Past | Future | Shape |
|------|--------|------|--------|-------|
| EIR | 15 | 11 | 4 | (11+4+1, 1) |
| Immunity Function | 245 | 245 | 0 | (245, 1) |
| Incidence | 245 | 245 | 0 | (245, 1) |

### Dataset Split
```
Training:   15,000 runs
Validation:    300 runs
Test:          200 runs
Total:      15,500 runs
```

### Data Transformations
```python
x_transformed = log(x + 1e-8)
Applied to: EIR_true, phi, prev_true, incall
```

---

## Loss Function

### Multi-Task Loss
```python
L_total = w_eir * L_eir + w_phi * L_phi + w_inc * L_inc

Where:
  L_eir, L_phi, L_inc = MSELoss (masked)
  w_eir, w_phi, w_inc = task weights
```

### Default Weights
```python
w_eir = 0.5   # EIR prediction weight
w_phi = 0.5   # Immunity function prediction weight  
w_inc = 0.3   # Incidence prediction weight
```

---

## Performance Baseline

*To be updated after initial experiments*

```
Task                  MAE      RMSE     R²
──────────────────────────────────────
EIR:                  —        —        —
Immunity Function:    —        —        —
Incidence:    —        —        —
```

---

## File Structure

```
MARLIN/
├── model_dev/
│   └── Temporal_Encoder_MSconv_multihead_for_eir_phi_inc.py  [MAIN MODEL]
├── src/
│   ├── model_exp.py
│   ├── inference.py
│   ├── evaluate.py
│   └── preprocessing.py
├── marlin_main/
│   ├── app.py
│   └── cli.py
└── docs/                      [DOCUMENTATION]
    ├── index.md
    ├── architecture.md
    ├── methodology.md
    ├── experiments.md
    ├── results.md
    └── GUIDE.md
```

---

## Quick Commands

### Training
```python
# Basic training
python train.py --epochs 100 --batch_size 32

# With custom config
python train.py --config configs/exp1.yaml

# Resume from checkpoint
python train.py --resume checkpoints/model_epoch50.pth
```

### Inference
```python
# Single prediction
model.eval()
with torch.no_grad():
    predictions = model(batch)

# Batch inference
predictions = model.predict(test_loader)
```

### Evaluation
```python
# Compute metrics
metrics = evaluate(model, test_loader)
print(f"RMSE (EIR): {metrics['eir_rmse']:.4f}")
print(f"R² (Phi): {metrics['phi_r2']:.4f}")
```

---

## Hyperparameter Ranges

### Learning Rate
```
Suggested: [1e-4, 1e-3, 5e-3]
Safe range: [5e-5, 1e-2]
```

### Batch Size
```
Suggested: 32, 64
Options: 16, 32, 64, 128
```

### Hidden Dimension
```
Baseline: 128
Options: 64, 128, 256
```

### LSTM Layers
```
Baseline: 3
Options: 1, 2, 3, 4
```

---

## Debugging Checklist

- [ ] Load data correctly (check shapes)
- [ ] Model forward pass works (test batch)
- [ ] Loss computation stable (no NaN)
- [ ] Gradients flowing (check grad norms)
- [ ] Learning rate appropriate (watch loss)
- [ ] Validation loss decreasing
- [ ] No GPU/memory issues

---

## Contact & References

**Model Development**: See [Methodology](methodology.html)  
**Architecture Details**: See [Architecture](architecture.html)  
**Experiment Configs**: See [Experiments](experiments.html)  
**Results**: See [Results](results.html)  

---

*Quick Reference Guide - February 2026*
