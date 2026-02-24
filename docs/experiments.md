# Experiments

## Experiment Configuration

### Baseline Experiments

#### Exp 1: Base Multi-Head Model (No Positional Encoding)
```python
Config:
  - Model: MultiHeadModel
  - Hidden dim: 128
  - LSTM layers: 3
  - Positional encoding: None
  - Conv channels: 8
  - Learning rate: 1e-3
  - Batch size: 32
  - Epochs: 100
  - Loss weights: w_eir=0.5, w_phi=0.5, w_inc=0.3
```

**Rationale**: Establishes baseline performance without complex positional encoding.

---

#### Exp 2: With Fourier Positional Encoding (EIR only)
```python
Config:
  - Model: MultiHeadModel
  - Hidden dim: 128
  - LSTM layers: 3
  - Positional encoding: FourierPositionalEncoding
    - num_freqs: 8
    - max_len: 256
  - Applied to: EIR temporal encoder only
  - Learning rate: 1e-3
  - Epochs: 100
```

**Rationale**: Tests whether Fourier encodings improve short-term EIR prediction.

---

#### Exp 3: Hybrid Positional Encoding
```python
Config:
  - Model: MultiHeadModel with HybridPositionalEncoding
  - Positional encoding: Fourier + Learned
    - num_freqs: 8
    - alpha_init: 1.0
    - beta_init: 0.5
  - Applied to: EIR and prevalence encoders
  - Learning rate: 1e-3
  - Epochs: 100
```

**Rationale**: Combines periodic awareness (Fourier) with learned adaptations (Embedding).

---

### Ablation Studies

#### Ablation A1: Effect of Attention Mechanism
```python
Experiment A1a (with attention):
  - TemporalEncoder.use_attention = True
  
Experiment A1b (without attention):
  - TemporalEncoder.use_attention = False
  - Uses last valid timestep pooling instead
```

**Metric**: Compare validation loss and test RMSE between variants.

---

#### Ablation A2: LSTM Layer Depth
```python
Experiment A2a: num_layers=1
Experiment A2b: num_layers=2
Experiment A2c: num_layers=3 (baseline)
Experiment A2d: num_layers=4
```

**Metric**: Training speed vs. validation performance trade-off.

---

#### Ablation A3: Multi-Scale Convolution Branch Effectiveness
```python
Experiment A3a: All three branches (baseline)
Experiment A3b: Only ultra-low + low branches
Experiment A3c: Only low + high branches
Experiment A3d: Only high dilation branch
```

**Metric**: Impact on capturing multi-scale temporal patterns.

---

### Loss Weighting Experiments

#### Exp 4: Task Balancing
```python
Grid search over:
  - w_eir ∈ [0.1, 0.3, 0.5, 0.7, 0.9]
  - w_phi ∈ [0.1, 0.3, 0.5, 0.7, 0.9]
  - w_inc ∈ [0.1, 0.3, 0.5, 0.7]
  
All combinations maintaining sum ≤ 2.0
```

**Metric**: Task-specific test RMSE and overall validation loss.

---

### Hyperparameter Sweep

#### Exp 5: Learning Rate Sensitivity
```python
Grid search:
  - lr ∈ [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
  
Fixed other params:
  - batch_size: 32
  - epochs: 100
  - scheduler: StepLR with gamma=0.5, step=25
```

---

#### Exp 6: Batch Size Impact
```python
Grid search:
  - batch_size ∈ [16, 32, 64, 128, 256]
  
Fixed other params:
  - lr: 1e-3
  - epochs: 100
```

---

## Expected Results

### Performance Benchmarks

| Task | Metric | Baseline | With Attention | With PE |
|------|--------|----------|----------------|---------|
| EIR | RMSE | X₁ | X₁ × 0.95 | X₁ × 0.92 |
| Prevalence | RMSE | Y₁ | Y₁ × 0.97 | Y₁ × 0.94 |
| Incidence | RMSE | Z₁ | Z₁ × 0.96 | Z₁ × 0.93 |

*X₁, Y₁, Z₁ = baseline RMSE values (to be populated after runs)*

---

## Experiment Tracking Template

For each experiment, record:

```markdown
## [Exp_Name]
**Date**: YYYY-MM-DD
**Config**: [See section above]
**Training Time**: X hours
**Best Epoch**: N (of M)

### Results
- Train Loss: X.XXX
- Val Loss: X.XXX  
- Test RMSE (EIR): X.XXX
- Test RMSE (Phi): X.XXX
- Test RMSE (Inc): X.XXX
- Test R² (EIR): X.XXX
- Test R² (Phi): X.XXX
- Test R² (Inc): X.XXX

### Observations
- Key findings
- Performance insights
- Potential improvements

### Model Checkpoint
- Path: `src/trained_model/[name].pth`
- Size: X MB
```

---

## Running Experiments

### Standard Experiment Run
```python
# In Python script or notebook
python train.py --config configs/exp1_baseline.yaml

# Or with command-line overrides
python train.py \
  --config configs/baseline.yaml \
  --lr 1e-3 \
  --batch_size 32 \
  --epochs 100 \
  --save_path "trained_model/exp1_baseline.pth"
```

### Batch Experiment Runs
```bash
# Run all experiments in sequence
for exp in baseline fourier hybrid ablation_*; do
  echo "Running $exp..."
  python train.py --config "configs/${exp}.yaml"
done
```

---

## Visualization and Analysis

### Training Curves
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Curves')
plt.legend()

# Prediction vs Actual
plt.subplot(1, 3, 2)
plt.scatter(predictions, actuals, alpha=0.5)
plt.plot([min, max], [min, max], 'r--')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('EIR: Prediction vs Actual')

# Residuals
plt.subplot(1, 3, 3)
residuals = predictions - actuals
plt.hist(residuals, bins=50)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Residual Distribution')

plt.tight_layout()
plt.savefig('results/exp1_analysis.png')
```

---

Next: [Results](results.html)
