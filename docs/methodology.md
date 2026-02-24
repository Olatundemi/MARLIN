# Methodology: Model Development Pipeline

## Development Workflow

### Phase 1: Data Preparation

#### 1.1 Data Loading and Splitting
```python
Load CSV with simulation runs
  ↓
Stratified split by run IDs:
  - Training: 15,000 runs
  - Evaluation: 300 runs  
  - Test: 200 runs
  ↓
Apply log transformation to positive quantities
```

**Transformation Details**:
- Applied columns: `EIR_true`, `phi`, `prev_true`, `incall`
- Formula: `x_log = log(x + 1e-8)` (ensures numerical stability)

#### 1.2 Sequence Creation

Three separate multi-stream sequences created per run:

**EIR Stream**:
- Extract prevalence history (`prev_true`) as input features
- Build sliding windows with past=11, future=4
- Align with EIR labels for prediction

**Immunity Function Stream**:
- Extract prevalence as input and maintain history
- Build windows with past=245, future=0
- Align with phi (immunity function) labels

**Incidence Stream**:
- Uses same temporal alignment as immunity function
- Targets: incidence values (`incall`)
- Powered by both EIR and immunity function encoders


### Phase 2: Dataset and DataLoader Setup

```python
MultiTaskDataset
  ↓
Returns batch dictionary:
{
  'eir': (X_eir, M_eir, y_eir),    # Input, Mask, Target
  'phi': (X_phi, M_phi, y_phi),
  'inc': (M_inc, y_inc)            # Mask shared with EIR
}
  ↓
DataLoader with batch_size and shuffling
```

**Masking**:
- Masks indicate valid (padded=1) vs invalid (padded=0) timesteps
- Used during attention computation and loss masking


### Phase 3: Model Training

#### 3.1 Initialization
```python
model = MultiHeadModel(max_len=256)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=epochs//4, gamma=0.5)
```

**Key Hyperparameters**:
- Learning rate: To be determined during tuning
- Batch size: Configurable (typical: 32-64)
- Epochs: Configurable (typical: 100-300)
- Gradient clipping: 1.0 (for stability)

#### 3.2 Training Loop
```
for epoch in range(num_epochs):
  for batch in train_loader:
    # Forward pass
    out_eir, out_phi, out_inc = model(batch)
    
    # Mask-aware loss computation
    loss_eir = MSELoss(out_eir, batch['eir'][2], mask=batch['eir'][1])
    loss_phi = MSELoss(out_phi, batch['phi'][2], mask=batch['phi'][1])
    loss_inc = MSELoss(out_inc, batch['inc'][1], mask=batch['inc'][0])
    
    # Multi-task loss
    loss = w_eir * loss_eir + w_phi * loss_phi + w_inc * loss_inc
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
  
  # Validation
  val_loss, metrics = evaluate(model, eval_loader)
  scheduler.step()
```

#### 3.3 Loss Computation

**Masked MSE Loss**:
```python
def masked_mse_loss(pred, target, mask):
    # pred, target: (B, 1)
    # mask: (B, T)
    
    diff = (pred - target) ** 2
    masked = diff * mask.mean(dim=1, keepdim=True)
    return masked.mean()
```

Ensures only valid timesteps contribute to loss.


### Phase 4: Evaluation

#### 4.1 Metrics
```python
for batch in eval_loader:
    predictions = model(batch)
    
    # Metrics computation:
    # - Mean Absolute Error (MAE)
    # - Root Mean Squared Error (RMSE)
    # - R² Score
    # - Mean Squared Error (MSE)
```

#### 4.2 Early Stopping
```python
best_val_loss = float('inf')
patience = 20
patience_counter = 0

for epoch in range(num_epochs):
    val_loss = evaluate(model, eval_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint(model)
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        break
```


### Phase 5: Testing

```python
# Load best checkpoint
model.load_state_dict(checkpoint)
model.eval()

# Inference on test set
with torch.no_grad():
    predictions = []
    actuals = []
    
    for batch in test_loader:
        out_eir, out_phi, out_inc = model(batch)
        predictions.append({
            'eir': out_eir.cpu().numpy(),
            'phi': out_phi.cpu().numpy(),
            'inc': out_inc.cpu().numpy()
        })
        actuals.append({
            'eir': batch['eir'][2].cpu().numpy(),
            'phi': batch['phi'][2].cpu().numpy(),
            'inc': batch['inc'][1].cpu().numpy()
        })

# Compute test metrics
test_metrics = compute_metrics(predictions, actuals)
```


## Hyperparameter Tuning Strategy

### Primary Tuning Objectives
1. **Loss weighting**: Balance between EIR, immunity function, and incidence tasks
2. **Learning rate**: Optimal gradient descent step size
3. **LSTM layers**: Depth of temporal context modeling
4. **Positional encoding**: Fourier vs learned vs hybrid

### Suggested Ranges
| Hyperparameter | Range | Default |
|---|---|---|
| Learning rate | [1e-4, 1e-2] | 1e-3 |
| Batch size | [16, 128] | 32 |
| Conv channels | [4, 16] | 8 |
| Hidden dim | [64, 256] | 128 |
| LSTM layers | [1, 4] | 3 |
| w_eir | [0.1, 1.0] | 0.5 |
| w_phi | [0.1, 1.0] | 0.5 |
| w_inc | [0.1, 1.0] | 0.3 |


## Reproducibility

```python
import torch
import numpy as np
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# GPU determinism (if using CUDA)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
```

All experiments in this repository use **seed=42** for data splitting and initialization.


## Key Design Decisions

### 1. Multi-Scale Convolution
- Different dilation rates capture varying temporal scales
- Parallel architecture allows efficient multi-scale processing
- No sequential application → faster training

### 2. Separate Temporal Encoders
- EIR and immunity function have different temporal dynamics
- Separate encoders allow task-specific temporal modeling
- Incidence head fuses both encodings for joint prediction

### 3. Hybrid Positional Encoding
- Fourier encoding captures periodic patterns
- Learned embeddings adapt to data-specific positions
- Hybrid approach provides flexibility and expressiveness

### 4. Mask-Aware Training
- Padding used for short sequences (EIR window: 15)
- Masks ensure only valid data contributes to loss
- Attention mechanism respects masking


---

Next: [Experiments](experiments.html)
