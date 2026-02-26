---
layout: default
title: Architecture
---

# Architecture: MARLIN Model Design

## System Overview

```
Input Data → MSConv Branches → Projection → Positional Encoding → LSTM → Attention → Output Heads
```
<img src="https://raw.githubusercontent.com/Olatundemi/MARLIN/main/assets/Multihead_Model.png" alt="Multihead Model" style="max-width: 800px; width: 100%; height: auto; margin: 0rem 0;">

## Core Components

### 1. Multi-Scale Convolution (MSConv)

The MSConv layer processes temporal sequences using three parallel convolutional branches:

#### Ultra-Low Dilation Branch
```python
Conv1d(1, 8, kernel=3, dilation=1)
  ↓ ReLU + BatchNorm
Conv1d(8, 8, kernel=3, dilation=1)
  ↓ ReLU + BatchNorm
```
- **Purpose**: Capture micro-structures and fine-grained patterns
- **Receptive field**: 5 timesteps

#### Low Dilation Branch
```python
Conv1d(1, 8, kernel=5, dilation=2)
  ↓ ReLU + BatchNorm
Conv1d(8, 8, kernel=5, dilation=2)
  ↓ ReLU + BatchNorm
```
- **Purpose**: Identify short-term trends
- **Receptive field**: ~13 timesteps

#### High Dilation Branch
```python
Conv1d(1, 8, kernel=7, dilation=8)
  ↓ ReLU + BatchNorm
Conv1d(8, 8, kernel=7, dilation=16)
  ↓ ReLU + BatchNorm
```
- **Purpose**: Extract long-range temporal dependencies
- **Receptive field**: >100 timesteps

**Output**: Concatenated 24-channel feature map from all three branches


### 2. Temporal Encoder

```python
MSConv Output (B, T, 24)
    ↓
Linear Projection (B, T, 128)
    ↓
Positional Encoding (optional)
    ↓
Masking
    ↓
LSTM (3 layers, 128 hidden)
    ↓
Self-Attention or Last Valid Timestep
    ↓
Context Vector (128,)
```

**Parameters**:
- Hidden dimension: 128
- LSTM layers: 3
- Bidirectional: False
- Use attention: True


### 3. Self-Attention Mechanism

```python
Input: x (B, T, H)
  ↓
Energy = tanh(W * x)  // (B, T, H)
  ↓
Scores = v^T * Energy  // (B, T, 1)
  ↓
Mask padding positions → -inf
  ↓
Weights = softmax(Scores)
  ↓
Context = Weights @ x  // (B, H)
```

- Computes attention weights over valid timesteps
- Produces weighted context vector
- Used for pooling sequence representations


### 4. Positional Encoding (Optional)

#### Fourier Positional Encoding
```python
t ∈ [0, 1]  // normalized time
freqs = [2^0, 2^1, ..., 2^7]
angles = 2π * t * freqs
PE = [sin(angles), cos(angles)]  // 16-dim
PE_proj = Linear(16 → H)
```

#### Learned Positional Embedding
```python
positions = [0, 1, 2, ..., T-1]
PE = Embedding(T, H)
```

#### Hybrid Approach
```python
PE_total = x + α * PE_fourier + β * PE_learned
```
- α, β: learnable parameters


### 5. Multi-Task Output Heads

```
EIR Head:         h_eir (128,) → Linear(128 → 1) → EIR output
Immunity Function Head:  h_phi (128,) → Linear(128 → 1) → φ output
Incidence Head:   [h_eir_seq (B, T, 128),
                   h_phi_seq (B, T, 128)]
                    ↓
                   Attention pooling on both sequences
                    ↓
                   Concat [z_eir, z_phi] (256,)
                    ↓
                   MLP: 256 → 256 → 128 → 1
```


## Multi-Head Model Architecture

```
Input: {
  'eir': (X_eir, M_eir, y_eir),
  'phi': (X_phi, M_phi, y_phi),
  'inc': (M_eir, y_inc)
}
  ↓
┌─────────────────────┬──────────────────────┐
│ EIR Encoder         │ Immunity Function    │
│ (TemporalEncoder)   │ Encoder (Temporal)   │
│ 3 LSTM layers       │ 3 LSTM layers        │
│ Attention pooling   │ Attention pooling    │
└──────┬──────────────┴──────────┬───────────┘
       │                         │
       ↓                         ↓
   h_eir (B,128)           h_phi (B,128)
       │                         │
       ├─→ Linear(128→1)         │
       │        ↓                │
       │     EIR output          │
       │                         │
       │                    ├─→ Linear(128→1)
       │                    │        ↓
       │                    │    Immunity output
       │                         │
       └─────────→ Incidence Head ←──┴─────
                    (MLP decoder)
                          ↓
                   Incidence output
```


## Sequence Creation Pipeline

### EIR Prediction
- **Window size**: 15 timesteps
- **Past context**: 11 timesteps (75% of window)
- **Future context**: 4 timesteps
- **Target**: EIR value at center timestep

### Immunuty function Prediction
- **Window size**: 245 timesteps
- **Past context**: 245 timesteps (full history)
- **Future context**: 0 timesteps
- **Target**: Current prevalence value

### Incidence Prediction
- **Multi-stream input**: Uses both EIR and immunity function encoders
- **Window alignment**: Synchronized with immunity function window
- **Target**: Incidence value at each timestep


## Data Transformations

Applied before training:
```python
# Log transformation for positive quantities
transform(x) = log(x + 1e-8)

# Applied to:
# - EIR_true
# - phi (immunity function)
# - prev_true 
# - incall (incidence)
```

This provides numerical stability and better gradient flow for epidemiological quantities.


## Loss Functions

Multi-task learning with weighted sum:

```python
L_total = w_eir * L_eir + w_phi * L_phi + w_inc * L_inc
```

Typical loss:
- **L_eir**: MSE(predicted EIR, target EIR)
- **L_phi**: MSE(predicted φ, target φ)
- **L_inc**: MSE(predicted incidence, target incidence)

Weight balancing strategies determined during training phase.


---

Next: [Training Methodology](methodology.html)
