# MARLIN: Multi-Scale Temporal Encoding for Epidemiological Prediction

## Overview

MARLIN is a deep learning model designed for **epidemiological forecasting and parameter estimation** using multi-scale temporal encoding with attention mechanisms. The model leverages multi-branch convolutional architectures to capture patterns at different temporal scales and uses a multi-task learning framework to simultaneously predict:

- **EIR** (Entomological Inoculation Rate)
- **Prevalence (φ)** 
- **Incidence** of disease

## Key Features

### 🔧 Multi-Scale Convolution (MSConv)
Processes temporal sequences using three parallel branches with different dilation rates:
- **Ultra-low dilation**: Captures fine-grained micro-structures
- **Low dilation**: Identifies short-term trends  
- **High dilation**: Extracts long-range dependencies

### 🧠 Hybrid Architecture
- **Temporal Encoders** with LSTM layers for sequence processing
- **Self-Attention Mechanisms** for adaptive temporal weighting
- **Hybrid Positional Encoding** combining Fourier and learned embeddings

### 📊 Multi-Task Learning
Joint training on three epidemiological outputs:
- EIR prediction for transmission intensity
- Prevalence estimation for disease burden
- Incidence forecasting for case prediction

## Quick Links

- **[Architecture](architecture.html)** - Detailed model components and design
- **[Methodology](methodology.html)** - Training procedures and workflow
- **[Experiments](experiments.html)** - Experiment configurations and runs
- **[Results](results.html)** - Performance metrics and analysis

## Model Specifications

| Component | Value |
|-----------|-------|
| Hidden Dimension | 128 |
| LSTM Layers | 3 |
| Conv Channels per Branch | 8 |
| Total Conv Output Channels | 24 |
| Positional Encoding | Fourier + Learned |
| Attention Mechanism | Self-Attention |

## Dataset

- **Training runs**: 15,000
- **Evaluation runs**: 300  
- **Test runs**: 200
- **Data source**: Simulation-based epidemiological data

## Getting Started

See the [Methodology](methodology.html) page for training procedures and [Architecture](architecture.html) for implementation details.

---

*Last updated: February 2026*
