---
layout: default
title: Home
---

## MARLIN 
Malaria ANC-based Reconstructions with Learning-based Inference using Neural networks

## Overview

MARLIN is a deep learning based surrogate designed to emulate certain unlerlying mechanistic behaviour of determistic version of [Malariasimulation](https://github.com/mrc-ide/anatembea) for faster **malaria epidemiological reconstruction/parameter estimation** using multi-scale temporal encoding with attention mechanisms. The model leverages multi-scale convolutional architectures to capture patterns at different temporal scales and uses a multi-task learning framework to simultaneously predict:
- **EIR** (Entomological Inoculation Rate)
- **Immunity function** (φ or phi)
- **Incidence** of disease

from **observed malaria prevalence** timeseries.

## Why this research matters
<img src="https://raw.githubusercontent.com/Olatundemi/MARLIN/main/assets/illustrative_example.png" alt="Illustrative example" style="max-width: 800px; width: 100%; height: auto; margin: 0rem 0;">

Malaria transmission is not directly observable. What we measure such as infection prevalence or clinical cases is a delayed and distorted reflection of the true underlying dynamics, including mosquito density, transmission intensity (EIR), immunity function etc. As illustrated above, these signals are temporally misaligned: peaks in rainfall drive mosquito populations, which in turn drive transmission, but observed prevalence and incidence respond later and more smoothly. Moreso, Identical prevalence values (purple dots) can therefore correspond to very different underlying incidence (red dots).
This mismatch makes epidemiological inference fundamentally challenging. By the time changes appear in routine data, the underlying transmission dynamics may have already shifted.
This research aims to bridge that gap by learning the hidden relationships between observable data and unobserved transmission processes, enabling faster and more accurate inference of malaria dynamics from routine surveillance.


## Key Features

### Mechanistic Backbone
Malariasimulation forms the mechanistic backbone of this surrogate due to its biological realism, rich representation of immunity, vector-host interactions, and heterogeneous intervention effects.

### Multi-stream Temporal Sequences construction 
- **Decoupled temporal receptive fields**: Different biological processes are learned using tailored input horizons.

- **Implicit temporal hierarchy**: The model learns both short-term dynamics (transmission) and long-term memory (immunity).

- **Structured supervision**: Targets are aligned per timestep, enabling stable multi-output learning.


### Multi-Scale Convolution (MSConv)
Processes temporal sequences using three parallel branches with different dilation rates:
- **Ultra-low dilation**: Captures fine-grained micro-structures
- **Low dilation**: Identifies short-term trends  
- **High dilation**: Extracts long-range dependencies

### Hybrid Architecture
- **Temporal Encoders** with LSTM layers for sequence processing
- **Self-Attention Mechanisms** for adaptive temporal weighting
- **Hybrid Positional Encoding** combining Fourier and learned embeddings (This is currently not in use but defined for potential usage)

### Multi-Task Learning
Joint training on three epidemiological outputs:
- EIR prediction for transmission intensity
- immunity estimation
- Incidence forecasting for case prediction

### Out of Sample/Distribution Detection
We use Mahalanobis latent distance to estimate out of distribution threshold

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
