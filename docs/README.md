# MARLIN Documentation - Getting Started

Welcome! This documentation site showcases the development methodology and architecture of the **MARLIN** (Multi-scale temporal encoding with Attention for epidemiological prediction) model.

## 📚 Documentation Pages

### Main Pages
1. **[Home](index.html)** - Project overview, key features, and specifications
2. **[Quick Reference](quick-reference.html)** - Fast lookup for model specs, hyperparameters, and commands
3. **[Architecture](architecture.html)** - Detailed breakdown of model components:
   - Multi-Scale Convolution (MSConv)
   - Temporal Encoders with LSTM
   - Self-Attention Mechanisms
   - Multi-Task Output Heads
4. **[Methodology](methodology.md)** - Complete training pipeline:
   - Data preparation and transformation
   - Sequence creation
   - Training procedures
   - Evaluation and testing
   - Hyperparameter tuning strategies
5. **[Experiments](experiments.html)** - Experiment configurations:
   - Baseline experiments
   - Ablation studies
   - Hyperparameter searches
   - Tracking templates
6. **[Results](results.html)** - Performance tracking and analysis:
   - Results tables (status: pending)
   - Benchmark comparisons
   - Cross-validation results
   - Key findings

### Support Pages
- **[GUIDE.md](GUIDE.md)** - Documentation maintenance and markdown tips
- **[quick-reference.html](quick-reference.html)** - Model specifications at a glance

---

## 🚀 Quick Start

### 1. Read the Overview
Start with the [Home Page](index.html) for high-level project understanding.

### 2. Understand the Model
Read [Architecture](architecture.html) to understand what makes MARLIN unique:
- Multi-scale convolution branches (ultra-low, low, high dilation)
- Temporal encoding with LSTM and attention
- Multi-task learning for EIR, immunity function (phi), and incidence

### 3. Learn the Process
Check [Methodology](methodology.md) for how to:
- Prepare your data
- Create training sequences
- Configure and train the model
- Evaluate performance

### 4. Run Experiments
Use [Experiments](experiments.html) page for:
- Baseline model configuration
- Ablation study setups
- Hyperparameter tuning ranges
- Experiment tracking spreadsheets

### 5. Track Results
Document your findings in [Results](results.html):
- Performance metrics for each experiment
- Model checkpoints
- Observations and key findings

---

## 📊 Model Summary

| Aspect | Details |
|--------|---------|
| **Purpose** | Epidemiological forecasting and parameter estimation |
| **Key Innovation** | Multi-scale temporal convolution + attention-based encoding |
| **Tasks** | EIR prediction, Prevalence estimation, Incidence forecasting |
| **Architecture** | 3 separate temporal encoders + 3 output heads + incidence fusion layer |
| **Total Parameters** | ~450K trainable parameters |
| **GPU Memory** | ~800MB (training), ~200MB (inference) |

---

## 📁 Documentation Structure

```
docs/
├── index.md                  ← START HERE
├── quick-reference.md        ← Fast lookup
├── architecture.md           ← Model design
├── methodology.md            ← How to train
├── experiments.md            ← Experiment configs
├── results.md                ← Track performance
├── GUIDE.md                  ← Docs maintenance
├── _config.yml               ← Jekyll settings
├── Gemfile                   ← Dependencies
└── .gitignore               ← Git settings
```

---

## 🔧 Publishing Your Docs

### Enable GitHub Pages
1. Go to **Settings → Pages**
2. Set **Source**: Deploy from branch → `main` → `/docs`
3. Your site will be live at: `https://olatundemi.github.io/MARLIN/`

### Test Locally
```bash
cd docs/
bundle install
bundle exec jekyll serve
# Open http://localhost:4000
```

See **GITHUB_PAGES_SETUP.md** in the project root for detailed instructions.

---

## 📝 Using This Documentation

### For Running Experiments
1. Choose configuration from [Experiments](experiments.html)
2. Run your training
3. Record results in [Results](results.html)
4. Document observations

### For Understanding the Model
1. Start with [Quick Reference](quick-reference.html) for specifications
2. Read [Architecture](architecture.html) for detailed component descriptions
3. Check [Methodology](methodology.md) for implementation details

### For Sharing Knowledge
- Update documentation after significant changes
- Use the templates provided in [Results](results.html)
- Follow markdown style from existing pages

---

## 💡 Key Concepts

### Multi-Scale Convolution
Three parallel conv branches with different dilation rates capture patterns at multiple temporal scales simultaneously.

### Temporal Encoder
LSTM-based sequence processor with:
- Convolutional feature extraction
- Optional positional encoding
- Self-attention pooling
- Efficient sequence modeling

### Multi-Task Learning
Single model predicts three related epidemiological quantities:
- **EIR**: Transmission intensity (64-timestep windows)
- **Prevalence**: Disease burden (245-timestep windows)
- **Incidence**: Case counts (synchronized with prevalence)

### Masked Training
Proper handling of padded sequences ensures only valid data contributes to loss and attention.

---

## 📈 Development Workflow

### Experiment Cycle
```
1. Design experiment (Experiments page)
2. Configure hyperparameters
3. Run training
4. Record metrics (Results page)
5. Analyze findings
6. Iterate or move to next experiment
```

### Documentation Updates
```
After each experiment:
- Update results table
- Record model checkpoint
- Note key observations
- Commit to git
```

---

## 🤝 Contributing

To contribute to documentation:
1. Edit relevant markdown file
2. Test locally with Jekyll
3. Commit changes with descriptive message
4. Push to main branch

See **GUIDE.md** for markdown conventions and best practices.

---

## 📚 External References

- [Jekyll Documentation](https://jekyllrb.com/)
- [GitHub Pages Guide](https://docs.github.com/en/pages)
- [Markdown Syntax](https://www.markdownguide.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## 📝 Last Updated

February 2026

---

**Start exploring**: [Go to Home Page →](index.html)

