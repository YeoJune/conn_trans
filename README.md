# Conn-Trans: Connection Transformer

A novel transformer architecture that performs iterative semantic reasoning through learnable connections between fixed intermediate representation (IR) nodes and dynamic activation states.

## 🎯 Overview

Conn-Trans introduces a revolutionary approach to multi-step reasoning by combining:

- **Fixed IR Nodes (H)**: Unchanging base knowledge concepts `∈ ℝ^(N_ir × d)`
- **Dynamic Activation States (X)**: Input-dependent reasoning states that evolve through iterations
- **Learnable Connection Matrix (C)**: Core reasoning mechanism `∈ ℝ^(N_ir × N_ir)`

### Key Innovation: Iterative Reasoning Process

```
X^t = (C ⊗ H) + (I + C) ⊗ X^{t-1}
```

This enables structured, interpretable reasoning while maintaining parameter efficiency.

## 🏗️ Architecture Variants

### 1. Pure Connection Transformer

- **Core**: Connection Matrix only for reasoning
- **Parameters**: ~20M (RTX 4090 optimized)
- **Hypothesis**: Novel connection mechanism sufficient for reasoning

### 2. Connection Transformer + FFN

- **Core**: Connection Matrix + Feed-Forward Networks
- **Parameters**: ~30M
- **Hypothesis**: FFN enhances connection-based reasoning

### 3. Standard Transformer (Baseline)

- **Core**: Classic multi-head attention + FFN
- **Parameters**: ~25M
- **Purpose**: Fair comparison baseline

## 🚀 Quick Start

### Prerequisites

- **Hardware**: RTX 4090 (24GB VRAM) recommended
- **Python**: 3.9+
- **CUDA**: 11.8+

### Installation

```bash
# Clone repository
git clone <repository-url>
cd conn-trans

# Create environment
conda create -n conn-trans python=3.9
conda activate conn-trans

# Install PyTorch for RTX 4090
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### Run Comprehensive Experiment

```bash
# Full comparison: Pure | +FFN | Standard Transformer
python conn_trans_prototype.py

# Expected runtime: ~4 hours on RTX 4090
# Outputs: Model checkpoints, visualizations, performance analysis
```

## 📊 Experimental Setup

### Dataset: bAbI Task 16 (Basic Induction) - 2024 Updated

```
Example:
Story: "If Jeff is a frog then Jeff is green. Jeff is a frog."
Question: "What color is Jeff?"
Answer: "green"

Dataset loading: Updated HuggingFace format (2024)
Task type: Conditional reasoning requiring rule application
Size: Variable based on loading method
```

**⚠️ Important Notes (2024):**

- HuggingFace dataset format changed: now requires `name="en"` and `task_no="qa16"`
- Original download links occasionally return 404 errors
- Multiple fallback methods implemented in code
- Automatic dummy dataset generation if loading fails

### Data Loading Methods

```python
# Primary method (2024 format)
dataset = load_dataset("facebook/babi_qa", name="en", task_no="qa16")

# Fallback methods implemented:
# 1. Alternative HuggingFace repositories
# 2. Manual download and local loading
# 3. Dummy dataset generation for architecture testing
```

### Model Configuration (RTX 4090 Optimized)

```python
CONFIG = {
    "d_model": 512,      # Model dimension
    "num_ir": 1024,      # IR nodes (2 × d_model)
    "num_steps": 4,      # Reasoning iterations
    "num_heads": 8,      # Attention heads
    "ffn_dim": 2048,     # FFN dimension
    "batch_size": 32,    # Batch size
    "max_epochs": 15,    # Training epochs
}
```

## 📈 Expected Results

### Performance Ranking (Anticipated)

```
🥇 Conn-Trans + FFN    : 88-92%
🥈 Standard Transformer: 85-90%
🥉 Pure Conn-Trans     : 82-88%
```

### Key Metrics

- **Parameter Efficiency**: Pure model achieves competitive performance with fewer parameters
- **Reasoning Capability**: Connection Matrix learns interpretable reasoning patterns
- **Scalability**: Performance scales with reasoning depth (num_steps)

## 🔬 Analysis Tools

### Connection Matrix Visualization

```python
# Automatic generation during training
visualize_connection_matrix(model, "connection_matrix.png")

# Outputs heatmap showing learned reasoning patterns
```

### Reasoning Trace Analysis

```python
# Track activation evolution through reasoning steps
trace = model.get_reasoning_trace(input_ids, attention_mask)
# Returns: [X^0, X^1, X^2, X^3, X^4] - reasoning evolution
```

### Performance Analysis

- Automated comparison reports
- Parameter efficiency metrics
- Architecture-specific insights
- Statistical significance testing

## 📁 Project Structure

```
conn-trans/
├── conn_trans_prototype.py     # Main implementation & experiment
├── requirements.txt            # Dependencies
├── README.md                  # This file
├── results/                   # Experimental outputs
│   ├── comprehensive_comparison_*.json
│   ├── *_connection_matrix.png
│   └── best_model_*.pt
└── analysis/                  # Additional analysis tools
    ├── connection_analysis.py
    └── reasoning_visualization.py
```

## 🎯 Key Research Questions

1. **Can pure connection mechanisms perform reasoning?**

   - Hypothesis: Connection Matrix alone sufficient for basic reasoning
   - Test: Pure Conn-Trans vs Standard Transformer

2. **Do FFNs enhance connection-based reasoning?**

   - Hypothesis: FFN + Connection > Pure Connection
   - Test: Compare all three variants

3. **What reasoning patterns emerge in Connection Matrix?**

   - Analysis: Visualize and interpret learned C matrix
   - Goal: Understand interpretable reasoning mechanisms

4. **Parameter efficiency vs performance trade-off?**
   - Metric: Performance per parameter
   - Comparison: All models with same training setup

## 🔧 Hardware Requirements

### Recommended: RTX 4090

```
- VRAM: 24GB
- Training time: ~4 hours for full experiment
- Memory usage: ~16GB during training
- Batch size: 32 (optimal)
```

### Minimum: RTX 3090/4080

```
- VRAM: 16GB+
- Reduce batch_size to 24
- Reduce d_model to 384 if needed
- Training time: ~6 hours
```

### Memory Optimization

```python
# For lower VRAM GPUs
CONFIG["batch_size"] = 16
CONFIG["d_model"] = 384
CONFIG["num_ir"] = 768

# Enable mixed precision
model = model.half()
```

## 📊 Reproducibility

### Experiment Configuration

- **Random seed**: Set for reproducible results
- **Data split**: Fixed train/validation split
- **Hyperparameters**: Identical across all models
- **Training procedure**: Same optimizer, scheduler, epochs

### Output Files

```
comprehensive_comparison_YYYYMMDD_HHMMSS.json
├── results: {model_name: accuracy}
├── config: All hyperparameters
├── analysis: Performance gaps, improvements
└── timestamp: Experiment metadata
```

## 🚀 Future Directions

### Immediate Extensions

1. **Multi-task evaluation**: Test on all 20 bAbI tasks
2. **Reasoning depth analysis**: Vary num_steps (1,2,3,4,5,6)
3. **IR node scaling**: Test different num_ir sizes
4. **Connection sparsity**: Experiment with sparse connection patterns

### Advanced Research

1. **Larger datasets**: Scale to more complex reasoning tasks
2. **Adaptive reasoning**: Dynamic num_steps based on input complexity
3. **Hierarchical connections**: Multi-level connection matrices
4. **Transfer learning**: Pre-train connections on multiple tasks

## 📝 Citation

```bibtex
@article{conn-trans-2024,
  title={Conn-Trans: Iterative Semantic Reasoning via Dynamic Activation over Fixed-Base Nodes},
  author={[Author Name]},
  journal={arXiv preprint},
  year={2024},
  note={Novel connection-based reasoning architecture}
}
```

## 🤝 Contributing

We welcome contributions in:

- **New reasoning datasets** and evaluation metrics
- **Architecture improvements** and variants
- **Analysis tools** for interpretability
- **Performance optimizations** and scaling

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Run tests
pytest tests/

# Format code
black . && flake8 .
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **bAbI Tasks**: Facebook AI Research for reasoning benchmarks
- **Transformer Architecture**: "Attention is All You Need" (Vaswani et al.)
- **Neuroscience Inspiration**: Connection patterns in biological neural networks
- **Community**: Open-source ML community for tools and frameworks

---

## 🎯 Quick Results Summary

After running the experiment, you'll get:

✅ **Performance comparison** of three reasoning approaches  
✅ **Connection Matrix visualizations** showing learned patterns  
✅ **Parameter efficiency analysis** with detailed metrics  
✅ **Reasoning trace evolution** through multiple steps  
✅ **Comprehensive analysis report** with research insights

**Status**: 🚧 Research Prototype - Ready for Experimentation

For questions, issues, or collaboration opportunities, please open an issue or contact the maintainers.

---

_"Advancing interpretable reasoning through learnable connections"_
