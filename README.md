# README

# Connection Transformer

A novel Transformer architecture with **bilinear slot-to-slot connections** for adaptive reasoning.

## 🏗️ Architecture

### Core Innovation: Bilinear Connections

```
Input Tokens → Semantic Slots → Bilinear Reasoning → Output Tokens
                     ↓
              Slot-to-Slot Connections
           W_source @ H_i @ W_target → H_j
```

**Key Features:**

- **Semantic Slots**: Fixed orthogonal representations for independent reasoning
- **Bilinear Connections**: Learnable slot-to-slot influence matrices
- **Adaptive Reasoning**: Dynamic number of reasoning steps until convergence
- **Orthogonal Regularization**: Ensures connection matrix orthogonality

### Encoder-Decoder Architecture

- **Encoder**: Input → Semantic slots → Bilinear reasoning
- **Decoder**: Standard Transformer decoder with cross-attention to semantic slots
- **Fair Comparison**: Parameter-matched baseline Transformer

## 🚀 Quick Start

### Installation

```bash
pip install torch transformers datasets matplotlib seaborn pandas
```

### Basic Usage

```bash
# Small dataset (recommended for testing)
python main.py --dataset strategyqa --model connection --model_size nano

# Medium dataset
python main.py --dataset logiqa --model connection --model_size micro

# Large dataset
python main.py --dataset multinli --model connection --model_size base

# Baseline comparison
python main.py --dataset multinli --model baseline --model_size base
```

### Verification

```bash
python final_verification.py
```

## 📊 Datasets & Model Sizes

### Supported Datasets

- **StrategyQA**: Yes/No reasoning (2.8K samples)
- **LogiQA**: Multiple-choice logic (8K samples)
- **GSM8K**: Math word problems (8.8K samples)
- **MultiNLI**: Natural Language Inference (433K samples)

### Model Sizes

| Size  | d_model | num_slots | bilinear_rank | Use Case        |
| ----- | ------- | --------- | ------------- | --------------- |
| nano  | 32      | 8         | 2             | StrategyQA only |
| micro | 64      | 16        | 4             | Small datasets  |
| tiny  | 128     | 32        | 8             | Medium datasets |
| small | 192     | 48        | 12            | Experimental    |
| base  | 256     | 64        | 16            | Large datasets  |

## 🔧 Key Components

### Models

- `ConnectionTransformer`: Novel architecture with bilinear connections
- `BaselineTransformer`: Parameter-matched standard Transformer

### Configs

- Automatic parameter matching between Connection and Baseline models
- Dataset-specific optimizations and overfitting prevention
- T5-optimized training settings

### Training

- Encoder-Decoder compatible data processing
- Mixed precision (BFloat16/Float16) support
- Orthogonal regularization loss
- Early stopping and gradient clipping

### Analysis

```bash
python analyze_results.py --results_dir ./outputs --output_dir ./analysis
```

## 📈 Key Features

### Bilinear Connections

```python
# Slot i influences slot j through bilinear transformation
influence = W_source[i,j] @ H_state[i] @ W_target[i,j]
H_state[j] += F.relu(influence)
```

### Orthogonal Regularization

```python
# Ensures W_source^T @ W_source = I and W_target @ W_target^T = I
orth_loss = ||W_source^T @ W_source - I||_F^2 + ||W_target @ W_target^T - I||_F^2
```

### Adaptive Reasoning

```python
for step in range(max_reasoning_steps):
    influence = bilinear_transform(H_state)
    H_state = H_state + F.relu(influence)
    if convergence_achieved:
        break
```

## 🎯 Results Analysis

The system automatically generates:

- **Performance comparison** between Connection and Baseline models
- **Connection matrix visualization** showing slot interactions
- **Reasoning efficiency analysis** for adaptive steps
- **Comprehensive experiment report** with improvement metrics

## 🔍 Verification & Testing

### Quick Test

```bash
python final_verification.py
```

### Full System Test

```bash
# Test all components
python -c "
from final_verification import *
test_basic_imports()
test_model_creation()
test_config_system()
test_training_setup()
"
```

## 📚 Project Structure

```
connection-transformer/
├── models/
│   ├── connection_transformer.py    # Novel architecture
│   └── baseline_transformer.py     # Parameter-matched baseline
├── configs/
│   ├── base_config.py              # Unified configuration system
│   ├── strategyqa_config.py        # Dataset-specific configs
│   ├── logiqa_config.py
│   ├── gsm8k_config.py
│   └── multinli_config.py
├── dataset/
│   ├── tokenizer_utils.py          # T5 tokenizer integration
│   └── *_dataset.py                # Dataset loaders
├── training/
│   ├── trainer.py                  # Encoder-decoder trainer
│   └── data_collator.py            # T5 data collation
├── utils/
│   ├── metrics.py                  # T5-optimized evaluation
│   └── visualization.py           # Analysis plots
├── main.py                         # Experiment runner
├── analyze_results.py              # Results analysis
└── final_verification.py          # System verification
```

## 🎨 Visualizations

The system generates:

- **Connection matrices**: Heatmaps of slot-to-slot influences
- **Training curves**: Loss and accuracy over epochs
- **Reasoning patterns**: Distribution of reasoning steps
- **Performance comparisons**: Bar charts comparing models

## 💡 Tips for Success

### Dataset Selection

- **Start small**: Use StrategyQA (nano) or LogiQA (micro) for initial testing
- **Scale up**: Try MultiNLI (base) for serious experimentation
- **Avoid overfitting**: Stick to recommended model sizes

### Training Tips

- Monitor orthogonal regularization loss for Connection Transformer
- Use early stopping to prevent overfitting
- BFloat16 precision works well with T5 tokenizer

### Analysis

- Compare Connection vs Baseline on same dataset/size
- Check connection sparsity and reasoning step convergence
- Use visualization to understand learned patterns

## 🔬 Research Applications

This implementation supports research into:

- **Adaptive reasoning** mechanisms in Transformers
- **Bilinear connection** learning and sparsity
- **Orthogonal regularization** effects on representation learning
- **Parameter efficiency** in reasoning architectures

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please check out our contribution guidelines and open an issue or pull request.
