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
- **Real-time Visualization**: Training progress and connection patterns

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
# Quick test (recommended first run)
python final_verification.py

# Small dataset training
python main.py --dataset strategyqa --model connection --model_size nano

# Medium dataset
python main.py --dataset logiqa --model connection --model_size micro

# Large dataset
python main.py --dataset multinli --model connection --model_size base

# Baseline comparison
python main.py --dataset multinli --model baseline --model_size base

# Dry run (verify setup without training)
python main.py --dataset strategyqa --model connection --model_size nano --dry_run
```

### System Verification

```bash
# Full system check
python final_verification.py

# Quick essential tests only
python final_verification.py --quick

# Verbose output for debugging
python final_verification.py --verbose
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

## 🔧 Project Structure

```
connection-transformer/
├── models/
│   ├── connection_transformer.py    # Novel architecture
│   └── baseline_transformer.py     # Parameter-matched baseline
├── configs/
│   ├── base_config.py              # Unified configuration system
│   └── *_config.py                 # Dataset-specific configs
├── dataset/
│   ├── base_dataset.py             # Abstract base class
│   ├── tokenizer_utils.py          # T5 tokenizer integration
│   └── *_dataset.py                # Dataset implementations
├── training/
│   ├── trainer.py                  # Enhanced trainer with visualization
│   └── data_collator.py            # T5 data collation
├── utils/
│   ├── metrics.py                  # Dataset-specific evaluation
│   └── visualization.py           # Real-time plotting
├── main.py                         # Experiment runner
├── analyze_results.py              # Results analysis
└── final_verification.py          # System verification
```

## 🎨 Real-time Visualizations

During training, the system automatically generates:

### Training Progress

- **Training curves**: Loss and accuracy over epochs
- **Reasoning efficiency**: Steps per reasoning cycle
- **Performance breakdown**: Correct vs incorrect predictions

### Connection Analysis (Connection Transformer)

- **Connection matrices**: Heatmaps of slot-to-slot influences
- **Sparsity evolution**: How connections become sparse over time
- **Orthogonality quality**: Regularization effectiveness

### Generated Files

```
outputs/
├── visualizations/
│   ├── training_curves_epoch_X.png
│   ├── connection_matrix_epoch_X.png
│   ├── accuracy_breakdown_epoch_X.png
│   └── connection_analysis_epoch_X.txt
├── analysis/
│   └── training_report_MODEL_DATASET.md
├── results_MODEL_DATASET_TIMESTAMP.json
└── best_MODEL_DATASET.pt
```

## 📈 Key Components

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

## 🔍 Results Analysis

### Automatic Analysis

```bash
# Analyze all results in outputs directory
python analyze_results.py --results_dir ./outputs --output_dir ./analysis
```

### Generated Analysis

- **Performance comparison** between Connection and Baseline models
- **Connection matrix visualization** showing slot interactions
- **Reasoning efficiency analysis** for adaptive steps
- **Comprehensive experiment report** with improvement metrics

## 💡 Usage Examples

### Training Examples

```bash
# Quick nano model on StrategyQA
python main.py --dataset strategyqa --model connection --model_size nano

# Compare Connection vs Baseline on LogiQA
python main.py --dataset logiqa --model connection --model_size micro
python main.py --dataset logiqa --model baseline --model_size micro

# Large-scale experiment on MultiNLI
python main.py --dataset multinli --model connection --model_size base --output_dir ./experiments/multinli
```

### Development Workflow

```bash
# 1. Verify system setup
python final_verification.py --quick

# 2. Run small test
python main.py --dataset strategyqa --model connection --model_size nano

# 3. Check visualizations
ls outputs/visualizations/

# 4. Analyze results
python analyze_results.py --results_dir ./outputs --output_dir ./analysis

# 5. Compare models
python main.py --dataset strategyqa --model baseline --model_size nano
```

## 🎯 Training Features

### Enhanced Trainer

- **Mixed Precision**: BFloat16/Float16 support
- **Gradient Clipping**: Stable training
- **Early Stopping**: Prevent overfitting
- **Real-time Monitoring**: Live visualization generation
- **Reasoning Tracking**: Connection Transformer reasoning steps

### Visualization During Training

- Connection matrices updated every 2 epochs
- Training curves updated every epoch
- Accuracy breakdown with sample predictions
- Orthogonality and sparsity analysis

### Sample Training Output

```
🚀 Training 3 epochs
Epoch 1/3
  Train Loss: 1.2340
  Eval Loss:  1.1890
  Accuracy:   0.4500
  Avg Steps:  2.1
  📈 Visualizations saved to ./outputs/visualizations/
  💾 New best: 0.4500

Epoch 2/3
  Train Loss: 0.8760
  Eval Loss:  0.8234
  Accuracy:   0.6250
  Avg Steps:  1.8
  📈 Visualizations saved to ./outputs/visualizations/
  💾 New best: 0.6250
```

## 🔬 Research Applications

This implementation supports research into:

- **Adaptive reasoning** mechanisms in Transformers
- **Bilinear connection** learning and sparsity patterns
- **Orthogonal regularization** effects on representation learning
- **Parameter efficiency** in reasoning architectures
- **Slot-based reasoning** and attention patterns

## 📊 Configuration System

### Flexible Configuration

```python
# Example: Custom configuration
from configs.base_config import BaseConfig

config = BaseConfig().set_size("micro").set_dataset(
    "custom_dataset",
    max_reasoning_steps=5,
    orthogonal_weight=0.05
)
```

### Dataset-Specific Optimizations

- **StrategyQA**: Optimized for Yes/No reasoning
- **LogiQA**: Multi-choice logic with option processing
- **GSM8K**: Mathematical reasoning with number extraction
- **MultiNLI**: Natural language inference with premise-hypothesis pairs

## 🛠️ Advanced Features

### Model Analysis

```python
# Get detailed connection analysis
analysis = model.get_connection_analysis()
print(f"Sparsity: {analysis['sparsity_ratio']:.3f}")
print(f"Orthogonality Quality: {analysis['orthogonality_quality']:.3f}")
```

### Custom Datasets

```python
# Extend BaseReasoningDataset for new datasets
from dataset.base_dataset import BaseReasoningDataset

class CustomDataset(BaseReasoningDataset):
    @property
    def dataset_name(self):
        return "CustomDataset"

    def _load_raw_data(self):
        # Your data loading logic
        pass

    def _process_item(self, item, idx):
        # Your preprocessing logic
        return {
            'input_text': processed_input,
            'target_text': processed_target,
            'metadata': {...}
        }
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   ```bash
   # Use smaller batch size or model size
   python main.py --dataset strategyqa --model connection --model_size nano
   ```

2. **Import Errors**

   ```bash
   # Run verification first
   python final_verification.py
   ```

3. **Dataset Loading Issues**
   ```bash
   # Check internet connection and dataset availability
   python final_verification.py --verbose
   ```

### Performance Tips

- Start with `nano` or `micro` model sizes for initial experiments
- Use `--dry_run` to verify setup before full training
- Monitor GPU memory with `nvidia-smi`
- Check `outputs/visualizations/` for training progress

## 📄 Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{connection-transformer,
  title={Connection Transformer: Bilinear Slot-to-Slot Connections for Adaptive Reasoning},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]/connection-transformer}
}
```

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. **Run verification**: `python final_verification.py`
2. **Test your changes**: Run on small dataset first
3. **Check visualizations**: Ensure plots generate correctly
4. **Submit PR**: Include verification results

For questions or issues, please open a GitHub issue with:

- System verification output
- Error logs
- Minimal reproduction example
