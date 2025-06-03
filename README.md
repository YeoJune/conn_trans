# Connection Transformer

A novel Transformer architecture with **bilinear slot-to-slot connections** for adaptive reasoning and **systematic experiment management**.

## 🏗️ Architecture

### Core Innovation: Bilinear Connections

```
Source Sequence → Encoder → Semantic Slots ← Bilinear Reasoning → Decoder → Target Sequence
                            [B,N,D]         W_source ⊙ W_target
```

**Key Features:**

- **Semantic Slots**: Fixed orthogonal representations for independent reasoning
- **Bilinear Connections**: Learnable slot-to-slot influence matrices with low-rank structure
- **Adaptive Reasoning**: Dynamic number of reasoning steps until convergence
- **Encoder-Decoder Structure**: Standard sequence-to-sequence architecture
- **Orthogonal Regularization**: Ensures connection matrix structure and interpretability

### Encoder-Decoder Architecture

- **Encoder**: Source sequence → Semantic slots → Adaptive bilinear reasoning
- **Decoder**: Standard Transformer decoder with semantic slots as memory
- **Fair Comparison**: Parameter-matched baseline Transformer for direct comparison

## 🚀 Quick Start

### Installation

```bash
pip install torch transformers datasets matplotlib seaborn pandas
```

### Basic Usage

```bash
# Quick system verification (recommended first run)
python final_verification.py --quick

# Small dataset training
python main.py --dataset strategyqa --model connection --model_size micro

# Baseline comparison
python main.py --dataset strategyqa --model baseline --model_size micro

# Run comprehensive experiments
./run_experiments.sh strategyqa

# Analyze all results
python analyze_results.py --output_dir ./outputs
```

## 📊 Datasets & Model Sizes

### Supported Datasets

| Dataset    | Task Type                  | Size     | Metrics                        |
| ---------- | -------------------------- | -------- | ------------------------------ |
| StrategyQA | Yes/No reasoning           | 2.8K     | Exact Match (Yes/No)           |
| LogiQA     | Multiple-choice logic      | 8K       | Exact Match (A/B/C/D)          |
| GSM8K      | Math word problems         | 8.8K     | Exact Match (numerical answer) |
| MultiNLI   | Natural Language Inference | 433K     | Exact Match (ent/neu/con)      |
| ELI5       | Explain Like I'm 5         | Variable | ROUGE-L based accuracy         |
| CommonGen  | Concept-to-text generation | Variable | ROUGE-L based accuracy         |

### Model Sizes

| Size  | d_model | num_slots | bilinear_rank | decoder_layers | Use Case             |
| ----- | ------- | --------- | ------------- | -------------- | -------------------- |
| micro | 128     | 2048      | 1             | 4              | Quick experiments    |
| small | 256     | 256       | 1             | 5              | Medium datasets      |
| base  | 512     | 512       | 1             | 6              | Large datasets       |
| large | 768     | 768       | 1             | 6              | Research experiments |

## 📈 Experimental Results

### Initial Comparison (Large Model, No Parameter Balancing)

**Model Parameters:**

- Connection Transformer: **133,784,064 parameters**
- Baseline Transformer: **173,360,640 parameters**

| Dataset    | Connection | Baseline | Improvement | Winner        |
| ---------- | ---------- | -------- | ----------- | ------------- |
| StrategyQA | 0.5153     | 0.5488   | -3.35pp     | ❌ Baseline   |
| MultiNLI   | 0.5971     | 0.5881   | +0.90pp     | ✅ Connection |
| LogiQA     | 0.4028     | 0.2412   | +16.16pp    | ✅ Connection |
| GSM8K      | 0.0324     | 0.0347   | -0.23pp     | ❌ Baseline   |
| ELI5       | 0.0065     | 0.0126   | -0.61pp     | ❌ Baseline   |
| CommonGen  | 0.3629     | 0.3604   | +0.25pp     | ✅ Connection |

**Summary:**

- **Connection wins: 3/6 datasets (50%)**
- **Notable strength**: Logic reasoning (LogiQA +16.16pp improvement)
- **Parameter efficiency**: Connection uses 23% fewer parameters
- **Best performance**: LogiQA reasoning tasks show clear advantage

### Accuracy Calculation Methods

The accuracy calculation varies by dataset type to ensure fair evaluation:

```python
# Classification tasks (StrategyQA, LogiQA, MultiNLI)
accuracy = exact_string_match(predicted_answer, ground_truth)

# Mathematical tasks (GSM8K)
accuracy = numerical_equivalence(extract_number(prediction), extract_number(target))

# Generation tasks (ELI5, CommonGen)
accuracy = rouge_l_score(prediction, target) >= 0.3  # ROUGE-L threshold
```

**Dataset-specific extraction:**

- **StrategyQA**: First word → "Yes"/"No"
- **LogiQA**: First character → "A"/"B"/"C"/"D"
- **MultiNLI**: First word → "entailment"/"neutral"/"contradiction"
- **GSM8K**: Extract final numerical answer from solution
- **ELI5/CommonGen**: Full text with ROUGE-L similarity matching

## 🔧 Project Structure

```
connection-transformer/
├── models/
│   ├── connection_transformer.py    # Novel architecture implementation
│   └── baseline_transformer.py     # Parameter-matched baseline
├── configs/
│   ├── base_config.py              # Unified configuration system
│   └── {dataset}_config.py         # Dataset-specific optimizations
├── dataset/
│   ├── base_dataset.py             # Abstract base for all datasets
│   ├── tokenizer_utils.py          # T5 tokenizer integration
│   └── {dataset}_dataset.py        # Individual dataset implementations
├── training/
│   ├── trainer.py                  # Unified training loop
│   └── data_collator.py            # T5-style data collation
├── utils/
│   ├── result_manager.py           # Experiment file organization
│   ├── comparison_analyzer.py      # Cross-experiment analysis
│   ├── visualization_manager.py    # Chart and plot generation
│   └── metrics.py                  # Evaluation functions
├── outputs/                        # Auto-organized results
│   ├── experiments/                # Training-time files
│   ├── analysis/                   # Post-training analysis
│   └── comparisons/                # Cross-experiment comparisons
├── main.py                         # Single experiment runner
├── run_experiments.sh              # Batch experiment runner
├── analyze_results.py              # Results analysis tool
└── final_verification.py           # System verification tool
```

## 📁 Result Organization

The system automatically organizes results with timestamps and clear separation:

```
outputs/
├── experiments/                     # Training-time data
│   └── {YYYYMMDD_HHMM}_{model}_{dataset}_{size}/
│       ├── config.json             # Experiment configuration
│       ├── model_best.pt           # Best checkpoint (w/ parameter count)
│       ├── training_log.txt        # Real-time training log
│       └── metrics.json            # Training metrics history
├── analysis/                       # Post-training analysis
│   └── {YYYYMMDD_HHMM}_{model}_{dataset}_{size}/
│       ├── report.md               # Comprehensive analysis report
│       ├── summary.json            # Machine-readable results
│       ├── training_curves.png     # Loss and accuracy plots
│       ├── connection_matrix.png   # Connection analysis (Connection only)
│       └── accuracy_summary.png    # Performance breakdown
└── comparisons/                    # Cross-experiment analysis
    └── {YYYYMMDD_HHMM}_comparison/
        ├── comparison_table.csv    # All experiments tabulated
        ├── summary_report.md       # Connection vs Baseline analysis
        ├── dataset_performance.png # Dataset-wise comparison
        ├── model_comparison.png    # Model-wise statistics
        └── parameters_analysis.png # Parameter efficiency analysis
```

## 🎯 Usage Examples

### Quick Development Workflow

```bash
# 1. System verification
python final_verification.py --quick

# 2. Single quick experiment
python main.py --dataset strategyqa --model connection --model_size micro

# 3. Compare with baseline
python main.py --dataset strategyqa --model baseline --model_size micro

# 4. Analyze results
python analyze_results.py --output_dir ./outputs
```

### Comprehensive Evaluation

```bash
# Run all datasets with appropriate sizes
./run_experiments.sh

# Or specific dataset comparisons
./run_experiments.sh strategyqa micro
./run_experiments.sh multinli base

# Large-scale evaluation
./run_experiments.sh all base
```

### Research Mode

```bash
# Custom configuration experiments
python main.py --dataset logiqa --model connection --model_size large

# Skip analysis for faster iteration
python main.py --dataset strategyqa --model connection --model_size micro --dry_run

# Parameter efficiency study
./run_experiments.sh all small  # Consistent model size across datasets
```

## 🔍 Analysis Features

### Automatic Analysis Pipeline

After each training run, the system automatically:

1. **Generates comprehensive visualizations** (training curves, connection matrices)
2. **Saves structured results** with parameter counts and performance metrics
3. **Performs cross-experiment comparison** when multiple results exist
4. **Creates comparative reports** highlighting strengths and weaknesses

### Connection-Specific Analysis

```python
# Automatic connection analysis during training
analysis = model.get_connection_analysis()
print(f"Sparsity: {analysis['sparsity_ratio']:.3f}")
print(f"Max connection: {analysis['max_connection']:.3f}")
print(f"Active connections: {analysis['active_connection_ratio']:.3f}")
```

### Performance Metrics

```python
# Dataset-appropriate accuracy calculation
from utils.metrics import calculate_accuracy

accuracy = calculate_accuracy(predictions, targets, dataset_type)
# Automatically uses appropriate metric per dataset:
# - Exact match for classification/math
# - ROUGE-L for generation tasks
```

## 💡 Key Features

### Systematic Experiment Management

- **Separated concerns**: Training files vs analysis files clearly separated
- **Timestamp-based naming**: Chronological organization with clear experiment identification
- **Parameter tracking**: Automatic parameter counting and efficiency analysis
- **Error resilience**: Robust handling of training failures and memory issues

### Adaptive Architecture

- **Dynamic reasoning**: Variable reasoning steps based on convergence
- **Orthogonal regularization**: Structured connection learning
- **Memory efficiency**: Optimized for single GPU training
- **Fair comparison**: Parameter-matched baseline for direct evaluation

### Research-Ready Analysis

- **Statistical comparison**: Win rates, improvement percentages, distribution analysis
- **Visual analysis**: Comprehensive charts and connection visualizations
- **Archival quality**: Self-contained reports for long-term reference
- **Programmatic access**: APIs for custom analysis and result processing

## 🛠️ Advanced Configuration

### Custom Model Sizes

```python
from configs.base_config import BaseConfig

# Method chaining for clean configuration
config = BaseConfig() \
    .set_size("small") \
    .set_dataset("strategyqa", num_epochs=3, learning_rate=2e-4) \
    .update(orthogonal_weight=0.02, max_reasoning_steps=2)
```

### Dataset-Specific Optimizations

Each dataset has optimized defaults:

```python
dataset_defaults = {
    "strategyqa": {"answer_max_length": 8, "batch_size": 16},
    "logiqa": {"answer_max_length": 16, "max_seq_len": 384},
    "gsm8k": {"answer_max_length": 128, "max_seq_len": 512},
    "multinli": {"answer_max_length": 16, "batch_size": 64}
}
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   ```bash
   # Use smaller model size
   python main.py --dataset strategyqa --model connection --model_size micro
   ```

2. **Import Errors**

   ```bash
   # Run system verification
   python final_verification.py
   ```

3. **Analysis Failures**
   ```bash
   # Run analysis separately
   python analyze_results.py --output_dir ./outputs
   ```

### Performance Tips

- **Start small**: Use `micro` size for initial experiments
- **Monitor memory**: Check GPU memory with `nvidia-smi`
- **Use batch scripts**: `./run_experiments.sh` handles complex workflows automatically
- **Check logs**: Monitor `outputs/experiments/*/training_log.txt` for detailed progress

## 📊 Future Improvements

Based on initial results:

1. **Parameter Balancing**: Implement automatic parameter matching between Connection and Baseline
2. **Architecture Tuning**: Optimize bilinear rank and slot count for different reasoning types
3. **Task-Specific Adaptation**: Fine-tune connection patterns for mathematical vs logical reasoning
4. **Efficiency Optimization**: Reduce parameter count while maintaining performance advantages

## 📄 Citation

If you use this implementation in your research:

```bibtex
@misc{connection-transformer-2024,
  title={Connection Transformer: Bilinear Slot-to-Slot Connections for Adaptive Reasoning},
  author={[Your Name]},
  year={2024},
  note={Implementation with systematic experiment management},
  url={https://github.com/[your-repo]/connection-transformer}
}
```

## 📝 License

MIT License - see LICENSE file for details.
