# Connection Transformer

A novel Transformer architecture with **bilinear slot-to-slot connections** for adaptive reasoning and **systematic experiment management**.

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
- **Systematic Result Management**: Automated experiment tracking and analysis

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
python main.py --dataset strategyqa --model connection --model_size micro --dry_run

# Small dataset training with automatic analysis
python main.py --dataset strategyqa --model connection --model_size micro

# Medium dataset
python main.py --dataset logiqa --model connection --model_size small

# Large dataset
python main.py --dataset multinli --model connection --model_size base

# Baseline comparison
python main.py --dataset multinli --model baseline --model_size base

# Skip automatic analysis (faster)
python main.py --dataset strategyqa --model connection --skip_analysis
```

## 📊 Datasets & Model Sizes

### Supported Datasets

- **StrategyQA**: Yes/No reasoning (2.8K samples)
- **LogiQA**: Multiple-choice logic (8K samples)
- **GSM8K**: Math word problems (8.8K samples)
- **MultiNLI**: Natural Language Inference (433K samples)

### Model Sizes

| Size    | d_model | num_slots | bilinear_rank | Use Case              |
| ------- | ------- | --------- | ------------- | --------------------- |
| micro   | 64      | 16        | 4             | Quick experiments     |
| x-small | 128     | 32        | 8             | Small-medium datasets |
| small   | 192     | 48        | 12            | Medium datasets       |
| base    | 256     | 64        | 16            | Large datasets        |
| large   | 512     | 128       | 32            | Research experiments  |

## 🔧 Improved Project Structure

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
│   ├── trainer.py                  # Streamlined trainer
│   └── data_collator.py            # T5 data collation
├── utils/                          # All implementation logic
│   ├── result_manager.py           # Experiment management
│   ├── visualization_manager.py    # Chart generation
│   ├── comparison_analyzer.py      # Cross-experiment analysis
│   └── metrics.py                  # Evaluation metrics
├── outputs/                        # Organized results
│   ├── experiments/                # Training-time files
│   ├── analysis/                   # Post-training analysis
│   └── comparisons/                # Cross-experiment comparisons
└── main.py                         # Experiment runner with auto-analysis
```

## 📁 Systematic Result Organization

### File Structure

```
outputs/
├── experiments/                     # Training-time files
│   └── {timestamp}_{model}_{dataset}_{size}/
│       ├── config.json             # Experiment configuration
│       ├── model_best.pt           # Best model checkpoint
│       ├── training_log.txt        # Real-time training log
│       └── metrics.json            # Live metrics updates
├── analysis/                       # Post-training analysis
│   └── {timestamp}_{model}_{dataset}_{size}/
│       ├── report.md               # Comprehensive report
│       ├── summary.json            # Machine-readable summary
│       ├── final_curves.png        # Training curves
│       ├── connection_matrix.png   # Connection analysis (Connection only)
│       └── accuracy_breakdown.png  # Performance analysis
└── comparisons/                    # Cross-experiment analysis
    └── {timestamp}_comparison/
        ├── comparison_table.csv    # All experiments comparison
        ├── summary_report.md       # Connection vs Baseline analysis
        ├── dataset_performance.png
        ├── model_comparison.png
        └── size_analysis.png
```

### Naming Convention

- **Format**: `{YYYYMMDD_HHMM}_{model}_{dataset}_{size}`
- **Example**: `20250602_1430_connection_strategyqa_micro`
- **Benefits**: Chronological sorting, clear identification, archival-friendly

## 🎨 Automated Visualizations

### Training-Time Features

- **Real-time logging**: Comprehensive training progress tracking
- **Live metrics**: JSON-based metrics updates during training
- **Checkpoint management**: Automatic best model saving

### Post-Training Analysis

- **Training Curves**: Loss, accuracy, and reasoning steps over epochs
- **Connection Matrix**: Slot-to-slot interaction heatmaps (Connection Transformer)
- **Accuracy Breakdown**: Detailed performance analysis with sample predictions
- **Connection Analysis**: Sparsity, orthogonality, and efficiency metrics

### Cross-Experiment Comparison

- **Performance Charts**: Dataset-wise and model-wise comparisons
- **Statistical Analysis**: Win rates, average improvements, distribution plots
- **Comprehensive Reports**: Markdown reports with actionable insights

## 📈 Enhanced Configuration System

### Flexible Configuration

```python
# Example: Custom configuration
from configs.base_config import BaseConfig

# Method chaining for clean configuration
config = BaseConfig() \
    .set_size("small") \
    .set_dataset("strategyqa", num_epochs=5, learning_rate=2e-4) \
    .update(orthogonal_weight=0.05, max_reasoning_steps=3)
```

### Size Configurations

```python
# Available sizes with automatic parameter scaling
sizes = ["micro", "x-small", "small", "base", "large"]

# Each size automatically configures:
# - Model dimensions (d_model, num_slots, bilinear_rank)
# - Training parameters (batch_size, learning_rate)
# - Sequence lengths and reasoning steps
```

### Dataset-Specific Optimizations

```python
# Automatic dataset optimization
dataset_configs = {
    "strategyqa": {"task_prefix": "strategy", "answer_max_length": 8},
    "logiqa": {"task_prefix": "reason", "answer_max_length": 16},
    "gsm8k": {"task_prefix": "solve", "answer_max_length": 32},
    "multinli": {"task_prefix": "infer", "answer_max_length": 16}
}
```

## 🔍 Automatic Analysis Pipeline

### Training Completion

After each training run, the system automatically:

1. **Generates final analysis** with comprehensive visualizations
2. **Saves structured results** in JSON and Markdown formats
3. **Performs cross-experiment comparison** if multiple experiments exist
4. **Creates comparison reports** highlighting Connection vs Baseline performance

### Analysis Features

- **Win Rate Analysis**: How often Connection Transformer outperforms Baseline
- **Improvement Metrics**: Percentage point improvements across datasets
- **Statistical Significance**: Distribution plots and confidence analysis
- **Archival Reports**: Self-contained analysis for long-term reference

### Sample Analysis Output

```
🔍 자동 비교 분석 시작...
📊 4개 실험 발견
💾 비교 테이블: comparison_table.csv
📊 데이터셋 성능 차트 저장
📈 모델 비교 차트 저장
📏 크기 분석 차트 저장
📋 요약 리포트 저장: summary_report.md

📊 비교 분석 완료!
   총 실험: 4개
   최고 성능: 0.7250
   결과 위치: outputs/comparisons/20250602_1445_comparison
```

## 🎯 Automated Experiment Runner

### Quick Batch Experiments

```bash
# Make script executable
chmod +x run_experiments.sh

# Run all datasets with optimal sizes
./run_experiments.sh

# Single dataset with default size
./run_experiments.sh strategyqa

# Single dataset with specific size
./run_experiments.sh multinli base

# All datasets with same size
./run_experiments.sh all micro
```

### Experiment Script Features

- **Smart Defaults**: Automatically selects optimal model sizes per dataset
- **Flexible Arguments**: Support for specific datasets and model sizes
- **Error Resilience**: Continues other experiments if one fails
- **Automatic Analysis**: Runs comprehensive comparison after all experiments
- **Progress Tracking**: Clear status updates for each experiment

### Default Model Sizes

| Dataset    | Default Size | Reasoning                    |
| ---------- | ------------ | ---------------------------- |
| StrategyQA | micro        | Small dataset, quick tests   |
| LogiQA     | small        | Medium complexity reasoning  |
| GSM8K      | small        | Mathematical reasoning tasks |
| MultiNLI   | base         | Large dataset, full capacity |

### Sample Output

```bash
🚀 Connection Transformer Experiments
=====================================
📊 Dataset: all
📏 Model Size: default

🔍 Quick verification...
✅ System ready!

📊 Dataset: strategyqa (size: micro)
--------------------------------
🔄 Running: strategyqa - connection (micro)
✅ Completed: strategyqa - connection (micro)

🔄 Running: strategyqa - baseline (micro)
✅ Completed: strategyqa - baseline (micro)

📈 Running final analysis...
============================
🎉 Analysis Complete!
📊 Total experiments: 8
🏆 Best accuracy: 0.7250
📁 Results: experiments_output/comparisons/20250602_1445_comparison
```

## 💡 Usage Examples

### Batch Experiments

```bash
# Quick comparison across all datasets
./run_experiments.sh

# Development workflow
./run_experiments.sh strategyqa micro  # Quick test
./run_experiments.sh logiqa small      # Medium test
./run_experiments.sh multinli base     # Full test

# Research experiments
./run_experiments.sh all small         # Consistent comparison
./run_experiments.sh all base          # Full-scale evaluation
```

### Single Experiments

```bash
# Quick test (recommended first run)
python main.py --dataset strategyqa --model connection --model_size micro --dry_run

# Small dataset training with automatic analysis
python main.py --dataset strategyqa --model connection --model_size micro

# Medium dataset
python main.py --dataset logiqa --model connection --model_size small

# Large dataset
python main.py --dataset multinli --model connection --model_size base

# Baseline comparison
python main.py --dataset multinli --model baseline --model_size base

# Skip automatic analysis (faster)
python main.py --dataset strategyqa --model connection --skip_analysis
```

## 🎯 Key Improvements

### Experiment Management

- **Separated file creation**: Training vs analysis files clearly separated
- **Systematic naming**: Consistent, archive-friendly naming convention
- **Reduced I/O overhead**: No visualization during training, batch creation after
- **Automatic analysis**: No need to run separate analysis scripts

### Visualization Quality

- **Information density**: More informative charts with better design
- **Focused analysis**: Only essential visualizations, higher quality
- **Connection-specific**: Specialized analysis for Connection Transformer features
- **Cross-experiment**: Comparative analysis across multiple runs

### Code Organization

- **Utils-based implementation**: All logic in utils/, other files just use them
- **Modular design**: Easy to extend with new analysis features
- **Clean interfaces**: Simple, consistent APIs across modules
- **Error resilience**: Robust error handling and fallback mechanisms

## 🔬 Research Applications

This implementation supports research into:

- **Adaptive reasoning** mechanisms with automatic step tracking
- **Bilinear connection** learning and sparsity pattern analysis
- **Orthogonal regularization** effects with quantitative monitoring
- **Parameter efficiency** through systematic baseline comparisons
- **Slot-based reasoning** with detailed connection visualization

## 🛠️ Advanced Features

### Connection Analysis

```python
# Automatic connection analysis during training
analysis = model.get_connection_analysis()
# Returns: sparsity_ratio, max_connection, orthogonality_quality
```

### Custom Metrics

```python
# Enhanced metrics with breakdown analysis
from utils.metrics import get_accuracy_breakdown

breakdown = get_accuracy_breakdown(predictions, targets, "strategyqa")
# Returns: accuracy, correct count, individual sample analysis
```

### Programmatic Access

```python
# Access comparison results programmatically
from utils.comparison_analyzer import ComparisonAnalyzer

analyzer = ComparisonAnalyzer("./outputs")
summary = analyzer.get_comparison_summary()
# Returns: status, total_experiments, best_accuracy, etc.
```

## 📊 Sample Results

### Training Output

```
🚀 Connection Transformer Experiment
   Dataset: strategyqa
   Model: connection
   Size: micro
   Output: outputs
📋 설정 저장: config.json
🚀 Training 3 epochs

Epoch 1/3
  Train Loss: 1.2340
  Eval Loss:  1.1890
  Accuracy:   0.4500
  Avg Steps:  2.1
  💾 New best: 0.4500

Epoch 2/3
  Train Loss: 0.8760
  Eval Loss:  0.8234
  Accuracy:   0.6250
  Avg Steps:  1.8
  💾 New best: 0.6250

✅ Training completed! Best accuracy: 0.6250
📊 최종 분석 시작...
📋 리포트 생성: report.md
📊 시각화 완료
💾 요약 저장: summary.json
✅ 분석 완료: outputs/analysis/20250602_1430_connection_strategyqa_micro

🔍 자동 비교 분석 시작...
📊 비교 분석 완료!
   총 실험: 2개
   최고 성능: 0.6250
   결과 위치: outputs/comparisons/20250602_1432_comparison
```

### Development and Debugging

```bash
# Quick test with minimal resources
python main.py --dataset strategyqa --model connection --model_size micro --dry_run

# Training with detailed logging
python main.py --dataset strategyqa --model connection --model_size micro

# Check real-time progress
tail -f outputs/experiments/*/training_log.txt

# Skip automatic analysis for faster iteration
python main.py --dataset strategyqa --model connection --skip_analysis
```

### Research Workflow

```bash
# 1. Quick system verification
./run_experiments.sh strategyqa micro

# 2. Comprehensive comparison
./run_experiments.sh all

# 3. Check results
ls outputs/comparisons/
cat outputs/comparisons/*/summary_report.md

# 4. Custom experiments
python main.py --dataset gsm8k --model connection --model_size base
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   ```bash
   # Use smaller model size
   ./run_experiments.sh strategyqa micro
   # Or run individual experiments
   python main.py --dataset strategyqa --model connection --model_size micro
   ```

2. **Script Permission Denied**

   ```bash
   # Make script executable
   chmod +x run_experiments.sh
   ```

3. **Experiment Failures**

   ```bash
   # Run verification first
   python main.py --dataset strategyqa --model connection --dry_run
   # Check individual components
   ./run_experiments.sh strategyqa micro
   ```

4. **Analysis Failures**

   ```bash
   # Skip automatic analysis and run manually
   python main.py --dataset strategyqa --model connection --skip_analysis
   ```

5. **File Permission Issues**
   ```bash
   # Check outputs directory permissions
   chmod -R 755 outputs/
   rm -rf outputs/  # Clean start if needed
   ```

### Performance Tips

- **Start small**: Use `micro` model size for initial experiments
- **Use batch script**: `./run_experiments.sh` handles everything automatically
- **Monitor progress**: Check `outputs/experiments/*/training_log.txt` for detailed progress
- **Development mode**: Use `--skip_analysis` for faster iteration during development
- **Memory management**: Kill other GPU processes before large experiments

### Quick Diagnostics

```bash
# System check
python main.py --dataset strategyqa --model connection --model_size micro --dry_run

# Memory check
nvidia-smi  # Check GPU memory

# Batch experiment check
./run_experiments.sh strategyqa micro  # Single dataset test

# Output verification
ls outputs/experiments/  # Check experiment files
ls outputs/analysis/     # Check analysis results
ls outputs/comparisons/  # Check comparison results
```

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

1. **Test your changes**: Run with `--dry_run` first
2. **Verify analysis**: Ensure visualizations generate correctly
3. **Check file structure**: Follow the organized output structure
4. **Document changes**: Update relevant configuration or analysis features

For questions or issues, please include:

- Command used and model configuration
- Training logs from `outputs/experiments/*/training_log.txt`
- Error messages and system information
