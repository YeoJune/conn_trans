#!/bin/bash

# Connection Transformer Experiment Runner
# Usage: ./run_experiments.sh [dataset] [model_size]
# Examples:
#   ./run_experiments.sh                    # All datasets, default sizes
#   ./run_experiments.sh strategyqa         # Only StrategyQA, default size
#   ./run_experiments.sh multinli base      # Only MultiNLI, base size
#   ./run_experiments.sh all micro          # All datasets, micro size

set -e  # Exit on any error

# Parse arguments
DATASET=${1:-"all"}
MODEL_SIZE=${2:-"default"}

echo "🚀 Connection Transformer Experiments"
echo "====================================="
echo "📊 Dataset: $DATASET"
echo "📏 Model Size: $MODEL_SIZE"
echo ""

# Create output directory
OUTPUT_DIR="outputs"
mkdir -p $OUTPUT_DIR

# Quick verification
echo "🔍 Quick verification..."
python main.py --dataset strategyqa --model connection --model_size micro --dry_run --output_dir $OUTPUT_DIR
if [ $? -ne 0 ]; then
   echo "❌ Verification failed! Check your setup."
   exit 1
fi
echo "✅ System ready!"
echo ""

# Function to run single experiment
run_experiment() {
    local dataset=$1
    local model=$2
    local size=$3
    
    echo "🔄 Running: $dataset - $model ($size)"
    python main.py --dataset $dataset --model $model --model_size $size --output_dir $OUTPUT_DIR --skip_analysis
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed: $dataset - $model ($size)"
    else
        echo "❌ Failed: $dataset - $model ($size)"
        return 1
    fi
    echo ""
}

# Function to run comparison for a dataset
run_comparison() {
    local dataset=$1
    local size=$2
    
    echo "📊 Dataset: $dataset (size: $size)"
    echo "--------------------------------"
    
    run_experiment $dataset "connection" $size
    run_experiment $dataset "baseline" $size
}

# Determine model sizes for each dataset
get_model_size() {
    local dataset=$1
    local requested_size=$2
    
    if [ "$requested_size" != "default" ]; then
        echo $requested_size
        return
    fi
    
    # Default sizes per dataset
    case $dataset in
        "strategyqa")
            echo "micro"
            ;;
        "logiqa")
            echo "small"
            ;;
        "gsm8k")
            echo "small"
            ;;
        "multinli")
            echo "base"
            ;;
        "eli5")
            echo "base"
            ;;
        "commongen")
            echo "small"
            ;;
        *)
            echo "micro"
            ;;
    esac
}

# Main experiment logic
if [ "$DATASET" = "all" ]; then
    echo "🎯 Running all datasets"
    echo ""
    
    for dataset in strategyqa logiqa gsm8k multinli eli5 commongen; do
        size=$(get_model_size $dataset $MODEL_SIZE)
        run_comparison $dataset $size
    done
else
    echo "🎯 Running single dataset: $DATASET"
    echo ""
    
    # Validate dataset
    case $DATASET in
        "strategyqa"|"logiqa"|"gsm8k"|"multinli"|"eli5"|"commongen")
            size=$(get_model_size $DATASET $MODEL_SIZE)
            run_comparison $DATASET $size
            ;;
        *)
            echo "❌ Unknown dataset: $DATASET"
            echo "Available: strategyqa, logiqa, gsm8k, multinli, eli5, commongen"
            exit 1
            ;;
    esac
fi

# Final analysis
echo "📈 Running final analysis..."
echo "============================"

# The main.py already runs analysis automatically on the last experiment
# But we can also run a comprehensive comparison
python -c "
from utils.comparison_analyzer import ComparisonAnalyzer
import sys

try:
    analyzer = ComparisonAnalyzer('$OUTPUT_DIR')
    success = analyzer.analyze_all_experiments()
    if success:
        summary = analyzer.get_comparison_summary()
        print(f'')
        print(f'🎉 Analysis Complete!')
        print(f'📊 Total experiments: {summary[\"total_experiments\"]}')
        print(f'🏆 Best accuracy: {summary[\"best_accuracy\"]:.4f}')
        print(f'📁 Results: {summary[\"comparison_dir\"]}')
    else:
        print('⚠️ Analysis skipped (insufficient data)')
except Exception as e:
    print(f'⚠️ Analysis error: {str(e)[:50]}...')
    sys.exit(0)  # Don't fail the script
"

echo ""
echo "✅ All experiments completed!"
echo ""
echo "📁 Experiment data: $OUTPUT_DIR/experiments/"
echo "📊 Analysis results: $OUTPUT_DIR/analysis/"
echo "📈 Comparisons: $OUTPUT_DIR/comparisons/"
echo ""

# Quick summary
if [ -d "$OUTPUT_DIR/comparisons" ]; then
    latest_comparison=$(ls -t $OUTPUT_DIR/comparisons/ | head -n 1)
    if [ -n "$latest_comparison" ] && [ -f "$OUTPUT_DIR/comparisons/$latest_comparison/summary_report.md" ]; then
        echo "📋 Quick Summary:"
        echo "================"
        head -n 20 "$OUTPUT_DIR/comparisons/$latest_comparison/summary_report.md"
        echo ""
        echo "📖 Full report: $OUTPUT_DIR/comparisons/$latest_comparison/summary_report.md"
    fi
fi

echo ""
echo "🎯 Usage examples for next runs:"
echo "  ./run_experiments.sh strategyqa micro    # Single dataset"
echo "  ./run_experiments.sh all small           # All datasets, small size"
echo "  ./run_experiments.sh                     # All datasets, default sizes"