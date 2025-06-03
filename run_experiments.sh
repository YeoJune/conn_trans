#!/bin/bash

# Connection Transformer Experiment Runner
# Usage: ./run_experiments.sh [dataset] [model_size]

set -e

# Configuration
OUTPUT_DIR="outputs"
DATASETS=("strategyqa" "logiqa" "gsm8k" "multinli" "eli5" "commongen")

# Parse arguments
DATASET=${1:-"all"}
MODEL_SIZE=${2:-"default"}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}🔍 $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_section() {
    echo ""
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}$(printf '=%.0s' {1..50})${NC}"
}

# Get appropriate model size for dataset
get_model_size() {
    local dataset=$1
    local requested_size=$2
    
    if [ "$requested_size" != "default" ]; then
        echo $requested_size
        return
    fi
    
    case $dataset in
        "strategyqa"|"logiqa") echo "micro" ;;
        "gsm8k"|"commongen") echo "small" ;;
        "multinli"|"eli5") echo "base" ;;
        *) echo "micro" ;;
    esac
}

# Validate dataset name
validate_dataset() {
    local dataset=$1
    if [ "$dataset" = "all" ]; then
        return 0
    fi
    
    for valid_dataset in "${DATASETS[@]}"; do
        if [ "$dataset" = "$valid_dataset" ]; then
            return 0
        fi
    done
    
    log_error "Unknown dataset: $dataset"
    echo "Available datasets: ${DATASETS[*]}"
    exit 1
}

# Run single experiment
run_experiment() {
    local dataset=$1
    local model=$2
    local size=$3
    
    log_info "Training: $dataset - $model ($size)"
    
    if python main.py --dataset $dataset --model $model --model_size $size --output_dir $OUTPUT_DIR; then
        log_success "Completed: $dataset - $model ($size)"
        return 0
    else
        log_error "Failed: $dataset - $model ($size)"
        return 1
    fi
}

# Run both models for a dataset
run_dataset_comparison() {
    local dataset=$1
    local size=$2
    
    log_section "📊 Dataset: $dataset (size: $size)"
    
    run_experiment $dataset "connection" $size
    run_experiment $dataset "baseline" $size
}

# System verification
verify_system() {
    log_section "🔍 System Verification"
    
    if python main.py --dataset strategyqa --model connection --model_size micro --dry_run --output_dir $OUTPUT_DIR; then
        log_success "System verification passed"
    else
        log_error "System verification failed"
        exit 1
    fi
}

# Run final analysis
run_analysis() {
    log_section "📈 Final Analysis"
    
    python analyze_results.py --output_dir $OUTPUT_DIR
}

# Show experiment summary
show_summary() {
    log_section "📋 Experiment Summary"
    
    if [ -d "$OUTPUT_DIR/experiments" ]; then
        local exp_count=$(find "$OUTPUT_DIR/experiments" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "📊 Total experiments: $exp_count"
    fi
    
    if [ -d "$OUTPUT_DIR/comparisons" ]; then
        local latest_comparison=$(ls -t "$OUTPUT_DIR/comparisons" 2>/dev/null | head -n 1)
        if [ -n "$latest_comparison" ]; then
            echo "📈 Latest comparison: $latest_comparison"
        fi
    fi
    
    echo ""
    echo "📁 Results locations:"
    echo "   Experiments: $OUTPUT_DIR/experiments/"
    echo "   Analysis: $OUTPUT_DIR/analysis/"
    echo "   Comparisons: $OUTPUT_DIR/comparisons/"
}

# Main execution
main() {
    log_section "🚀 Connection Transformer Experiments"
    echo "📊 Dataset: $DATASET"
    echo "📏 Model Size: $MODEL_SIZE"
    
    # Setup
    mkdir -p $OUTPUT_DIR
    validate_dataset $DATASET
    verify_system
    
    # Run experiments
    if [ "$DATASET" = "all" ]; then
        log_section "🎯 Running All Datasets"
        for dataset in "${DATASETS[@]}"; do
            size=$(get_model_size $dataset $MODEL_SIZE)
            run_dataset_comparison $dataset $size
        done
    else
        log_section "🎯 Running Single Dataset"
        size=$(get_model_size $DATASET $MODEL_SIZE)
        run_dataset_comparison $DATASET $size
    fi
    
    # Analysis and summary
    run_analysis
    show_summary
    
    log_success "All experiments completed!"
    echo ""
    echo "🎯 Usage examples:"
    echo "  ./run_experiments.sh                     # All datasets, default sizes"
    echo "  ./run_experiments.sh strategyqa micro    # Single dataset"
    echo "  ./run_experiments.sh all small           # All datasets, small size"
}

# Script entry point
main "$@"