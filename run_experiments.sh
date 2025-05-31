#!/bin/bash
# run_experiments.sh - 실험 실행 스크립트

# 실험 디렉토리 생성
mkdir -p experiments/results
mkdir -p experiments/checkpoints
mkdir -p experiments/logs

echo "🚀 Connection Transformer Experiments Starting..."
echo "=================================================="

# Phase 1: 기본 비교 실험 (LogiQA)
echo "📋 Phase 1: Basic Comparison on LogiQA"
echo "--------------------------------------------------"

# Connection Transformer (base)
echo "🔹 Training Connection Transformer (base) on LogiQA..."
python main.py --dataset logiqa --model connection --model_size base --output_dir experiments/results 2>&1 | tee experiments/logs/conn_logiqa_base.log

# Baseline Transformer (matched parameters)
echo "🔶 Training Baseline Transformer (matched) on LogiQA..."
python main.py --dataset logiqa --model baseline --model_size base --output_dir experiments/results 2>&1 | tee experiments/logs/baseline_logiqa_base.log

# Connection Transformer (large)
echo "🔹 Training Connection Transformer (large) on LogiQA..."
python main.py --dataset logiqa --model connection --model_size large --output_dir experiments/results 2>&1 | tee experiments/logs/conn_logiqa_large.log

# Baseline Transformer (large, matched)
echo "🔶 Training Baseline Transformer (large) on LogiQA..."
python main.py --dataset logiqa --model baseline --model_size large --output_dir experiments/results 2>&1 | tee experiments/logs/baseline_logiqa_large.log

echo "✅ Phase 1 completed!"
echo ""

# Phase 2: 다른 데이터셋에서 검증
echo "📋 Phase 2: Cross-dataset Validation"
echo "--------------------------------------------------"

# GSM8K
echo "🔹 Training Connection Transformer on GSM8K..."
python main.py --dataset gsm8k --model connection --model_size base --output_dir experiments/results 2>&1 | tee experiments/logs/conn_gsm8k_base.log

echo "🔶 Training Baseline Transformer on GSM8K..."
python main.py --dataset gsm8k --model baseline --model_size base --output_dir experiments/results 2>&1 | tee experiments/logs/baseline_gsm8k_base.log

# StrategyQA
echo "🔹 Training Connection Transformer on StrategyQA..."
python main.py --dataset strategyqa --model connection --model_size base --output_dir experiments/results 2>&1 | tee experiments/logs/conn_strategyqa_base.log

echo "🔶 Training Baseline Transformer on StrategyQA..."
python main.py --dataset strategyqa --model baseline --model_size base --output_dir experiments/results 2>&1 | tee experiments/logs/baseline_strategyqa_base.log

echo "✅ Phase 2 completed!"
echo ""

# Phase 3: 결과 분석
echo "📋 Phase 3: Results Analysis"
echo "--------------------------------------------------"

# 결과 분석 스크립트 실행
python analyze_results.py --results_dir experiments/results --output_dir experiments/analysis

echo "✅ All experiments completed!"
echo "📊 Check experiments/analysis/ for detailed results"

# 간단한 요약 출력
echo ""
echo "📋 Quick Summary:"
echo "=================="
find experiments/logs -name "*.log" -exec echo "📄 {}" \; -exec grep -H "Best accuracy" {} \; 2>/dev/null | head -20