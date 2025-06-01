#!/bin/bash

# Connection Transformer Experiment Runner
echo "🚀 Starting Connection Transformer Experiments"
echo "=============================================="

# Create output directory
mkdir -p experiments_output

# Verification
echo "🔍 Running system verification..."
python final_verification.py
if [ $? -ne 0 ]; then
   echo "❌ Verification failed! Please fix issues before running experiments."
   exit 1
fi

echo "✅ Verification passed!"
echo ""

# Small dataset experiments (fast)
echo "📊 Small Dataset Experiments"
echo "----------------------------"

echo "1. StrategyQA - Connection (nano)"
python main.py --dataset strategyqa --model connection --model_size nano --output_dir experiments_output

echo "2. StrategyQA - Baseline (nano)"  
python main.py --dataset strategyqa --model baseline --model_size nano --output_dir experiments_output

echo "3. LogiQA - Connection (micro)"
python main.py --dataset logiqa --model connection --model_size micro --output_dir experiments_output

echo "4. LogiQA - Baseline (micro)"
python main.py --dataset logiqa --model baseline --model_size micro --output_dir experiments_output

# Medium dataset experiments
echo ""
echo "📊 Medium Dataset Experiments"
echo "-----------------------------"

echo "5. GSM8K - Connection (micro)"
python main.py --dataset gsm8k --model connection --model_size micro --output_dir experiments_output

echo "6. GSM8K - Baseline (micro)"
python main.py --dataset gsm8k --model baseline --model_size micro --output_dir experiments_output

# Large dataset experiments (if enough memory)
echo ""
echo "📊 Large Dataset Experiments"
echo "----------------------------"

echo "7. MultiNLI - Connection (base)"
python main.py --dataset multinli --model connection --model_size base --output_dir experiments_output

echo "8. MultiNLI - Baseline (base)"
python main.py --dataset multinli --model baseline --model_size base --output_dir experiments_output

# Results analysis
echo ""
echo "📈 Analyzing Results"
echo "-------------------"

python analyze_results.py --results_dir experiments_output --output_dir analysis_output

echo ""
echo "✅ All experiments completed!"
echo "📁 Results saved in: experiments_output/"
echo "📊 Analysis saved in: analysis_output/"
echo ""
echo "🎉 Check analysis_output/experiment_report.md for detailed results!"

# Optional: Display quick summary
echo ""
echo "📋 Quick Summary:"
echo "----------------"
if [ -f "analysis_output/results_comparison.csv" ]; then
   echo "Results found:"
   head -n 10 analysis_output/results_comparison.csv
else
   echo "No analysis results found. Check for errors above."
fi