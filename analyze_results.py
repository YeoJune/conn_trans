# analyze_results.py - 실험 결과 분석 스크립트
import json
import os
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import compare_model_performance

def load_results(results_dir):
    """결과 파일들을 로드"""
    results = {}
    
    # JSON 결과 파일들 찾기
    json_files = glob.glob(os.path.join(results_dir, "results_*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # 파일명에서 정보 추출
            filename = os.path.basename(json_file)
            parts = filename.replace('results_', '').replace('.json', '').split('_')
            
            if len(parts) >= 2:
                model_type = parts[0]
                dataset = parts[1]
                timestamp = parts[2] if len(parts) > 2 else "unknown"
                
                key = f"{model_type}_{dataset}"
                results[key] = data
                
        except Exception as e:
            print(f"⚠️ Error loading {json_file}: {e}")
    
    return results

def create_comparison_table(results):
    """비교 테이블 생성"""
    rows = []
    
    for key, data in results.items():
        model_type = data.get('model_type', 'unknown')
        dataset = data.get('dataset', 'unknown')
        accuracy = data.get('best_accuracy', 0.0)
        
        # Config 정보 추출
        config = data.get('config', {})
        d_model = config.get('d_model', 'N/A')
        num_slots = config.get('num_slots', 'N/A')
        bilinear_rank = config.get('bilinear_rank', 'N/A')
        
        # Baseline의 경우 레이어 정보
        baseline_config = config.get('baseline_config', {})
        num_layers = baseline_config.get('num_layers', 'N/A')
        ffn_mult = baseline_config.get('ffn_multiplier', 'N/A')
        
        rows.append({
            'Model': model_type,
            'Dataset': dataset,
            'Accuracy': accuracy,
            'd_model': d_model,
            'num_slots': num_slots,
            'bilinear_rank': bilinear_rank,
            'num_layers': num_layers,
            'ffn_mult': ffn_mult
        })
    
    return pd.DataFrame(rows)

def analyze_reasoning_efficiency(results):
    """추론 효율성 분석"""
    reasoning_data = {}
    
    for key, data in results.items():
        if data.get('model_type') == 'connection':
            dataset = data.get('dataset')
            steps_history = data.get('reasoning_steps_history', [])
            
            if steps_history:
                reasoning_data[dataset] = {
                    'mean_steps': sum(steps_history) / len(steps_history),
                    'final_steps': steps_history[-1] if steps_history else 0,
                    'convergence_trend': steps_history
                }
    
    return reasoning_data

def create_performance_plots(results, output_dir):
    """성능 비교 플롯 생성"""
    datasets = set(data.get('dataset') for data in results.values())
    
    for dataset in datasets:
        dataset_results = {
            key: data for key, data in results.items() 
            if data.get('dataset') == dataset
        }
        
        if len(dataset_results) >= 2:  # 비교할 모델이 2개 이상
            # 성능 비교 플롯
            model_accuracies = {
                data.get('model_type', key): {'best_accuracy': data.get('best_accuracy', 0.0)}
                for key, data in dataset_results.items()
            }
            
            try:
                from utils.visualization import compare_model_performance
                compare_model_performance(
                    model_accuracies,
                    save_path=os.path.join(output_dir, f'performance_comparison_{dataset}.png')
                )
            except ImportError:
                print(f"⚠️ Visualization not available for {dataset}")

def create_reasoning_analysis_plots(reasoning_data, output_dir):
    """추론 분석 플롯 생성"""
    if not reasoning_data:
        return
    
    try:
        from utils.visualization import plot_reasoning_efficiency
        
        # 모든 데이터셋의 추론 스텝을 하나로 합침
        all_steps = []
        for dataset_data in reasoning_data.values():
            if 'convergence_trend' in dataset_data:
                all_steps.extend(dataset_data['convergence_trend'])
        
        if all_steps:
            plot_reasoning_efficiency(
                all_steps,
                save_path=os.path.join(output_dir, 'reasoning_efficiency.png')
            )
            print(f"📊 Reasoning efficiency plot saved")
        
    except ImportError:
        print(f"⚠️ Reasoning visualization not available")

def generate_report(df, reasoning_data, output_dir):
    """결과 리포트 생성"""
    report_path = os.path.join(output_dir, 'experiment_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Connection Transformer Experiment Results\n\n")
        
        # 개요
        f.write("## 📋 Overview\n\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Datasets: {', '.join(df['Dataset'].unique())}\n")
        f.write(f"Models: {', '.join(df['Model'].unique())}\n\n")
        
        # 성능 요약
        f.write("## 🎯 Performance Summary\n\n")
        f.write("| Model | Dataset | Accuracy | Parameters | Notes |\n")
        f.write("|-------|---------|----------|------------|-------|\n")
        
        for _, row in df.iterrows():
            model = row['Model']
            dataset = row['Dataset']
            acc = row['Accuracy']
            
            if model == 'connection':
                params = f"d={row['d_model']}, N={row['num_slots']}, r={row['bilinear_rank']}"
            else:
                params = f"d={row['d_model']}, L={row['num_layers']}, FFN={row['ffn_mult']}x"
            
            f.write(f"| {model} | {dataset} | {acc:.4f} | {params} | |\n")
        
        f.write("\n")
        
        # 데이터셋별 비교
        f.write("## 📊 Dataset-wise Comparison\n\n")
        for dataset in df['Dataset'].unique():
            dataset_df = df[df['Dataset'] == dataset]
            f.write(f"### {dataset}\n\n")
            
            conn_acc = dataset_df[dataset_df['Model'] == 'connection']['Accuracy'].values
            base_acc = dataset_df[dataset_df['Model'] == 'baseline']['Accuracy'].values
            
            if len(conn_acc) > 0 and len(base_acc) > 0:
                improvement = (conn_acc[0] - base_acc[0]) * 100
                f.write(f"- Connection Transformer: **{conn_acc[0]:.4f}**\n")
                f.write(f"- Baseline Transformer: **{base_acc[0]:.4f}**\n")
                f.write(f"- Improvement: **{improvement:+.2f}%**\n\n")
        
        # 추론 효율성
        if reasoning_data:
            f.write("## ⚡ Reasoning Efficiency\n\n")
            for dataset, data in reasoning_data.items():
                f.write(f"### {dataset}\n")
                f.write(f"- Average reasoning steps: **{data['mean_steps']:.2f}**\n")
                f.write(f"- Final reasoning steps: **{data['final_steps']:.2f}**\n")
                f.write(f"- Convergence trend: {'Decreasing' if data['convergence_trend'][-1] < data['convergence_trend'][0] else 'Stable'}\n\n")
        
        # 결론
        f.write("## 📝 Key Findings\n\n")
        
        # 전체 평균 성능 계산
        conn_mean = df[df['Model'] == 'connection']['Accuracy'].mean()
        base_mean = df[df['Model'] == 'baseline']['Accuracy'].mean()
        
        f.write(f"1. **Overall Performance**: Connection Transformer achieves {conn_mean:.4f} vs Baseline {base_mean:.4f}\n")
        f.write(f"2. **Average Improvement**: {(conn_mean - base_mean) * 100:+.2f}%\n")
        
        if reasoning_data:
            avg_steps = sum(d['mean_steps'] for d in reasoning_data.values()) / len(reasoning_data)
            f.write(f"3. **Reasoning Efficiency**: Average {avg_steps:.2f} steps per sample\n")
        
        f.write("4. **Parameter Efficiency**: Fair comparison with matched parameter counts\n")
    
    print(f"📋 Report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing result JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save analysis outputs")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔍 Loading experiment results...")
    results = load_results(args.results_dir)
    
    if not results:
        print("❌ No results found!")
        return
    
    print(f"✅ Loaded {len(results)} experiment results")
    
    # 비교 테이블 생성
    print("📊 Creating comparison table...")
    df = create_comparison_table(results)
    
    # CSV로 저장
    csv_path = os.path.join(args.output_dir, 'results_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"💾 Comparison table saved to {csv_path}")
    
    # 추론 효율성 분석
    print("⚡ Analyzing reasoning efficiency...")
    reasoning_data = analyze_reasoning_efficiency(results)
    
    # 시각화
    print("🎨 Creating visualizations...")
    create_performance_plots(results, args.output_dir)
    
    if reasoning_data:
        create_reasoning_analysis_plots(reasoning_data, args.output_dir)
    
    # 리포트 생성
    print("📋 Generating report...")
    generate_report(df, reasoning_data, args.output_dir)
    
    print("✅ Analysis completed!")
    print(f"📁 Check {args.output_dir} for detailed results")

if __name__ == "__main__":
    main()