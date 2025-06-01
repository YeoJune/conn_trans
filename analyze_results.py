# analyze_results.py
import json
import os
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt

def load_results(results_dir):
    """결과 파일들 로드"""
    results = {}
    json_files = glob.glob(os.path.join(results_dir, "results_*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            filename = os.path.basename(json_file)
            parts = filename.replace('results_', '').replace('.json', '').split('_')
            
            if len(parts) >= 2:
                model_type = parts[0]
                dataset = parts[1]
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
        
        config = data.get('config', {})
        d_model = config.get('d_model', 'N/A')
        
        if model_type == 'connection':
            architecture = f"slots={config.get('num_slots', 'N/A')}, rank={config.get('bilinear_rank', 'N/A')}"
        else:
            baseline_config = config.get('baseline_config', {})
            architecture = f"layers={baseline_config.get('num_layers', 'N/A')}, ffn={baseline_config.get('ffn_multiplier', 'N/A')}x"
        
        rows.append({
            'Model': model_type,
            'Dataset': dataset,
            'Accuracy': accuracy,
            'd_model': d_model,
            'Architecture': architecture
        })
    
    return pd.DataFrame(rows)

def generate_report(df, output_dir):
    """간단한 결과 리포트 생성"""
    report_path = os.path.join(output_dir, 'experiment_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Connection Transformer Results\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write("| Model | Dataset | Accuracy | d_model | Architecture |\n")
        f.write("|-------|---------|----------|---------|-------------|\n")
        
        for _, row in df.iterrows():
            f.write(f"| {row['Model']} | {row['Dataset']} | {row['Accuracy']:.4f} | {row['d_model']} | {row['Architecture']} |\n")
        
        f.write("\n## Dataset-wise Comparison\n\n")
        for dataset in df['Dataset'].unique():
            dataset_df = df[df['Dataset'] == dataset]
            f.write(f"### {dataset}\n\n")
            
            conn_acc = dataset_df[dataset_df['Model'] == 'connection']['Accuracy'].values
            base_acc = dataset_df[dataset_df['Model'] == 'baseline']['Accuracy'].values
            
            if len(conn_acc) > 0 and len(base_acc) > 0:
                improvement = (conn_acc[0] - base_acc[0]) * 100
                f.write(f"- Connection: **{conn_acc[0]:.4f}**\n")
                f.write(f"- Baseline: **{base_acc[0]:.4f}**\n")
                f.write(f"- Improvement: **{improvement:+.2f}%**\n\n")
            elif len(conn_acc) > 0:
                f.write(f"- Connection: **{conn_acc[0]:.4f}**\n\n")
            elif len(base_acc) > 0:
                f.write(f"- Baseline: **{base_acc[0]:.4f}**\n\n")
    
    print(f"📋 Report saved to {report_path}")

def create_performance_plot(df, output_dir):
    """성능 비교 플롯"""
    datasets = df['Dataset'].unique()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = []
    labels = []
    colors = []
    accuracies = []
    
    for i, dataset in enumerate(datasets):
        dataset_df = df[df['Dataset'] == dataset]
        
        for j, (_, row) in enumerate(dataset_df.iterrows()):
            x_pos.append(i * 2 + j * 0.8)
            labels.append(f"{row['Model']}\n{dataset}")
            colors.append('skyblue' if row['Model'] == 'connection' else 'lightcoral')
            accuracies.append(row['Accuracy'])
    
    bars = ax.bar(x_pos, accuracies, color=colors, alpha=0.8)
    
    # 값 표시
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks([i * 2 + 0.4 for i in range(len(datasets))])
    ax.set_xticklabels(datasets)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 범례
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='skyblue', label='Connection'),
                      Patch(facecolor='lightcoral', label='Baseline')]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"📊 Performance plot saved")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔍 Loading results...")
    results = load_results(args.results_dir)
    
    if not results:
        print("❌ No results found!")
        return
    
    print(f"✅ Loaded {len(results)} results")
    
    # 비교 테이블
    df = create_comparison_table(results)
    csv_path = os.path.join(args.output_dir, 'results_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"💾 Table saved to {csv_path}")
    
    # 리포트 생성
    generate_report(df, args.output_dir)
    
    # 성능 플롯
    if len(df) > 1:
        create_performance_plot(df, args.output_dir)
    
    print(f"✅ Analysis completed! Check {args.output_dir}")

if __name__ == "__main__":
    main()