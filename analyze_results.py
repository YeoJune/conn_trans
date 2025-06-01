# analyze_results.py
import json
import os
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class ResultAnalyzer:
    """Improved results analyzer with better error handling"""
    
    def __init__(self, results_dir, output_dir):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_results(self):
        """Load all result JSON files"""
        results = {}
        pattern = self.results_dir / "results_*.json"
        
        for json_file in glob.glob(str(pattern)):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract model and dataset from filename or data
                filename = Path(json_file).stem
                
                model_type = data.get('model_type', 'unknown')
                dataset = data.get('dataset', 'unknown')
                
                key = f"{model_type}_{dataset}"
                results[key] = data
                
                print(f"✅ Loaded: {key}")
                
            except Exception as e:
                print(f"⚠️ Error loading {json_file}: {e}")
        
        print(f"📊 Total results loaded: {len(results)}")
        return results
    
    def create_comparison_table(self, results):
        """Create structured comparison table"""
        rows = []
        
        for key, data in results.items():
            try:
                # Basic info
                model_type = data.get('model_type', 'unknown')
                dataset = data.get('dataset', 'unknown')
                accuracy = data.get('best_accuracy', 0.0)
                
                # Config info (unified access)
                config = data.get('config', {})
                d_model = config.get('d_model', 'N/A')
                batch_size = config.get('batch_size', 'N/A')
                learning_rate = config.get('learning_rate', 'N/A')
                
                # Architecture-specific info
                if model_type == 'connection':
                    num_slots = config.get('num_slots', 'N/A')
                    bilinear_rank = config.get('bilinear_rank', 'N/A')
                    architecture = f"slots={num_slots}, rank={bilinear_rank}"
                else:
                    # For baseline, use consistent field names
                    num_encoder_layers = config.get('num_encoder_layers', 'N/A')
                    num_decoder_layers = config.get('num_decoder_layers', 'N/A')
                    architecture = f"enc={num_encoder_layers}, dec={num_decoder_layers}"
                
                # Training metrics
                metrics = data.get('metrics', {})
                final_train_loss = 'N/A'
                if 'train_losses' in metrics and metrics['train_losses']:
                    final_train_loss = f"{metrics['train_losses'][-1]:.4f}"
                
                rows.append({
                    'Model': model_type.title(),
                    'Dataset': dataset.upper(),
                    'Accuracy': accuracy,
                    'd_model': d_model,
                    'Architecture': architecture,
                    'Batch_Size': batch_size,
                    'Learning_Rate': learning_rate,
                    'Final_Train_Loss': final_train_loss
                })
                
            except Exception as e:
                print(f"⚠️ Error processing {key}: {e}")
                continue
        
        return pd.DataFrame(rows)
    
    def generate_report(self, df):
        """Generate comprehensive markdown report"""
        report_path = self.output_dir / 'experiment_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Connection Transformer Experiment Results\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write("## 📊 Complete Results\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            
            # Dataset-wise analysis
            f.write("## 🎯 Dataset-wise Performance\n\n")
            
            for dataset in df['Dataset'].unique():
                dataset_df = df[df['Dataset'] == dataset].copy()
                f.write(f"### {dataset}\n\n")
                
                # Find Connection and Baseline results
                conn_rows = dataset_df[dataset_df['Model'] == 'Connection']
                base_rows = dataset_df[dataset_df['Model'] == 'Baseline']
                
                if len(conn_rows) > 0 and len(base_rows) > 0:
                    conn_acc = conn_rows['Accuracy'].iloc[0]
                    base_acc = base_rows['Accuracy'].iloc[0]
                    improvement = (conn_acc - base_acc) * 100
                    
                    f.write(f"- **Connection Transformer**: {conn_acc:.4f}\n")
                    f.write(f"- **Baseline Transformer**: {base_acc:.4f}\n")
                    f.write(f"- **Improvement**: {improvement:+.2f} percentage points\n")
                    
                    if improvement > 0:
                        f.write(f"- **Result**: ✅ Connection Transformer outperforms baseline\n\n")
                    else:
                        f.write(f"- **Result**: ⚠️ Baseline performs better\n\n")
                
                elif len(conn_rows) > 0:
                    conn_acc = conn_rows['Accuracy'].iloc[0]
                    f.write(f"- **Connection Transformer**: {conn_acc:.4f}\n")
                    f.write(f"- **Note**: No baseline comparison available\n\n")
                
                elif len(base_rows) > 0:
                    base_acc = base_rows['Accuracy'].iloc[0]
                    f.write(f"- **Baseline Transformer**: {base_acc:.4f}\n")
                    f.write(f"- **Note**: No connection transformer results available\n\n")
            
            # Overall summary
            f.write("## 🏆 Overall Summary\n\n")
            
            total_comparisons = 0
            connection_wins = 0
            
            for dataset in df['Dataset'].unique():
                dataset_df = df[df['Dataset'] == dataset]
                conn_rows = dataset_df[dataset_df['Model'] == 'Connection']
                base_rows = dataset_df[dataset_df['Model'] == 'Baseline']
                
                if len(conn_rows) > 0 and len(base_rows) > 0:
                    total_comparisons += 1
                    if conn_rows['Accuracy'].iloc[0] > base_rows['Accuracy'].iloc[0]:
                        connection_wins += 1
            
            if total_comparisons > 0:
                win_rate = connection_wins / total_comparisons * 100
                f.write(f"- **Datasets compared**: {total_comparisons}\n")
                f.write(f"- **Connection Transformer wins**: {connection_wins}/{total_comparisons} ({win_rate:.1f}%)\n")
                
                if win_rate >= 50:
                    f.write(f"- **Conclusion**: ✅ Connection Transformer shows promising results\n")
                else:
                    f.write(f"- **Conclusion**: ⚠️ Mixed results, further investigation needed\n")
            else:
                f.write("- **Note**: No direct comparisons available\n")
        
        print(f"📋 Report saved: {report_path}")
        return report_path
    
    def create_visualizations(self, df):
        """Create comprehensive visualizations"""
        # Performance comparison plot
        self._create_performance_plot(df)
        
        # Architecture comparison (if available)
        self._create_architecture_plot(df)
        
        print("📈 All visualizations created")
    
    def _create_performance_plot(self, df):
        """Create performance comparison plot"""
        if len(df) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        datasets = sorted(df['Dataset'].unique())
        models = sorted(df['Model'].unique())
        
        x = range(len(datasets))
        width = 0.35
        
        for i, model in enumerate(models):
            model_data = []
            for dataset in datasets:
                subset = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
                if len(subset) > 0:
                    model_data.append(subset['Accuracy'].iloc[0])
                else:
                    model_data.append(0)
            
            color = 'skyblue' if model == 'Connection' else 'lightcoral'
            bars = ax.bar([xi + i * width for xi in x], model_data, 
                         width, label=model, color=color, alpha=0.8)
            
            # Add value labels
            for bar, value in zip(bars, model_data):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance Comparison by Dataset')
        ax.set_xticks([xi + width/2 for xi in x])
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_architecture_plot(self, df):
        """Create architecture comparison plot"""
        if 'd_model' not in df.columns:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Model size distribution
        d_models = [x for x in df['d_model'] if x != 'N/A']
        if d_models:
            ax1.hist(d_models, bins=10, alpha=0.7, color='lightgreen')
            ax1.set_xlabel('d_model')
            ax1.set_ylabel('Count')
            ax1.set_title('Model Size Distribution')
            ax1.grid(True, alpha=0.3)
        
        # Accuracy vs model size
        valid_df = df[df['d_model'] != 'N/A'].copy()
        if len(valid_df) > 0:
            for model in valid_df['Model'].unique():
                model_subset = valid_df[valid_df['Model'] == model]
                color = 'blue' if model == 'Connection' else 'red'
                ax2.scatter(model_subset['d_model'], model_subset['Accuracy'], 
                           label=model, alpha=0.7, s=50, color=color)
            
            ax2.set_xlabel('d_model')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Accuracy vs Model Size')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'architecture_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze(self):
        """Run complete analysis"""
        print("🔍 Starting results analysis...")
        
        # Load results
        results = self.load_results()
        if not results:
            print("❌ No results found!")
            return False
        
        # Create comparison table
        df = self.create_comparison_table(results)
        if len(df) == 0:
            print("❌ No valid data to analyze!")
            return False
        
        # Save CSV
        csv_path = self.output_dir / 'results_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"💾 Table saved: {csv_path}")
        
        # Generate report
        self.generate_report(df)
        
        # Create visualizations
        self.create_visualizations(df)
        
        print(f"✅ Analysis complete! Check {self.output_dir}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Analyze Connection Transformer experiment results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing result JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save analysis outputs")
    
    args = parser.parse_args()
    
    analyzer = ResultAnalyzer(args.results_dir, args.output_dir)
    success = analyzer.analyze()
    
    if success:
        print("\n🎉 Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())