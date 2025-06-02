# utils/comparison_analyzer.py
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ComparisonAnalyzer:
    """실험 간 교차 비교 분석기"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.comparison_dir = self.base_dir / "comparisons" / f"{time.strftime('%Y%m%d_%H%M')}_comparison"
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 비교 분석 디렉토리: {self.comparison_dir}")
    
    def analyze_all_experiments(self) -> bool:
        """모든 실험 결과를 분석하고 비교"""
        # 1. 실험 결과 수집
        experiments = self._collect_experiments()
        if len(experiments) < 2:
            print(f"⚠️ 비교할 실험이 부족합니다 (현재: {len(experiments)}개)")
            return False
        
        print(f"🔍 {len(experiments)}개 실험 발견")
        
        # 2. 비교 테이블 생성
        df = self._create_comparison_table(experiments)
        
        # 3. 결과 저장
        self._save_comparison_results(df, experiments)
        
        # 4. 시각화 생성
        self._create_comparison_charts(df)
        
        # 5. 요약 리포트 생성
        self._generate_summary_report(df, experiments)
        
        print(f"✅ 비교 분석 완료: {self.comparison_dir}")
        return True
    
    def _collect_experiments(self) -> List[Dict[str, Any]]:
        """experiments 디렉토리에서 모든 실험 수집"""
        experiments = []
        exp_dir = self.base_dir / "experiments"
        
        if not exp_dir.exists():
            return experiments
        
        for exp_folder in exp_dir.iterdir():
            if not exp_folder.is_dir():
                continue
            
            try:
                # config.json 읽기
                config_path = exp_folder / "config.json"
                summary_path = self.base_dir / "analysis" / exp_folder.name / "summary.json"
                
                if config_path.exists() and summary_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                    
                    experiments.append({
                        'folder': exp_folder.name,
                        'config': config,
                        'summary': summary
                    })
                    
            except Exception as e:
                print(f"⚠️ {exp_folder.name} 읽기 실패: {str(e)[:30]}...")
                continue
        
        return experiments
    
    def _create_comparison_table(self, experiments: List[Dict]) -> pd.DataFrame:
        """비교 테이블 생성"""
        rows = []
        
        for exp in experiments:
            config = exp['config']
            summary = exp['summary']
            
            row = {
                'Experiment_ID': exp['folder'],
                'Model': config.get('model_type', 'unknown').title(),
                'Dataset': config.get('dataset', 'unknown').upper(),
                'Size': config.get('model_size', 'unknown'),
                'Accuracy': summary.get('best_accuracy', 0.0),
                'd_model': config.get('d_model', 'N/A'),
                'Batch_Size': config.get('batch_size', 'N/A'),
                'Learning_Rate': config.get('learning_rate', 'N/A'),
                'Epochs': summary.get('final_metrics', {}).get('num_epochs', 'N/A'),
                'Final_Train_Loss': summary.get('final_metrics', {}).get('train_loss', 'N/A'),
                'Timestamp': config.get('timestamp', 'unknown')
            }
            
            # Connection Transformer 전용 정보
            if config.get('model_type') == 'connection':
                row['Slots'] = config.get('num_slots', 'N/A')
                row['Bilinear_Rank'] = config.get('bilinear_rank', 'N/A')
                row['Max_Steps'] = config.get('max_reasoning_steps', 'N/A')
            else:
                row['Slots'] = 'N/A'
                row['Bilinear_Rank'] = 'N/A'
                row['Max_Steps'] = 'N/A'
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _save_comparison_results(self, df: pd.DataFrame, experiments: List[Dict]):
        """비교 결과 저장"""
        # CSV 저장
        csv_path = self.comparison_dir / "comparison_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 비교 테이블: {csv_path.name}")
        
        # 상세 데이터 JSON 저장
        detailed_path = self.comparison_dir / "detailed_results.json"
        with open(detailed_path, 'w') as f:
            json.dump(experiments, f, indent=2, default=str)
    
    def _create_comparison_charts(self, df: pd.DataFrame):
        """비교 차트 생성"""
        if len(df) == 0:
            return
        
        # 1. 데이터셋별 성능 비교
        self._plot_dataset_performance(df)
        
        # 2. 모델 타입별 비교
        self._plot_model_comparison(df)
        
        # 3. 모델 크기별 성능
        self._plot_size_analysis(df)
    
    def _plot_dataset_performance(self, df: pd.DataFrame):
        """데이터셋별 성능 비교 차트"""
        datasets = df['Dataset'].unique()
        models = df['Model'].unique()
        
        if len(datasets) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(datasets))
        width = 0.35
        
        for i, model in enumerate(models):
            model_data = []
            for dataset in datasets:
                subset = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
                if len(subset) > 0:
                    # 같은 데이터셋에서 최고 성능 선택
                    model_data.append(subset['Accuracy'].max())
                else:
                    model_data.append(0)
            
            color = '#3498db' if model == 'Connection' else '#e74c3c'
            bars = ax.bar([xi + i * width for xi in x], model_data, 
                         width, label=model, color=color, alpha=0.8)
            
            # 값 표시
            for bar, value in zip(bars, model_data):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Accuracy')
        ax.set_title('Performance Comparison by Dataset')
        ax.set_xticks([xi + width/2 for xi in x])
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.comparison_dir / 'dataset_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 데이터셋 성능 차트 저장")
    
    def _plot_model_comparison(self, df: pd.DataFrame):
        """모델 타입별 전체 비교"""
        if len(df['Model'].unique()) < 2:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 평균 성능 비교
        model_avg = df.groupby('Model')['Accuracy'].agg(['mean', 'std', 'count']).reset_index()
        
        bars = ax1.bar(model_avg['Model'], model_avg['mean'], 
                      yerr=model_avg['std'], capsize=5,
                      color=['#3498db', '#e74c3c'], alpha=0.8)
        
        # 값과 개수 표시
        for bar, mean, count in zip(bars, model_avg['mean'], model_avg['count']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}\n(n={count})', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_ylabel('Average Accuracy')
        ax1.set_title('Average Performance by Model Type')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 성능 분포 박스플롯
        models = df['Model'].unique()
        data = [df[df['Model'] == model]['Accuracy'].values for model in models]
        
        box_plot = ax2.boxplot(data, labels=models, patch_artist=True)
        colors = ['#3498db', '#e74c3c']
        for patch, color in zip(box_plot['boxes'], colors[:len(models)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Distribution by Model Type')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.comparison_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 모델 비교 차트 저장")
    
    def _plot_size_analysis(self, df: pd.DataFrame):
        """모델 크기별 성능 분석"""
        if 'd_model' not in df.columns or df['d_model'].nunique() < 2:
            return
        
        # 숫자 변환 가능한 d_model만 필터링
        numeric_df = df[df['d_model'] != 'N/A'].copy()
        try:
            numeric_df['d_model_num'] = pd.to_numeric(numeric_df['d_model'])
        except:
            return
        
        if len(numeric_df) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model in numeric_df['Model'].unique():
            model_data = numeric_df[numeric_df['Model'] == model]
            color = '#3498db' if model == 'Connection' else '#e74c3c'
            
            ax.scatter(model_data['d_model_num'], model_data['Accuracy'], 
                      label=model, alpha=0.7, s=80, color=color)
        
        ax.set_xlabel('Model Dimension (d_model)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Model Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.comparison_dir / 'size_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📏 크기 분석 차트 저장")
    
    def _generate_summary_report(self, df: pd.DataFrame, experiments: List[Dict]):
        """종합 요약 리포트 생성"""
        report_path = self.comparison_dir / "summary_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# 실험 비교 분석 리포트\n\n")
            f.write(f"생성 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 전체 요약
            f.write("## 📊 전체 요약\n\n")
            f.write(f"- **총 실험 수**: {len(experiments)}\n")
            f.write(f"- **모델 타입**: {', '.join(df['Model'].unique())}\n")
            f.write(f"- **데이터셋**: {', '.join(df['Dataset'].unique())}\n")
            f.write(f"- **최고 정확도**: {df['Accuracy'].max():.4f}\n")
            f.write(f"- **평균 정확도**: {df['Accuracy'].mean():.4f}\n\n")
            
            # 데이터셋별 최고 성능
            f.write("## 🏆 데이터셋별 최고 성능\n\n")
            for dataset in df['Dataset'].unique():
                dataset_df = df[df['Dataset'] == dataset]
                if len(dataset_df) > 0:
                    best_row = dataset_df.loc[dataset_df['Accuracy'].idxmax()]
                    f.write(f"### {dataset}\n")
                    f.write(f"- **최고 성능**: {best_row['Accuracy']:.4f}\n")
                    f.write(f"- **모델**: {best_row['Model']}\n")
                    f.write(f"- **실험 ID**: `{best_row['Experiment_ID']}`\n\n")
            
            # Connection vs Baseline 비교
            f.write("## ⚔️ Connection vs Baseline 비교\n\n")
            
            total_comparisons = 0
            connection_wins = 0
            improvements = []
            
            for dataset in df['Dataset'].unique():
                dataset_df = df[df['Dataset'] == dataset]
                conn_rows = dataset_df[dataset_df['Model'] == 'Connection']
                base_rows = dataset_df[dataset_df['Model'] == 'Baseline']
                
                if len(conn_rows) > 0 and len(base_rows) > 0:
                    conn_best = conn_rows['Accuracy'].max()
                    base_best = base_rows['Accuracy'].max()
                    improvement = (conn_best - base_best) * 100
                    
                    f.write(f"### {dataset}\n")
                    f.write(f"- **Connection**: {conn_best:.4f}\n")
                    f.write(f"- **Baseline**: {base_best:.4f}\n")
                    f.write(f"- **개선**: {improvement:+.2f}%p\n")
                    
                    total_comparisons += 1
                    improvements.append(improvement)
                    
                    if conn_best > base_best:
                        connection_wins += 1
                        f.write(f"- **결과**: ✅ Connection 승리\n\n")
                    else:
                        f.write(f"- **결과**: ❌ Baseline 승리\n\n")
            
            # 전체 결론
            f.write("## 🎯 종합 결론\n\n")
            
            if total_comparisons > 0:
                win_rate = connection_wins / total_comparisons * 100
                avg_improvement = sum(improvements) / len(improvements)
                
                f.write(f"- **직접 비교 가능한 데이터셋**: {total_comparisons}개\n")
                f.write(f"- **Connection Transformer 승률**: {connection_wins}/{total_comparisons} ({win_rate:.1f}%)\n")
                f.write(f"- **평균 성능 개선**: {avg_improvement:+.2f}%p\n\n")
                
                if win_rate >= 70:
                    f.write("**🎉 Connection Transformer가 대부분의 태스크에서 우수한 성능을 보입니다.**\n")
                elif win_rate >= 50:
                    f.write("**✅ Connection Transformer가 전반적으로 좋은 성능을 보입니다.**\n")
                else:
                    f.write("**⚠️ 혼재된 결과로 추가 분석이 필요합니다.**\n")
            else:
                f.write("- **직접 비교 불가**: 같은 데이터셋에서 두 모델 결과가 없습니다.\n")
            
            # 권장사항
            f.write("\n## 💡 권장사항\n\n")
            f.write("1. **성능이 좋은 설정**을 다른 데이터셋에 적용해보세요\n")
            f.write("2. **부족한 데이터셋**에서 추가 실험을 진행하세요\n")
            f.write("3. **하이퍼파라미터 튜닝**으로 성능을 더 개선해보세요\n")
            
            # 파일 위치 안내
            f.write(f"\n## 📁 상세 결과\n\n")
            f.write(f"- **비교 테이블**: `{self.comparison_dir.name}/comparison_table.csv`\n")
            f.write(f"- **차트들**: `{self.comparison_dir.name}/`\n")
            f.write(f"- **개별 실험 결과**: `analysis/` 디렉토리\n")
        
        print(f"📋 요약 리포트 저장: {report_path.name}")
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """프로그래밍 방식으로 사용할 수 있는 요약 반환"""
        experiments = self._collect_experiments()
        if len(experiments) < 2:
            return {'status': 'insufficient_data', 'count': len(experiments)}
        
        df = self._create_comparison_table(experiments)
        
        return {
            'status': 'success',
            'total_experiments': len(experiments),
            'datasets': list(df['Dataset'].unique()),
            'models': list(df['Model'].unique()),
            'best_accuracy': float(df['Accuracy'].max()),
            'average_accuracy': float(df['Accuracy'].mean()),
            'comparison_dir': str(self.comparison_dir)
        }