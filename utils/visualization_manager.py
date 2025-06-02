# utils/visualization_manager.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Optional

# 스타일 설정
plt.style.use('default')
sns.set_palette("husl")

class VisualizationManager:
    """간결하고 효과적인 시각화 관리자"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_final_training_curves(self, train_losses: List[float], 
                                 eval_accuracies: List[float], 
                                 reasoning_steps: Optional[List[float]] = None):
        """최종 학습 곡선 - 깔끔하고 정보량 많게"""
        n_plots = 3 if reasoning_steps else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        
        if n_plots == 2:
            axes = [axes[0], axes[1]]
        
        epochs = range(1, len(train_losses) + 1)
        
        # 1. 손실 곡선
        axes[0].plot(epochs, train_losses, 'b-', linewidth=2.5, alpha=0.8, label='Train Loss')
        axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 2. 정확도 곡선
        acc_epochs = range(1, len(eval_accuracies) + 1)
        axes[1].plot(acc_epochs, eval_accuracies, 'g-', linewidth=2.5, alpha=0.8, label='Eval Accuracy')
        axes[1].axhline(y=max(eval_accuracies), color='r', linestyle='--', alpha=0.7, 
                       label=f'Best: {max(eval_accuracies):.3f}')
        axes[1].set_title('Evaluation Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # 3. 추론 단계 (Connection Transformer만)
        if reasoning_steps:
            # 에포크별 평균 계산
            steps_per_epoch = len(reasoning_steps) // max(len(train_losses), 1)
            epoch_avg_steps = []
            for i in range(len(train_losses)):
                start_idx = i * steps_per_epoch
                end_idx = min((i + 1) * steps_per_epoch, len(reasoning_steps))
                if start_idx < len(reasoning_steps):
                    epoch_avg = sum(reasoning_steps[start_idx:end_idx]) / max(end_idx - start_idx, 1)
                    epoch_avg_steps.append(epoch_avg)
            
            if epoch_avg_steps:
                axes[2].plot(range(1, len(epoch_avg_steps) + 1), epoch_avg_steps, 
                           'orange', linewidth=2.5, alpha=0.8, label='Avg Steps')
                axes[2].set_title('Reasoning Steps', fontsize=12, fontweight='bold')
                axes[2].set_xlabel('Epoch')
                axes[2].set_ylabel('Steps')
                axes[2].grid(True, alpha=0.3)
                axes[2].legend()
        
        plt.tight_layout()
        save_path = self.output_dir / 'final_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 학습 곡선 저장: {save_path.name}")
    
    def plot_connection_matrix(self, model):
        """연결 행렬 시각화 - Connection Transformer 전용"""
        if not hasattr(model, 'get_connection_analysis'):
            return
        
        try:
            analysis = model.get_connection_analysis()
            matrix = analysis.get('connection_matrix')
            
            if matrix is None:
                return
            
            matrix_np = matrix.cpu().numpy() if hasattr(matrix, 'cpu') else matrix
            
            plt.figure(figsize=(8, 6))
            
            # 히트맵 생성
            sns.heatmap(matrix_np, 
                       cmap='RdBu_r', 
                       center=0, 
                       square=True,
                       cbar_kws={'label': 'Connection Strength'},
                       fmt='.3f')
            
            plt.title('Connection Matrix\n(Red: Positive, Blue: Negative)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Target Slot')
            plt.ylabel('Source Slot')
            
            # 통계 정보 추가
            sparsity = analysis.get('sparsity_ratio', 0)
            max_conn = analysis.get('max_connection', 0)
            plt.figtext(0.02, 0.02, f'Sparsity: {sparsity:.3f} | Max: {max_conn:.3f}', 
                       fontsize=10, style='italic')
            
            save_path = self.output_dir / 'connection_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"🔗 연결 행렬 저장: {save_path.name}")
            
        except Exception as e:
            print(f"⚠️ 연결 행렬 시각화 실패: {str(e)[:50]}...")
    
    def plot_accuracy_breakdown(self, predictions: List[str], targets: List[str], dataset_type: str):
        """정확도 세부 분석 - 간결하고 유용하게"""
        if not predictions or not targets:
            return
        
        try:
            from .metrics import exact_match_score
            
            # 정확도 계산
            correct = sum(exact_match_score(p, t, dataset_type) for p, t in zip(predictions, targets))
            total = len(predictions)
            accuracy = correct / total
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 1. 전체 정확도 파이 차트
            labels = ['Correct', 'Incorrect']
            sizes = [correct, total - correct]
            colors = ['#2ecc71', '#e74c3c']  # 초록, 빨강
            
            wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90,
                                              textprops={'fontsize': 11})
            ax1.set_title(f'Overall Accuracy\n{correct}/{total} = {accuracy:.1%}', 
                         fontsize=12, fontweight='bold')
            
            # 2. 샘플 분석 (처음 10개)
            sample_size = min(10, len(predictions))
            sample_correct = [exact_match_score(predictions[i], targets[i], dataset_type) 
                            for i in range(sample_size)]
            
            x_pos = range(sample_size)
            colors_bar = ['#2ecc71' if c else '#e74c3c' for c in sample_correct]
            
            ax2.bar(x_pos, [1] * sample_size, color=colors_bar, alpha=0.7)
            ax2.set_title(f'Sample Results (First {sample_size})', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Correct (1) / Incorrect (0)')
            ax2.set_xticks(x_pos)
            ax2.set_ylim(0, 1.2)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 정확한 개수 표시
            sample_correct_count = sum(sample_correct)
            ax2.text(0.02, 0.95, f'{sample_correct_count}/{sample_size}', 
                    transform=ax2.transAxes, fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            save_path = self.output_dir / 'accuracy_breakdown.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 정확도 분석 저장: {save_path.name}")
            
        except Exception as e:
            print(f"⚠️ 정확도 분석 실패: {str(e)[:50]}...")