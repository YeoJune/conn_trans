# utils/visualization_manager.py
import torch
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
        """간결하고 색깔이 셀에 가득 찬 연결 행렬 시각화"""
        if not hasattr(model, 'W_source') or not hasattr(model, 'W_target'):
            return
        
        try:
            # 연결 행렬 계산
            with torch.no_grad():
                connection_matrix = torch.sum(model.W_source * model.W_target, dim=-1)
                num_slots = connection_matrix.size(0)
                mask = torch.eye(num_slots, device=connection_matrix.device, dtype=torch.bool)
                connection_matrix = connection_matrix.masked_fill(mask, 0.0)
                matrix_np = connection_matrix.cpu().numpy()
            
            # 단일 큰 히트맵
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # 색상 범위 설정 - 더 극적으로
            abs_max = np.abs(matrix_np).max()
            if abs_max < 0.01:
                abs_max = 0.01  # 최소 범위 보장
            
            # 색상 꽉 찬 히트맵 - seaborn 사용
            sns.heatmap(matrix_np, 
                    cmap='RdBu_r',
                    center=0,
                    vmin=-abs_max, 
                    vmax=abs_max,
                    square=True,
                    linewidths=0.5,  # 얇은 격자선
                    linecolor='white',
                    cbar_kws={
                        'shrink': 0.8,
                        'label': 'Connection Strength'
                    },
                    ax=ax)
            
            # 축 설정 - 간결하게
            ax.set_title(f'Connection Matrix ({num_slots}×{num_slots})', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Target Slot', fontsize=12)
            ax.set_ylabel('Source Slot', fontsize=12)
            
            # 틱 설정 - 너무 많으면 간소화
            if num_slots <= 32:
                step = 1
            elif num_slots <= 128:
                step = 8
            else:
                step = 32
                
            ticks = list(range(0, num_slots, step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            
            # 통계 정보 - 간결하게
            non_zero = (matrix_np != 0).sum()
            total_possible = num_slots * (num_slots - 1)
            positive = (matrix_np > 0.01).sum()
            negative = (matrix_np < -0.01).sum()
            
            stats_text = (f'Active: {non_zero}/{total_possible} ({non_zero/total_possible*100:.0f}%) | '
                        f'Pos: {positive} | Neg: {negative} | Max: {abs_max:.3f}')
            
            ax.text(0.5, -0.08, stats_text, transform=ax.transAxes, 
                ha='center', fontsize=11, style='italic')
            
            plt.tight_layout()
            
            save_path = self.output_dir / 'connection_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"🔗 연결 행렬 저장: {save_path.name}")
            print(f"   📊 {non_zero}/{total_possible} 활성 연결 ({non_zero/total_possible*100:.0f}%)")
            
        except Exception as e:
            print(f"⚠️ 연결 행렬 시각화 실패: {str(e)}")

    def plot_final_training_curves(self, train_losses: List[float], 
                                eval_accuracies: List[float], 
                                reasoning_steps: Optional[List[float]] = None):
        """간결한 학습 곡선"""
        n_plots = 3 if reasoning_steps else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
        
        if n_plots == 2:
            axes = [axes[0], axes[1]]
        
        epochs = range(1, len(train_losses) + 1)
        
        # 1. 손실 곡선 - 간결하게
        axes[0].plot(epochs, train_losses, 'b-', linewidth=3, alpha=0.8)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 정확도 곡선 - 간결하게
        acc_epochs = range(1, len(eval_accuracies) + 1)
        axes[1].plot(acc_epochs, eval_accuracies, 'g-', linewidth=3, alpha=0.8)
        best_acc = max(eval_accuracies)
        axes[1].axhline(y=best_acc, color='r', linestyle='--', alpha=0.7)
        axes[1].set_title(f'Accuracy (Best: {best_acc:.3f})', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 추론 단계 (있으면)
        if reasoning_steps:
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
                        'orange', linewidth=3, alpha=0.8)
                axes[2].set_title('Reasoning Steps', fontsize=14, fontweight='bold')
                axes[2].set_xlabel('Epoch')
                axes[2].set_ylabel('Steps')
                axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 학습 곡선 저장: {save_path.name}")

    def plot_accuracy_breakdown(self, predictions: List[str], targets: List[str], dataset_type: str):
        """간결한 정확도 분석"""
        if not predictions or not targets:
            return
        
        try:
            from .metrics import get_accuracy_breakdown
            breakdown = get_accuracy_breakdown(predictions, targets, dataset_type)
            
            if breakdown['total'] == 0:
                return
            
            accuracy = breakdown['accuracy']
            correct = breakdown['correct']
            total = breakdown['total']
            
            # 단일 파이 차트
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            labels = ['Correct', 'Incorrect']
            sizes = [correct, total - correct]
            colors = ['#27ae60', '#e74c3c']  # 더 선명한 초록/빨강
            
            # 파이 차트 - 색깔 진하게
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                            autopct='%1.1f%%', startangle=90,
                                            textprops={'fontsize': 14, 'fontweight': 'bold'},
                                            wedgeprops={'linewidth': 2, 'edgecolor': 'white'})
            
            # 제목 - 간결하게
            ax.set_title(f'{dataset_type.upper()} Accuracy\n{correct}/{total} = {accuracy:.1%}', 
                        fontsize=16, fontweight='bold', pad=30)
            
            # 중앙에 큰 정확도 숫자
            ax.text(0, 0, f'{accuracy:.1%}', ha='center', va='center', 
                fontsize=24, fontweight='bold', color='navy')
            
            plt.tight_layout()
            save_path = self.output_dir / 'accuracy_summary.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 정확도 요약 저장: {save_path.name} ({accuracy:.1%})")
            
        except Exception as e:
            print(f"⚠️ 정확도 분석 실패: {str(e)[:50]}...")