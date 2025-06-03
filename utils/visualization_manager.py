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
        """연결 행렬 시각화 - 더 직관적이고 명확한 버전"""
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
            
            # 더 큰 그림으로 설정
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # 1. 메인 연결 행렬 히트맵 - 색상을 꽉 차게!
            # 값이 작더라도 색상이 꽉 차도록 percentile 기반 정규화
            non_zero_values = matrix_np[matrix_np != 0]
            if len(non_zero_values) > 0:
                # 상위/하위 5% 기준으로 색상 범위 설정 (더 극적인 색상)
                vmax = np.percentile(np.abs(non_zero_values), 95)
                vmin = -vmax
            else:
                vmax = np.abs(matrix_np).max()
                vmin = -vmax
            
            # 만약 범위가 너무 작으면 강제로 확장
            if vmax < 0.01:
                vmax = 0.01
                vmin = -0.01
                
            im1 = ax1.imshow(matrix_np, 
                            cmap='RdBu_r', 
                            interpolation='nearest',
                            aspect='equal',
                            vmin=vmin,
                            vmax=vmax)
            
            # 셀 전체를 색으로 꽉 채우기 - 격자선 없애기
            ax1.set_xticks(np.arange(num_slots) + 0.5, minor=True)
            ax1.set_yticks(np.arange(num_slots) + 0.5, minor=True)
            ax1.set_xticks(range(num_slots))
            ax1.set_yticks(range(num_slots))
            ax1.set_xticklabels(range(num_slots))
            ax1.set_yticklabels(range(num_slots))
            # 격자선 제거하여 색상이 꽉 차게
            
            # 강한 연결에만 값 표시 (threshold 적용)
            threshold = np.abs(matrix_np).max() * 0.3  # 최대값의 30% 이상만 표시
            for i in range(num_slots):
                for j in range(num_slots):
                    if abs(matrix_np[i, j]) > threshold:
                        color = 'white' if abs(matrix_np[i, j]) > np.abs(matrix_np).max() * 0.6 else 'black'
                        ax1.text(j, i, f'{matrix_np[i, j]:.2f}',
                                ha='center', va='center',
                                color=color, fontweight='bold', fontsize=10)
            
            ax1.set_title('Slot-to-Slot Connection Matrix\n(빨강: 양의 영향, 파랑: 음의 영향)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax1.set_xlabel('Target Slot →', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Source Slot ↓', fontsize=12, fontweight='bold')
            
            # 컬러바 추가
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('Connection Strength', fontsize=12, fontweight='bold')
            
            # 2. 간단한 연결 분포 막대 그래프
            flat_connections = matrix_np[matrix_np != 0]
            
            # 양수/음수 연결 개수
            positive_count = (matrix_np > 0.01).sum()
            negative_count = (matrix_np < -0.01).sum()
            weak_count = ((matrix_np >= -0.01) & (matrix_np <= 0.01) & (matrix_np != 0)).sum()
            
            categories = ['Strong\nPositive\n(>0.01)', 'Weak\n(-0.01~0.01)', 'Strong\nNegative\n(<-0.01)']
            counts = [positive_count, weak_count, negative_count]
            colors = ['#d62728', '#ff7f0e', '#2ca02c']  # 빨강, 주황, 초록
            
            bars = ax2.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            ax2.set_title('Connection Distribution', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Number of Connections', fontsize=12, fontweight='bold')
            ax2.grid(True, axis='y', alpha=0.3)
            
            # 막대 위에 값 표시
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # 전체 통계 정보
            total_possible = num_slots * (num_slots - 1)
            active_connections = positive_count + negative_count + weak_count
            sparsity = 1 - (active_connections / total_possible)
            
            stats_text = (f'Total Slots: {num_slots}×{num_slots} | '
                        f'Active Connections: {active_connections}/{total_possible} ({active_connections/total_possible*100:.1f}%)\n'
                        f'Max Strength: {np.abs(matrix_np).max():.3f} | '
                        f'Sparsity: {sparsity:.3f} | '
                        f'Std: {matrix_np.std():.3f}')
            
            fig.suptitle('Connection Matrix Analysis', fontsize=16, fontweight='bold', y=0.95)
            fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11, style='italic',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85, bottom=0.15)
            
            # 파일 저장
            save_path = self.output_dir / 'connection_matrix_improved.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"🔗 개선된 연결 행렬 저장: {save_path.name}")
            print(f"   📊 활성 연결: {active_connections}/{total_possible} ({active_connections/total_possible*100:.1f}%)")
            print(f"   💪 강한 연결: {positive_count + negative_count} | 약한 연결: {weak_count}")
            print(f"   🎯 최대 연결 강도: {np.abs(matrix_np).max():.3f}")
            
        except Exception as e:
            print(f"⚠️ 연결 행렬 시각화 실패: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def plot_accuracy_breakdown(self, predictions: List[str], targets: List[str], dataset_type: str):
        """정확도 세부 분석 - 전체 데이터 기반"""
        if not predictions or not targets:
            return
        
        try:
            from .metrics import get_accuracy_breakdown
            
            # 전체 데이터에 대한 세부 분석
            breakdown = get_accuracy_breakdown(predictions, targets, dataset_type)
            
            if breakdown['total'] == 0:
                return
            
            accuracy = breakdown['accuracy']
            correct = breakdown['correct']
            total = breakdown['total']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # 1. 전체 정확도 파이 차트
            labels = ['Correct', 'Incorrect']
            sizes = [correct, total - correct]
            colors = ['#2ecc71', '#e74c3c']  # 초록, 빨강
            
            wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90,
                                              textprops={'fontsize': 12})
            ax1.set_title(f'Overall Accuracy\n{correct}/{total} = {accuracy:.1%}', 
                         fontsize=14, fontweight='bold')
            
            # 2. 샘플별 정확도 분포 (처음 20개)
            sample_size = min(20, len(breakdown['details']))
            sample_details = breakdown['details'][:sample_size]
            
            x_pos = range(sample_size)
            colors_bar = ['#2ecc71' if detail['correct'] else '#e74c3c' 
                         for detail in sample_details]
            
            bars = ax2.bar(x_pos, [1] * sample_size, color=colors_bar, alpha=0.7, width=0.8)
            
            # 정확/오답 개수 표시
            sample_correct = sum(1 for detail in sample_details if detail['correct'])
            ax2.set_title(f'Sample Results (First {sample_size})\n{sample_correct}/{sample_size} Correct', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Result')
            ax2.set_xticks(range(0, sample_size, max(1, sample_size//10)))
            ax2.set_ylim(0, 1.2)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 범례 추가
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#2ecc71', label='Correct'),
                             Patch(facecolor='#e74c3c', label='Incorrect')]
            ax2.legend(handles=legend_elements, loc='upper right')
            
            # 통계 정보 추가
            stats_text = f"Dataset: {dataset_type.upper()}\n"
            stats_text += f"Total Samples: {total:,}\n"
            stats_text += f"Accuracy: {accuracy:.1%}"
            
            ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, 
                    fontsize=10, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            save_path = self.output_dir / 'accuracy_breakdown.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 정확도 분석 저장: {save_path.name} (전체 {total}개 샘플 분석)")
            
        except Exception as e:
            print(f"⚠️ 정확도 분석 실패: {str(e)[:50]}...")