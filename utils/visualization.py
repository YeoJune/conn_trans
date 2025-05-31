# utils/visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

def plot_training_curves(train_losses, eval_accuracies, reasoning_steps_history=None, save_path=None):
    """훈련 곡선 시각화"""
    fig, axes = plt.subplots(1, 3 if reasoning_steps_history else 2, figsize=(15, 5))
    
    # Loss 곡선
    axes[0].plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Accuracy 곡선
    axes[1].plot(eval_accuracies, label='Eval Accuracy', color='green', alpha=0.7)
    axes[1].set_title('Evaluation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Reasoning Steps (Connection Transformer만)
    if reasoning_steps_history:
        axes[2].plot(reasoning_steps_history, label='Avg Reasoning Steps', color='red', alpha=0.7)
        axes[2].set_title('Average Reasoning Steps')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Steps')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to {save_path}")
    
    plt.close()

def visualize_connection_matrix(model, save_path=None, title_suffix=""):
    """Connection Matrix 시각화"""
    if not hasattr(model, 'get_connection_analysis'):
        print(f"Model doesn't support connection analysis")
        return
    
    analysis = model.get_connection_analysis()
    connection_matrix = analysis['connection_matrix'].cpu().numpy()
    
    plt.figure(figsize=(12, 10))
    
    # 히트맵 생성
    sns.heatmap(
        connection_matrix, 
        cmap='RdBu_r', 
        center=0, 
        square=True, 
        linewidths=0.01,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title(f'Bilinear Connection Matrix{title_suffix}', fontsize=14)
    plt.xlabel('Target Slot Index', fontsize=12)
    plt.ylabel('Source Slot Index', fontsize=12)
    
    # 통계 정보 추가
    stats_text = f"""Statistics:
Max: {analysis['max_connection']:.3f}
Mean: {analysis['mean_connection']:.3f}
Sparsity: {analysis['sparsity_ratio']:.1%}"""
    
    plt.text(1.02, 0.5, stats_text, transform=plt.gca().transAxes,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Connection matrix saved to {save_path}")
    
    plt.close()

def analyze_reasoning_patterns(model, sample_input=None, save_path=None):
    """추론 패턴 분석"""
    if not hasattr(model, 'get_connection_analysis'):
        print("Model doesn't support reasoning analysis")
        return
    
    # Connection matrix 분석
    analysis = model.get_connection_analysis()
    connection_matrix = analysis['connection_matrix'].cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Connection strength distribution
    axes[0, 0].hist(connection_matrix.flatten(), bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Connection Strength Distribution')
    axes[0, 0].set_xlabel('Connection Strength')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Sparsity pattern
    binary_matrix = (connection_matrix > 0.01).astype(float)
    im2 = axes[0, 1].imshow(binary_matrix, cmap='Blues', aspect='equal')
    axes[0, 1].set_title('Sparsity Pattern (Threshold: 0.01)')
    axes[0, 1].set_xlabel('Target Slot')
    axes[0, 1].set_ylabel('Source Slot')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. Row-wise connection strength (how much each slot influences others)
    row_sums = connection_matrix.sum(axis=1)
    axes[1, 0].bar(range(len(row_sums)), row_sums, alpha=0.7, color='lightcoral')
    axes[1, 0].set_title('Outgoing Connection Strength by Slot')
    axes[1, 0].set_xlabel('Source Slot Index')
    axes[1, 0].set_ylabel('Total Outgoing Strength')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Column-wise connection strength (how much each slot is influenced)
    col_sums = connection_matrix.sum(axis=0)
    axes[1, 1].bar(range(len(col_sums)), col_sums, alpha=0.7, color='lightgreen')
    axes[1, 1].set_title('Incoming Connection Strength by Slot')
    axes[1, 1].set_xlabel('Target Slot Index')
    axes[1, 1].set_ylabel('Total Incoming Strength')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🔍 Reasoning analysis saved to {save_path}")
    
    plt.close()

def plot_reasoning_convergence(reasoning_trace, save_path=None):
    """추론 수렴 과정 시각화"""
    if not reasoning_trace or len(reasoning_trace) == 0:
        print("No reasoning trace available")
        return
    
    # 각 스텝에서의 활성화 norm 계산
    step_norms = []
    for step_state in reasoning_trace:
        # [batch, num_slots, d_model] -> [num_slots]
        norms = torch.norm(step_state, dim=-1).mean(dim=0).cpu().numpy()
        step_norms.append(norms)
    
    step_norms = np.array(step_norms)  # [num_steps, num_slots]
    
    plt.figure(figsize=(12, 8))
    
    # 각 슬롯의 활성화 변화 시각화
    for slot_idx in range(min(10, step_norms.shape[1])):  # 처음 10개 슬롯만
        plt.plot(step_norms[:, slot_idx], 
                label=f'Slot {slot_idx}', 
                alpha=0.7,
                linewidth=2)
    
    plt.title('Reasoning State Convergence', fontsize=14)
    plt.xlabel('Reasoning Step', fontsize=12)
    plt.ylabel('Activation Norm', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Convergence plot saved to {save_path}")
    
    plt.close()

def compare_model_performance(results_dict, save_path=None):
    """모델 성능 비교 시각화"""
    models = list(results_dict.keys())
    accuracies = [results_dict[model]['best_accuracy'] for model in models]
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(models, accuracies, 
                   color=['skyblue', 'lightcoral', 'lightgreen'][:len(models)],
                   alpha=0.8)
    
    # 막대 위에 정확도 값 표시
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Model Performance Comparison', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, max(accuracies) * 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance comparison saved to {save_path}")
    
    plt.close()

def plot_reasoning_efficiency(reasoning_steps_data, save_path=None):
    """추론 효율성 분석"""
    if not reasoning_steps_data:
        print("No reasoning steps data available")
        return
    
    plt.figure(figsize=(12, 8))
    
    # 히스토그램
    plt.subplot(2, 2, 1)
    plt.hist(reasoning_steps_data, bins=range(1, max(reasoning_steps_data)+2), 
             alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Reasoning Steps')
    plt.xlabel('Number of Steps')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 시간에 따른 변화
    plt.subplot(2, 2, 2)
    plt.plot(reasoning_steps_data, alpha=0.7, color='red')
    plt.title('Reasoning Steps Over Time')
    plt.xlabel('Sample Index')
    plt.ylabel('Number of Steps')
    plt.grid(True, alpha=0.3)
    
    # 누적 분포
    plt.subplot(2, 2, 3)
    sorted_steps = np.sort(reasoning_steps_data)
    y_values = np.arange(1, len(sorted_steps) + 1) / len(sorted_steps)
    plt.plot(sorted_steps, y_values, linewidth=2, color='green')
    plt.title('Cumulative Distribution')
    plt.xlabel('Number of Steps')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.3)
    
    # 통계 정보
    plt.subplot(2, 2, 4)
    stats_text = f"""Statistics:
Mean: {np.mean(reasoning_steps_data):.2f}
Median: {np.median(reasoning_steps_data):.2f}
Std: {np.std(reasoning_steps_data):.2f}
Min: {np.min(reasoning_steps_data)}
Max: {np.max(reasoning_steps_data)}
Early termination rate: {(np.array(reasoning_steps_data) < max(reasoning_steps_data)).mean():.1%}"""
    
    plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes,
             verticalalignment='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"⚡ Reasoning efficiency plot saved to {save_path}")
    
    plt.close()