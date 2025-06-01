# utils/visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any

# 기본 스타일 설정
plt.style.use('default')
sns.set_palette("husl")

def plot_training_curves(train_losses: List[float], 
                        eval_accuracies: List[float], 
                        reasoning_steps_history: Optional[List[float]] = None,
                        save_path: Optional[str] = None):
    """간단한 훈련 곡선 시각화"""
    
    # Connection Transformer 여부에 따라 플롯 개수 결정
    n_plots = 3 if reasoning_steps_history else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    
    epochs = range(1, len(train_losses) + 1)
    
    # 1. Training Loss
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2, alpha=0.8)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Evaluation Accuracy
    axes[1].plot(epochs[:len(eval_accuracies)], eval_accuracies, 'g-', linewidth=2, alpha=0.8)
    axes[1].set_title('Evaluation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Reasoning Steps (Connection Transformer만)
    if reasoning_steps_history:
        axes[2].plot(epochs[:len(reasoning_steps_history)], reasoning_steps_history, 'r-', linewidth=2, alpha=0.8)
        axes[2].set_title('Reasoning Steps')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Steps')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to {save_path}")
    
    plt.close()

def visualize_connection_matrix(model, save_path: Optional[str] = None, title_suffix: str = ""):
    """Connection Matrix 간단 시각화"""
    if not hasattr(model, 'get_connection_analysis'):
        print("Model doesn't support connection analysis")
        return
    
    analysis = model.get_connection_analysis()
    connection_matrix = analysis['connection_matrix'].cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    
    # 히트맵
    sns.heatmap(connection_matrix, 
                cmap='RdBu_r', 
                center=0, 
                square=True, 
                cbar_kws={"shrink": 0.8})
    
    plt.title(f'Connection Matrix{title_suffix}')
    plt.xlabel('Target Slot')
    plt.ylabel('Source Slot')
    
    # 간단한 통계
    stats = f"Max: {analysis['max_connection']:.3f}, Mean: {analysis['mean_connection']:.3f}"
    plt.figtext(0.02, 0.02, stats, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Connection matrix saved to {save_path}")
    
    plt.close()

def analyze_reasoning_patterns(model, save_path: Optional[str] = None):
    """간단한 추론 패턴 분석"""
    if not hasattr(model, 'get_connection_analysis'):
        print("Model doesn't support reasoning analysis")
        return
    
    analysis = model.get_connection_analysis()
    connection_matrix = analysis['connection_matrix'].cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 1. Connection strength distribution
    axes[0].hist(connection_matrix.flatten(), bins=30, alpha=0.7, color='skyblue')
    axes[0].set_title('Connection Strength Distribution')
    axes[0].set_xlabel('Connection Strength')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Row-wise connection strength
    row_sums = connection_matrix.sum(axis=1)
    axes[1].bar(range(len(row_sums)), row_sums, alpha=0.7, color='lightcoral')
    axes[1].set_title('Outgoing Connection Strength by Slot')
    axes[1].set_xlabel('Source Slot')
    axes[1].set_ylabel('Total Strength')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🔍 Reasoning analysis saved to {save_path}")
    
    plt.close()

def compare_model_performance(results_dict: Dict[str, Dict], save_path: Optional[str] = None):
    """간단한 모델 성능 비교"""
    models = list(results_dict.keys())
    accuracies = [results_dict[model]['best_accuracy'] for model in models]
    
    plt.figure(figsize=(8, 5))
    
    bars = plt.bar(models, accuracies, 
                   color=['skyblue', 'lightcoral', 'lightgreen'][:len(models)],
                   alpha=0.8)
    
    # 값 표시
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Model Performance Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, max(accuracies) * 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance comparison saved to {save_path}")
    
    plt.close()

def plot_reasoning_efficiency(reasoning_steps_data: List[float], save_path: Optional[str] = None):
    """간단한 추론 효율성 분석"""
    if not reasoning_steps_data:
        print("No reasoning steps data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # 1. 히스토그램
    axes[0].hist(reasoning_steps_data, bins=range(1, max(reasoning_steps_data)+2), 
                 alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title('Reasoning Steps Distribution')
    axes[0].set_xlabel('Number of Steps')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 시간에 따른 변화
    axes[1].plot(reasoning_steps_data, alpha=0.7, color='red')
    axes[1].set_title('Reasoning Steps Over Time')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Number of Steps')
    axes[1].grid(True, alpha=0.3)
    
    # 통계 정보 추가
    stats_text = f"Mean: {np.mean(reasoning_steps_data):.2f}, Std: {np.std(reasoning_steps_data):.2f}"
    plt.figtext(0.02, 0.02, stats_text, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"⚡ Reasoning efficiency plot saved to {save_path}")
    
    plt.close()

def plot_accuracy_breakdown(predictions: List[str], targets: List[str], 
                          dataset_type: str, save_path: Optional[str] = None):
    """정확도 분석 (새로운 함수)"""
    from .metrics import extract_final_answer, exact_match_score
    
    correct = sum(exact_match_score(p, t, dataset_type) for p, t in zip(predictions, targets))
    total = len(predictions)
    
    plt.figure(figsize=(8, 5))
    
    # 파이 차트
    labels = ['Correct', 'Incorrect']
    sizes = [correct, total - correct]
    colors = ['lightgreen', 'lightcoral']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'Accuracy: {correct}/{total} = {correct/total:.2%}')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Accuracy breakdown saved to {save_path}")
    
    plt.close()