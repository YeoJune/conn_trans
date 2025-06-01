# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

plt.style.use('default')

def plot_training_curves(train_losses: List[float], eval_accuracies: List[float], 
                        reasoning_steps: Optional[List[float]] = None, save_path: Optional[str] = None):
    """훈련 곡선 시각화"""
    n_plots = 3 if reasoning_steps else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs[:len(eval_accuracies)], eval_accuracies, 'g-', linewidth=2)
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    # Reasoning steps
    if reasoning_steps:
        axes[2].plot(epochs[:len(reasoning_steps)], reasoning_steps, 'r-', linewidth=2)
        axes[2].set_title('Reasoning Steps')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Steps')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {save_path}")
    
    plt.close()

def visualize_connection_matrix(model, save_path: Optional[str] = None):
    """Connection Matrix 시각화"""
    if not hasattr(model, 'get_connection_analysis'):
        return
    
    analysis = model.get_connection_analysis()
    matrix = analysis['connection_matrix'].cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap='RdBu_r', center=0, square=True)
    plt.title('Connection Matrix')
    plt.xlabel('Target Slot')
    plt.ylabel('Source Slot')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Matrix saved: {save_path}")
    
    plt.close()

def plot_accuracy_breakdown(predictions: List[str], targets: List[str], 
                          dataset_type: str, save_path: Optional[str] = None):
    """정확도 분석"""
    from .metrics import exact_match_score
    
    correct = sum(exact_match_score(p, t, dataset_type) for p, t in zip(predictions, targets))
    total = len(predictions)
    
    plt.figure(figsize=(6, 4))
    
    labels = ['Correct', 'Incorrect']
    sizes = [correct, total - correct]
    colors = ['lightgreen', 'lightcoral']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title(f'Accuracy: {correct}/{total} = {correct/total:.2%}')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Breakdown saved: {save_path}")
    
    plt.close()
