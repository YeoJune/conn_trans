# utils/visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_connection_matrix(model, save_path="connection_matrix.png", title_suffix=""):
    if not hasattr(model, 'C') or model.C is None:
        print(f"Model {title_suffix} doesn't have Connection Matrix 'C'")
        return
    with torch.no_grad():
        C_numpy = model.C.data.cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(C_numpy, cmap='RdBu_r', center=0, square=True, linewidths=0.01, cbar_kws={"shrink": .7})
    plt.title(f'Connection Matrix (C){title_suffix}', fontsize=12)
    plt.xlabel('Slot Index', fontsize=10);
    plt.ylabel('Slot Index', fontsize=10)
    plt.xticks(fontsize=8);
    plt.yticks(fontsize=8);
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=200); print(f"💾 CM saved: {save_path}")
    except Exception as e:
        print(f"⚠️ Error saving CM: {e}")
    plt.close()


def analyze_reasoning_evolution(model, sample_batch, save_path="reasoning_evolution.png", model_name=""):
    if not hasattr(model, 'get_reasoning_trace'):
        print(f"Model {model_name} doesn't support get_reasoning_trace")
        return None, None
    model.eval()
    trace_states, norms = None, []
    with torch.no_grad():
        device = next(model.parameters()).device
        input_ids_s = sample_batch['input_ids'].to(device)
        attention_mask_s = sample_batch['attention_mask'].to(device)
        token_type_ids_s = sample_batch.get('token_type_ids', None)
        if token_type_ids_s is not None: token_type_ids_s = token_type_ids_s.to(device)

        trace_states, norms = model.get_reasoning_trace(input_ids_s, attention_mask_s, token_type_ids_s)

    if not norms: print(f"No norm trace from {model_name}."); return trace_states, norms
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(norms)), norms, 'o-', linewidth=1.5, markersize=5)
    plt.xlabel('Reasoning Step');
    plt.ylabel('Avg. Activation Norm')
    plt.title(f'Reasoning State Norm Evolution ({model_name})');
    plt.grid(True, alpha=0.6)
    plt.tight_layout();
    try:
        plt.savefig(save_path, dpi=150); print(f"💾 Evo plot saved: {save_path}")
    except Exception as e:
        print(f"⚠️ Error saving evo plot: {e}")
    plt.close()
    print(f"  Norm evo ({model_name}): {' → '.join([f'{n:.2f}' for n in norms])}")
    return trace_states, norms


def print_comparison_results(results_dict, metric_name="Accuracy"):
    print("\n" + f"🎯 MODEL COMPARISON ({metric_name})" + "\n" + "=" * 70)
    if not results_dict: print("  No results."); return
    valid_results = {k: v for k, v in results_dict.items() if isinstance(v, float)}
    if not valid_results: print("  No valid float results."); return
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1], reverse=True)
    print(f"🏆 Performance Ranking ({metric_name}):")
    for i, (model_name, acc_val) in enumerate(sorted_results, 1):  # acc -> acc_val
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
        print(f"  {emoji} {model_name:<30}: {acc_val:.4f}")
    print("=" * 70)