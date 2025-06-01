# utils/__init__.py

# 메트릭 시스템 - T5에 최적화된 함수들
from .metrics import (
    calculate_accuracy,
    calculate_comprehensive_metrics,
    analyze_error_cases,
    extract_final_answer,
    exact_match_score,
    detect_dataset_type
)

# 시각화 시스템 - 간단하고 실용적인 함수들
from .visualization import (
    plot_training_curves, 
    visualize_connection_matrix, 
    analyze_reasoning_patterns,
    compare_model_performance,
    plot_reasoning_efficiency,
    plot_accuracy_breakdown
)

# 하위호환성을 위한 별칭
calculate_detailed_metrics = calculate_comprehensive_metrics

__all__ = [
    # 핵심 메트릭 함수들
    'calculate_accuracy', 
    'calculate_comprehensive_metrics',
    'calculate_detailed_metrics',
    'analyze_error_cases',
    'extract_final_answer',
    'exact_match_score',
    'detect_dataset_type',
    
    # 시각화 함수들
    'plot_training_curves', 
    'visualize_connection_matrix', 
    'analyze_reasoning_patterns',
    'compare_model_performance',
    'plot_reasoning_efficiency',
    'plot_accuracy_breakdown'
]