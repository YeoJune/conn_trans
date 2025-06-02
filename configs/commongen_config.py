# configs/commongen_config.py (수정됨)
from .base_config import BaseConfig

def get_config(model_size="base"):
    """CommonGen 설정 - 파이프라인 호환성 개선"""
    config = BaseConfig().set_size(model_size).set_dataset(
        "commongen",
        task_prefix="connect",
        answer_max_length=80,       # 🔧 더 현실적인 길이
        max_seq_len=200,           # 🔧 개념 나열은 짧음
        num_epochs=10,             # 🔧 적당한 에포크
        batch_size=32,             # 🔧 배치 사이즈 유지
        learning_rate=1e-4,        # 🔧 표준 학습률
        gradient_clip=1.0,
        label_smoothing = 0.05
    )
    
    # 🔧 FIX: 개념 연결에 특화된 설정
    config.max_reasoning_steps = 2  # 🔧 개념 연결은 단순할 수 있음
    config.convergence_threshold = 0.02
    config.orthogonal_weight = 0.012   # 🔧 적당한 정규화
    config.early_stopping_patience = 4
    config.eval_every = 200
    config.warmup_ratio = 0.08
    config.weight_decay = 0.008
    
    return config
