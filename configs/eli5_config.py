# configs/eli5_config.py
from .base_config import BaseConfig

def get_config(model_size="base"):
    """ELI5 설정 - 파이프라인 호환성 개선"""
    config = BaseConfig().set_size(model_size).set_dataset(
        "eli5",
        task_prefix="explain",
        answer_max_length=200,      # 🔧 현실적인 길이로 조정 
        max_seq_len=320,           # 🔧 질문 길이 고려
        num_epochs=6,              # 🔧 적당한 에포크
        batch_size=12,             # 🔧 메모리 고려해서 줄임
        learning_rate=8e-5,        # 🔧 조금 낮춤
        gradient_clip=1.0,
        label_smoothing=0.1
    )
    
    # 🔧 FIX: 긴 생성에 특화된 설정
    config.early_stopping_patience = 3
    config.eval_every = 250
    config.warmup_ratio = 0.1      # 🔧 표준 워밍업
    config.weight_decay = 0.01
    
    # 🔧 Connection Transformer 특화
    config.max_reasoning_steps = 3  # ELI5는 복잡한 추론 필요
    config.convergence_threshold = 0.015
    config.orthogonal_weight = 0.01
    
    return config