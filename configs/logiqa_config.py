# configs/logiqa_config.py
from .base_config import BaseConfig

def get_config(model_size="base"):
    """LogiQA 실험용 설정 - RTX 4090 최적화"""
    config = BaseConfig()
    
    # LogiQA 특화 설정
    config.update(
        # 데이터 설정
        dataset_name="logiqa",
        task_prefix="reason",
        max_seq_len=256,
        batch_size=16,
        gradient_accumulation_steps=2,  # 실질적 batch = 32
        
        # 훈련 설정 - 빠른 실험
        num_epochs=10,       # 15 → 10
        learning_rate=1.2e-4,
        warmup_ratio = 0.1,
        
        # LogiQA 특수 설정
        answer_max_length=32,
        context_max_length=224
    )
    
    if model_size == "large":
        config.update(
            d_model=512,
            num_slots=96,        # 256 → 96
            bilinear_rank=24,    # 64 → 24
            batch_size=8,
            gradient_accumulation_steps=4,  # 실질적 batch = 32
            max_reasoning_steps=6
        )
    elif model_size == "small":
        # 빠른 테스트용
        config.update(
            d_model=128,
            num_slots=32,
            bilinear_rank=8,
            batch_size=32,
            max_reasoning_steps=3,
            num_epochs=5
        )
    
    config.print_model_info()
    return config

