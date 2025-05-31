# configs/gsm8k_config.py  
from .base_config import BaseConfig

def get_config(model_size="base"):
    """GSM8K 실험용 설정 - RTX 4090 최적화"""
    config = BaseConfig()
    
    config.update(
        # 데이터 설정
        dataset_name="gsm8k",
        task_prefix="solve",
        max_seq_len=384,     # 수학 문제는 조금 길게
        batch_size=12,       # 긴 시퀀스로 인해 작게
        gradient_accumulation_steps=3,  # 실질적 batch = 36
        
        # 수학 문제 특화
        num_epochs=12,
        learning_rate=8e-5,   # 조금 작은 학습률
        warmup_ratio=0.15,
        max_reasoning_steps=5,  # 수학은 추론 단계 더 필요
        
        # GSM8K 특수 설정
        answer_max_length=96,
        problem_max_length=288
    )
    
    if model_size == "large":
        config.update(
            d_model=512,
            num_slots=96,
            bilinear_rank=24,
            batch_size=6,
            gradient_accumulation_steps=6,  # 실질적 batch = 36
            max_reasoning_steps=7
        )
    elif model_size == "small":
        config.update(
            d_model=128,
            num_slots=32,
            bilinear_rank=8,
            batch_size=24,
            max_reasoning_steps=4,
            num_epochs=8
        )
    
    config.print_model_info()
    return config

