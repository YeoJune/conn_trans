# configs/gsm8k_config.py
from .base_config import BaseConfig

def get_config(model_size="base"):
    """GSM8K 실험용 설정"""
    config = BaseConfig()
    
    # GSM8K 특화 설정
    config.update(
        # 데이터 설정
        max_seq_len=512,  # 수학 문제는 길 수 있음
        batch_size=24,    # 조금 작게
        task_prefix=config.get_task_prefix("gsm8k"),  # "solve"
        dataset_name="gsm8k",
        
        # 훈련 설정
        num_epochs=20,
        learning_rate=8e-5,  # 조금 작은 학습률
        warmup_ratio=0.15,
        max_reasoning_steps=8,  # 수학 문제는 더 많은 추론 필요
        
        # 토크나이저 설정 (T5 고정)
        tokenizer_name="t5-base",
        
        # 데이터셋별 특수 설정
        answer_max_length=128,  # 수학 답안은 길 수 있음
        problem_max_length=384
    )
    
    if model_size == "large":
        config.update(
            d_model=512,
            num_slots=256,
            bilinear_rank=64,
            batch_size=12,
            max_reasoning_steps=10
        )
    
    return config