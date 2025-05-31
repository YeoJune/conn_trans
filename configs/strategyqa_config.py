# configs/strategyqa_config.py
from .base_config import BaseConfig

def get_config(model_size="base"):
    """StrategyQA 실험용 설정"""
    config = BaseConfig()
    
    # StrategyQA 특화 설정
    config.update(
        # 데이터 설정
        max_seq_len=256,  # 질문이 비교적 짧음
        batch_size=36,
        task_prefix=config.get_task_prefix("strategyqa"),  # "strategy"
        dataset_name="strategyqa",
        
        # 훈련 설정
        num_epochs=18,
        learning_rate=1.2e-4,
        warmup_ratio=0.12,
        
        # 토크나이저 설정 (T5 고정)
        tokenizer_name="t5-base",
        
        # 데이터셋별 특수 설정
        answer_max_length=32,   # Yes/No + 설명
        question_max_length=224
    )
    
    if model_size == "large":
        config.update(
            d_model=512,
            num_slots=256,
            bilinear_rank=64,
            batch_size=18,
            max_reasoning_steps=8
        )
    
    return config