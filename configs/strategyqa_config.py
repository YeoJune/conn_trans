# configs/strategyqa_config.py
from .base_config import BaseConfig

def get_config(model_size="base"):
    """StrategyQA 실험용 설정 - RTX 4090 최적화"""
    config = BaseConfig()
    
    config.update(
        # 데이터 설정
        dataset_name="strategyqa", 
        task_prefix="strategy",
        max_seq_len=256,
        batch_size=20,       # Yes/No 문제라 배치 크게
        gradient_accumulation_steps=2,  # 실질적 batch = 40
        
        # StrategyQA 특화
        num_epochs=10,
        learning_rate=1.5e-4,  # 조금 큰 학습률
        warmup_ratio=0.12,
        max_reasoning_steps=4,   # 전략적 추론
        
        # StrategyQA 특수 설정
        answer_max_length=16,    # Yes/No + 짧은 설명
        question_max_length=240
    )
    
    if model_size == "large":
        config.update(
            d_model=512,
            num_slots=96,
            bilinear_rank=24,
            batch_size=10,
            gradient_accumulation_steps=4,  # 실질적 batch = 40
            max_reasoning_steps=6
        )
    elif model_size == "small":
        config.update(
            d_model=128,
            num_slots=32,
            bilinear_rank=8,
            batch_size=40,
            max_reasoning_steps=3,
            num_epochs=6
        )
    
    config.print_model_info()
    return config