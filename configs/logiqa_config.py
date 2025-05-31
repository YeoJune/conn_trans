# configs/logiqa_config.py
from .base_config import BaseConfig

def get_config(model_size="base"):
    """LogiQA 실험용 설정"""
    config = BaseConfig()
    
    # LogiQA 특화 설정
    config.update(
        # 데이터 설정
        max_seq_len=256,  # LogiQA는 상대적으로 짧음
        batch_size=32,
        task_prefix=config.get_task_prefix("logiqa"),  # "reason"
        dataset_name="logiqa",
        
        # 훈련 설정
        num_epochs=15,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        
        # 토크나이저 설정 (T5 고정)
        tokenizer_name="t5-base",
        
        # 데이터셋별 특수 설정
        answer_max_length=64,
        context_max_length=192
    )
    
    if model_size == "large":
        config.update(
            d_model=512,
            num_slots=256,
            bilinear_rank=64,
            batch_size=16,  # 메모리 때문에 배치 크기 줄임
            max_reasoning_steps=8
        )
    
    return config