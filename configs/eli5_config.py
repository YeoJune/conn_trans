# configs/eli5_config.py
from .base_config import BaseConfig

def get_config(model_size="base"):
    """ELI5 설정 - 파이프라인 호환성 개선"""
    config = BaseConfig().set_size(model_size).set_dataset(
        "eli5",
        task_prefix="explain",
        answer_max_length=200,      # 🔧 현실적인 길이로 조정 
        max_seq_len=320,           # 🔧 질문 길이 고려
        num_epochs=12,              # 🔧 적당한 에포크
    )
    
    
    return config