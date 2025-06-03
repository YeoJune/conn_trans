# configs/commongen_config.py (수정됨)
from .base_config import BaseConfig

def get_config(model_size="base"):
    """CommonGen 설정 - 파이프라인 호환성 개선"""
    config = BaseConfig().set_size(model_size).set_dataset(
        "commongen",
        task_prefix="connect",
        answer_max_length=80,       # 🔧 더 현실적인 길이
        max_seq_len=200,           # 🔧 개념 나열은 짧음
        num_epochs=8,             # 🔧 적당한 에포크
    )
    
    return config
