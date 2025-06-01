# configs/gsm8k_config.py

from .base_config import BaseConfig

def get_config(model_size="micro"):
    """GSM8K 설정"""
    return BaseConfig().set_size(model_size).set_dataset(
        "gsm8k",
        max_reasoning_steps=3,  # 수학은 추론 단계 하나 더
        answer_max_length=48 if model_size in ["tiny", "base"] else 32
    )

