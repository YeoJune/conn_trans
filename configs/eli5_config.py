# configs/eli5_config.py
from .base_config import BaseConfig

def get_config(model_size="base"):
    """ELI5 설정"""
    return BaseConfig().set_size(model_size).set_dataset("eli5")