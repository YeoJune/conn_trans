# configs/multinli_config.py
from .base_config import BaseConfig

def get_config(model_size="base"):
    """MultiNLI 설정"""
    return BaseConfig().set_size(model_size).set_dataset("multinli")