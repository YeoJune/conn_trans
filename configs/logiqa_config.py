# configs/logiqa_config.py
from .base_config import BaseConfig

def get_config(model_size="micro"):
    """LogiQA 설정"""
    return BaseConfig().set_size(model_size).set_dataset("logiqa")