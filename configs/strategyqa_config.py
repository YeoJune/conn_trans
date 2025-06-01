# configs/strategyqa_config.py

from .base_config import BaseConfig

def get_config(model_size="nano"):
    """StrategyQA 설정"""
    return BaseConfig().set_size(model_size).set_dataset(
        "strategyqa",
        dropout=0.5 if model_size == "nano" else 0.4,
        weight_decay=0.3 if model_size == "nano" else 0.2,
        early_stopping_patience=2,
        eval_every=20
    )