# configs/strategyqa_config.py

from .base_config import BaseConfig

def get_config(model_size="x-small"):
    """StrategyQA 설정"""
    return BaseConfig().set_size(model_size).set_dataset(
        "strategyqa",
        dropout=0.5,
        weight_decay=0.3,
        early_stopping_patience=2,
        eval_every=20
    )