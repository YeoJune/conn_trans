# configs/multinli_config.py

from .base_config import BaseConfig

def get_config(model_size="base"):
    """MultiNLI 설정"""
    return BaseConfig().set_size(model_size).set_dataset(
        "multinli",
        max_seq_len=512 if model_size == "base" else 384,
        batch_size=32 if model_size == "base" else 16,
        dropout=0.1 if model_size == "base" else 0.2,
        weight_decay=0.01 if model_size == "base" else 0.05
    )

