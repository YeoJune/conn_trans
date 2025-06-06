# configs/__init__.py
from .base_config import BaseConfig
from . import logiqa_config, gsm8k_config, strategyqa_config, multinli_config, eli5_config, commongen_config

__all__ = [
    'BaseConfig', 
    'logiqa_config', 
    'gsm8k_config', 
    'strategyqa_config', 
    'multinli_config',
    'eli5_config',
    'commongen_config'
]