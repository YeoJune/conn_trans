# configs/logiqa_config.py

from .base_config import BaseConfig

def get_config(model_size="micro"):
    """LogiQA 실험용 설정"""
    config = BaseConfig()
    
    # 모델 크기 설정
    config.set_model_size(model_size)
    
    # LogiQA 특화 설정
    config.update(
        dataset_name="logiqa",
        task_prefix="reason",
        
        # 시퀀스 길이 (LogiQA는 짧음)
        max_seq_len=128,
        answer_max_length=16,
        context_max_length=112,
        
        # LogiQA 특화 정규화
        early_stopping_patience=3,
        eval_every=25 if model_size == "nano" else 50,
    )
    
    # 모델 정보 출력
    config.print_model_info()
    config.analyze_overfitting_risk(8027)  # LogiQA total size
    
    return config
