# configs/gsm8k_config.py

from .base_config import BaseConfig

def get_config(model_size="micro"):
    """GSM8K 실험용 설정"""
    config = BaseConfig()
    
    # 모델 크기 설정
    config.set_model_size(model_size)
    
    # GSM8K 특화 설정
    config.update(
        dataset_name="gsm8k",
        task_prefix="solve",
        
        # 시퀀스 길이 (수학 문제는 조금 길게)
        max_seq_len=256 if model_size in ["tiny", "small", "base"] else 128,
        answer_max_length=32,
        problem_max_length=224 if model_size in ["tiny", "small", "base"] else 96,
        
        # 수학 특화 설정
        max_reasoning_steps=config.max_reasoning_steps + 1,  # 수학은 추론 1단계 더
        reasoning_cost_weight=0.005,  # 추론 비용 조금 낮게
        
        # GSM8K 특화 정규화
        early_stopping_patience=3,
        eval_every=25 if model_size == "nano" else 50,
    )
    
    # 모델 정보 출력
    config.print_model_info()
    config.analyze_overfitting_risk(8792)  # GSM8K total size
    
    return config
