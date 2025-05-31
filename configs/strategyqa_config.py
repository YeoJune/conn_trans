# configs/strategyqa_config.py

from .base_config import BaseConfig

def get_config(model_size="nano"):  # StrategyQA는 기본 nano
    """StrategyQA 실험용 설정"""
    config = BaseConfig()
    
    # 모델 크기 설정 (StrategyQA는 데이터가 가장 적어서 nano 기본)
    if model_size == "micro" and model_size != "nano":
        print("⚠️ Warning: StrategyQA 데이터가 매우 적습니다. nano 모델 권장.")
    
    config.set_model_size(model_size)
    
    # StrategyQA 특화 설정
    config.update(
        dataset_name="strategyqa",
        task_prefix="strategy",
        
        # 시퀀스 길이 (Yes/No 질문은 짧음)
        max_seq_len=96,
        answer_max_length=8,   # "Yes" or "No" + 짧은 설명
        question_max_length=88,
        
        # 극강 정규화 (데이터가 가장 적음)
        dropout=0.4,
        weight_decay=0.2,
        learning_rate=1e-5 if model_size == "nano" else 2e-5,
        
        # 매우 빠른 종료
        num_epochs=2,
        early_stopping_patience=2,
        eval_every=20,
    )
    
    # 모델 정보 출력
    config.print_model_info()
    config.analyze_overfitting_risk(2780)  # StrategyQA total size
    
    return config