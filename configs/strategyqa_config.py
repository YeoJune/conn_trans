# configs/strategyqa_config.py
from .base_config import BaseConfig

def get_config(model_size="nano"):  # StrategyQA는 데이터가 가장 적어서 nano 기본
    """StrategyQA Encoder-Decoder 실험용 설정"""
    config = BaseConfig()
    
    # 모델 크기 설정
    if model_size not in ["nano", "micro"]:
        print("⚠️ Warning: StrategyQA 데이터가 매우 적습니다. nano 또는 micro 모델 권장.")
    
    config.set_model_size(model_size)
    
    # StrategyQA 특화 설정
    config.update(
        dataset_name="strategyqa",
        task_prefix="strategy",
        
        # Encoder-Decoder 시퀀스 길이 설정
        max_seq_len=128 if model_size in ["micro", "tiny"] else 96,
        answer_max_length=8,   # Yes, No + 간단한 설명
        
        # 극강 정규화 (데이터가 가장 적음)
        dropout=0.5 if model_size == "nano" else 0.4,
        weight_decay=0.3 if model_size == "nano" else 0.2,
        learning_rate=1e-5 if model_size == "nano" else 2e-5,
        orthogonal_weight=0.2,
        
        # 매우 빠른 종료
        num_epochs=2,
        early_stopping_patience=2,
        eval_every=20,
        
        # Yes/No 질문 특화
        label_smoothing=0.05,  # 간단한 답변이므로 낮게
        max_reasoning_steps=max(1, config.max_reasoning_steps - 1),  # 추론 단계 줄임
    )
    
    # 모델 정보 출력
    config.print_model_info()
    config.analyze_overfitting_risk(2780)  
    
    # Baseline 호환 설정 출력
    baseline_config = config.get_compatible_baseline_config()
    print(f"\n🔄 Compatible Baseline Config:")
    print(f"   Encoder layers: {baseline_config['num_encoder_layers']}")
    print(f"   Decoder layers: {baseline_config['num_decoder_layers']}")
    print(f"   FFN multiplier: {baseline_config['ffn_multiplier']}")
    
    return config

