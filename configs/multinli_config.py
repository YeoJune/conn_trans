# configs/multinli_config.py
from .base_config import BaseConfig

def get_config(model_size="base"):  # MultiNLI는 큰 데이터셋이므로 base 기본
    """MultiNLI Encoder-Decoder 실험용 설정"""
    config = BaseConfig()
    
    # 모델 크기 설정
    config.set_model_size(model_size)
    
    # MultiNLI 특화 설정 (큰 데이터셋의 이점 활용)
    config.update(
        dataset_name="multinli",
        task_prefix="infer",
        
        # Encoder-Decoder 시퀀스 길이 설정
        max_seq_len=512 if model_size == "base" else 384,
        answer_max_length=16,  # entailment, neutral, contradiction
        
        # 큰 데이터셋의 이점 - 정규화 완화
        learning_rate=1e-4 if model_size == "base" else 8e-5,
        weight_decay=0.01 if model_size == "base" else 0.05,
        dropout=0.1 if model_size == "base" else 0.2,
        orthogonal_weight=0.01 if model_size == "base" else 0.05,
        
        # 더 많은 에폭 가능
        num_epochs=8 if model_size == "base" else 5,
        batch_size=32 if model_size == "base" else 16,
        gradient_accumulation_steps=2 if model_size == "base" else 4,
        
        # 조기 종료 설정
        early_stopping_patience=5,
        eval_every=500,
        
        # NLI 특화 최적화
        label_smoothing=0.1,
        max_reasoning_steps=config.max_reasoning_steps + 2,  # NLI는 복잡한 추론
    )
    
    # 모델 정보 출력
    config.print_model_info()
    config.analyze_overfitting_risk(433000)  
    
    # Baseline 호환 설정 출력
    baseline_config = config.get_compatible_baseline_config()
    print(f"\n🔄 Compatible Baseline Config:")
    print(f"   Encoder layers: {baseline_config['num_encoder_layers']}")
    print(f"   Decoder layers: {baseline_config['num_decoder_layers']}")
    print(f"   FFN multiplier: {baseline_config['ffn_multiplier']}")
    
    return config
