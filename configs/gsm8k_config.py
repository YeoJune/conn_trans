# configs/gsm8k_config.py
from .base_config import BaseConfig

def get_config(model_size="micro"):
    """GSM8K Encoder-Decoder 실험용 설정"""
    config = BaseConfig()
    
    # 모델 크기 설정
    config.set_model_size(model_size)
    
    # GSM8K 특화 설정
    config.update(
        dataset_name="gsm8k",
        task_prefix="solve",
        
        # Encoder-Decoder 시퀀스 길이 설정
        max_seq_len=256 if model_size in ["tiny", "small", "base"] else 128,
        answer_max_length=48 if model_size in ["tiny", "small", "base"] else 32,
        
        # 수학 특화 설정
        max_reasoning_steps=config.max_reasoning_steps + 1,  
        reasoning_cost_weight=0.005,  
        
        # GSM8K 특화 정규화
        early_stopping_patience=3,
        eval_every=25 if model_size == "nano" else 50,
        
        # Decoder 최적화 (수학 문제는 순차적 생성이 중요)
        label_smoothing=0.05,  # 수학 답은 정확해야 하므로 낮게
    )
    
    # 모델 정보 출력
    config.print_model_info()
    config.analyze_overfitting_risk(8792)  
    
    # Baseline 호환 설정 출력
    baseline_config = config.get_compatible_baseline_config()
    print(f"\n🔄 Compatible Baseline Config:")
    print(f"   Encoder layers: {baseline_config['num_encoder_layers']}")
    print(f"   Decoder layers: {baseline_config['num_decoder_layers']}")
    print(f"   FFN multiplier: {baseline_config['ffn_multiplier']}")
    print(f"   Parameter difference: {baseline_config['param_diff']:,} ({baseline_config['param_diff']/baseline_config['total_params']*100:.1f}%)")
    
    return config

