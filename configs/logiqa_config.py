# configs/logiqa_config.py
from .base_config import BaseConfig

def get_config(model_size="micro"):
    """LogiQA Encoder-Decoder 실험용 설정"""
    config = BaseConfig()
    
    # 모델 크기 설정
    config.set_model_size(model_size)
    
    # LogiQA 특화 설정
    config.update(
        dataset_name="logiqa",
        task_prefix="reason",
        
        # Encoder-Decoder 시퀀스 길이 설정
        max_seq_len=256 if model_size in ["tiny", "small", "base"] else 128,
        answer_max_length=16,  # A, B, C, D 답변은 짧음
        
        # LogiQA 특화 정규화
        early_stopping_patience=3,
        eval_every=25 if model_size == "nano" else 50,
        
        # 논리 추론 최적화
        max_reasoning_steps=config.max_reasoning_steps + 1,  # 논리 추론 단계 증가
        convergence_threshold=0.05,  # 더 정밀한 수렴
    )
    
    # 모델 정보 출력
    config.print_model_info()
    config.analyze_overfitting_risk(8027)  
    
    # Baseline 호환 설정 출력
    baseline_config = config.get_compatible_baseline_config()
    print(f"\n🔄 Compatible Baseline Config:")
    print(f"   Encoder layers: {baseline_config['num_encoder_layers']}")
    print(f"   Decoder layers: {baseline_config['num_decoder_layers']}")
    print(f"   FFN multiplier: {baseline_config['ffn_multiplier']}")
    
    return config

