# configs/multinli_config.py

from .base_config import BaseConfig

def get_config(model_size="base"):  # MultiNLI는 기본 base!
    """MultiNLI 실험용 설정 - 큰 데이터셋의 이점 활용"""
    config = BaseConfig()
    
    # 모델 크기 설정 (큰 데이터셋이므로 더 큰 모델 사용 가능)
    config.set_model_size(model_size)
    
    # MultiNLI 특화 설정 (433K 데이터로 안전함)
    config.update(
        dataset_name="multinli",
        task_prefix="infer",
        
        # 시퀀스 길이 (premise + hypothesis가 길 수 있음)
        max_seq_len=384 if model_size in ["base", "small"] else 256,
        answer_max_length=16,  # "entailment", "neutral", "contradiction"
        premise_max_length=192 if model_size in ["base", "small"] else 128,
        hypothesis_max_length=128 if model_size in ["base", "small"] else 96,
        
        # 🎯 큰 데이터셋의 이점 - 정규화 완화 가능
        learning_rate=1e-4 if model_size == "base" else 8e-5,
        weight_decay=0.01 if model_size == "base" else 0.05,  # 덜 강한 정규화
        dropout=0.1 if model_size == "base" else 0.2,         # 덜 강한 dropout
        orthogonal_weight=0.01 if model_size == "base" else 0.05,
        
        # 🎯 더 많은 에폭 가능 (오버피팅 위험 낮음)
        num_epochs=8 if model_size == "base" else 5,
        batch_size=32 if model_size == "base" else 16,
        gradient_accumulation_steps=2 if model_size == "base" else 4,
        
        # 조기 종료 설정 (여유롭게)
        early_stopping_patience=5,  # 더 긴 patience
        eval_every=500,             # 덜 자주 평가 (안정적)
        
        # 평가 최적화
        label_smoothing=0.1,        # 덜 강한 smoothing
    )
    
    # 🔥 MultiNLI 특별 최적화
    if model_size == "base":
        print("🚀 MultiNLI 대용량 데이터셋 - base 모델 안전 사용!")
        config.update(
            # Base 모델도 안전하게 사용 가능
            reasoning_cost_weight=0.005,  # 추론 비용 완화
            max_reasoning_steps=5,        # 더 많은 추론 단계
        )
    
    # 모델 정보 출력
    config.print_model_info()
    config.analyze_overfitting_risk(433000)  # MultiNLI 총 크기
    
    return config