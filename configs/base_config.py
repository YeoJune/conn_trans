# configs/base_config.py

# 모든 실험에 공통적으로 적용될 수 있는 기본 하이퍼파라미터
# 실제 값은 각 실험용 config 파일에서 덮어쓰거나 확장될 수 있음

BASE_CONFIG = {
    # Architecture parameters
    "d_model": 512,
    "num_slots": 512,  # ConnTrans의 N (Number of semantic slots)
    "num_reasoning_steps": 4,  # ConnTrans의 K (Number of iterative reasoning steps)
    "num_transformer_layers": 4,  # StandardTransformer의 레이어 수 (K와 동일하게 설정 가능)
    "num_heads": 8,  # MultiHeadAttention 헤드 수
    "ffn_dim_multiplier": 4,  # FFN 내부 차원 (d_model * multiplier)
    "dropout": 0.1,

    # Training parameters
    "weight_decay": 0.01,
    "gradient_clip": 1.0,

    # Stability parameters (ConnTrans용)
    "connection_init_std": 0.01,
    "spectral_radius_limit": 0.95,  # enforce_spectral_radius에서 사용
    "connection_regularization": 1e-4,

    # Tokenizer (데이터셋별 config에서 구체적인 모델명 지정)
    "tokenizer_name": "bert-base-uncased",  # 기본값, 필요시 변경
}


def get_base_config():
    return BASE_CONFIG.copy()  # 복사본 반환하여 원본 변경 방지