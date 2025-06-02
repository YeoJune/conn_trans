# configs/base_config.py
class BaseConfig:
    """
    Simple한 기본 설정
    """
    
    def __init__(self):
        # 모델 아키텍처 (Attention Is All You Need 기반 조정)
        self.d_model = 256          # 효율성을 위해 256
        self.num_slots = 32         # d_model/8
        self.bilinear_rank = 16     # d_model/16
        self.max_reasoning_steps = 4
        self.num_decoder_layers = 4  # auto_balance()에서 조정됨
        self.num_heads = 8          # 원논문과 동일
        self.convergence_threshold = 0.01
        
        # 토크나이저 (기존 유지)
        self.tokenizer_name = "google-t5/t5-base"
        self.vocab_size = None
        self.pad_token_id = None
        
        # 시퀀스 길이 (기존보다 증가)
        self.max_seq_len = 256      # 기존 128 → 256
        self.answer_max_length = 64 # 기존 32 → 64
        
        # 훈련 설정 (Attention Is All You Need 기반)
        self.learning_rate = 1e-4
        self.batch_size = 32        # 기존 8 → 32
        self.num_epochs = 3
        self.dropout = 0.1          # 기존 0.3 → 0.1 (원논문)
        self.weight_decay = 0.01    # 기존 0.1 → 0.01
        
        # 정규화 (원논문 기반)
        self.orthogonal_weight = 0.01   # 기존 0.1 → 0.01
        self.label_smoothing = 0.1      # 원논문과 동일
        self.gradient_clip = 1.0
        
        # 기타 (기존 유지)
        self.bf16 = True
        self.early_stopping_patience = 3
        self.eval_every = 100       # 기존 50 → 100
        
        # 데이터셋별 설정 (기존 유지)
        self.dataset_name = "unknown"
        self.task_prefix = "answer"

    def set_size(self, size):
        """모델 크기 설정 (기존 메서드 + 값 개선)"""
        sizes = {
            "micro": {
                "d_model": 128, "num_slots": 16, "bilinear_rank": 8,
                "max_reasoning_steps": 3, "num_decoder_layers": 3, "num_heads": 4,
                "max_seq_len": 128, "batch_size": 64, "learning_rate": 3e-4
            },
            "small": {
                "d_model": 192, "num_slots": 24, "bilinear_rank": 12,
                "max_reasoning_steps": 3, "num_decoder_layers": 4, "num_heads": 6,
                "max_seq_len": 256, "batch_size": 48, "learning_rate": 2e-4
            },
            "base": {
                "d_model": 256, "num_slots": 32, "bilinear_rank": 16,
                "max_reasoning_steps": 4, "num_decoder_layers": 4, "num_heads": 8,
                "max_seq_len": 256, "batch_size": 32, "learning_rate": 1e-4
            },
            "large": {
                "d_model": 384, "num_slots": 48, "bilinear_rank": 24,
                "max_reasoning_steps": 5, "num_decoder_layers": 6, "num_heads": 12,
                "max_seq_len": 384, "batch_size": 24, "learning_rate": 5e-5
            }
        }
        
        if size in sizes:
            for key, value in sizes[size].items():
                setattr(self, key, value)
        
        return self

    def set_dataset(self, dataset_name, **kwargs):
        """데이터셋별 설정 (기존 메서드 + 데이터셋 특성 반영)"""
        self.dataset_name = dataset_name
        
        # 데이터셋별 최적화된 기본값
        dataset_defaults = {
            "strategyqa": {
                "task_prefix": "strategy", 
                "answer_max_length": 8, 
                "num_epochs": 5,           # 작은 데이터셋(2.7K)
                "batch_size": 16,
                "learning_rate": 2e-4      # 작은 데이터셋엔 높은 lr
            },
            "logiqa": {
                "task_prefix": "reason", 
                "answer_max_length": 16, 
                "num_epochs": 3,
                "max_seq_len": 384,        # 논리 추론엔 긴 컨텍스트
                "learning_rate": 1.5e-4
            },
            "gsm8k": {
                "task_prefix": "solve", 
                "answer_max_length": 128,   # 수학 풀이는 길어질 수 있음
                "num_epochs": 3,
                "max_seq_len": 512,        # 수학 문제 설명이 길 수 있음
                "max_reasoning_steps": 6   # 수학은 더 많은 추론 단계
            },
            "multinli": {
                "task_prefix": "infer", 
                "answer_max_length": 16, 
                "num_epochs": 2,           # 큰 데이터셋(433K)
                "batch_size": 64,
                "learning_rate": 5e-5,     # 큰 데이터셋엔 낮은 lr
                "early_stopping_patience": 2
            }
        }
        
        # 기본값 적용
        if dataset_name in dataset_defaults:
            for key, value in dataset_defaults[dataset_name].items():
                setattr(self, key, value)
        
        # 사용자 정의 설정 적용 (기존 로직 유지)
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        return self

    def update(self, **kwargs):
        """설정 업데이트 (기존 메서드 유지)"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def auto_balance(self):
        """Connection과 Baseline 파라미터 자동 균형화 🔥"""
        vocab_size = self.vocab_size or 32000
        
        # Connection Transformer 파라미터 추정
        conn_bilinear = 2 * self.num_slots**2 * self.d_model * self.bilinear_rank
        conn_other = 6 * self.d_model**2 + 2 * vocab_size * self.d_model
        conn_total = conn_bilinear + conn_other
        
        # Baseline Transformer 파라미터 추정  
        baseline_layers = self.num_decoder_layers * 2  # encoder + decoder
        baseline_attn = baseline_layers * 4 * self.d_model**2
        baseline_ffn = baseline_layers * 2 * self.d_model * (self.d_model * 4)
        baseline_other = 2 * vocab_size * self.d_model
        baseline_total = baseline_attn + baseline_ffn + baseline_other
        
        # 파라미터 차이가 10% 이상이면 조정
        diff_ratio = abs(conn_total - baseline_total) / max(conn_total, baseline_total)
        
        if diff_ratio > 0.1:
            if baseline_total > conn_total:
                # Baseline이 크면 레이어 줄이기
                target_ratio = conn_total / baseline_total
                self.num_decoder_layers = max(2, int(self.num_decoder_layers * target_ratio**0.5))
            else:
                # Connection이 크면 bilinear_rank 줄이기
                target_ratio = baseline_total / conn_total
                self.bilinear_rank = max(4, int(self.bilinear_rank * target_ratio**0.5))
        
        return self