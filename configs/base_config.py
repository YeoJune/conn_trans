# configs/base_config.py
class BaseConfig:
    """
    Simple한 기본 설정
    """
    
    def __init__(self):
        # 모델 아키텍처
        self.d_model = 256
        self.num_slots = 32
        self.bilinear_rank = 16
        self.max_reasoning_steps = 4
        self.num_decoder_layers = 4  # auto_balance()에서 조정됨
        self.num_heads = 8
        self.convergence_threshold = 10
        
        # 토크나이저
        self.tokenizer_name = "google-t5/t5-base"
        self.vocab_size = None
        self.pad_token_id = None
        
        # 시퀀스 길이
        self.max_seq_len = 256
        self.answer_max_length = 64
        
        # 훈련 설정
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.num_epochs = 10
        self.dropout = 0.1
        self.weight_decay = 0.01
        
        # 정규화 (원논문 기반)
        self.orthogonal_weight = 0.01
        self.label_smoothing = 0.1
        self.gradient_clip = 1.0
        
        # 기타 (기존 유지)
        self.bf16 = True
        self.early_stopping_patience = 3
        self.eval_every = 100
        
        # 데이터셋별 설정 (기존 유지)
        self.dataset_name = "unknown"
        self.task_prefix = "answer"

    def set_size(self, size):
        """모델 크기 설정"""
        sizes = {
            "micro": {
                "d_model": 128, "num_slots": 128, "bilinear_rank": 1,
                "max_reasoning_steps": 1, "num_decoder_layers": 4, "num_heads": 4,
                "max_seq_len": 128, "batch_size": 64, "learning_rate": 2e-4
            },
            "small": {
                "d_model": 256, "num_slots": 256, "bilinear_rank": 1,
                "max_reasoning_steps": 1, "num_decoder_layers": 5, "num_heads": 4,
                "max_seq_len": 256, "batch_size": 48, "learning_rate": 2e-4
            },
            "base": {
                "d_model": 512, "num_slots": 512, "bilinear_rank": 1,
                "max_reasoning_steps": 1, "num_decoder_layers": 6, "num_heads": 8,
                "max_seq_len": 256, "batch_size": 32, "learning_rate": 1e-4
            },
            "large": {
                "d_model": 768, "num_slots": 768, "bilinear_rank": 1,
                "max_reasoning_steps": 1, "num_decoder_layers": 6, "num_heads": 8,
                "max_seq_len": 256, "batch_size": 24, "learning_rate": 1e-4
            }
        }
        
        if size in sizes:
            for key, value in sizes[size].items():
                setattr(self, key, value)
        
        return self

    def set_dataset(self, dataset_name, **kwargs):
        """데이터셋별 설정"""
        self.dataset_name = dataset_name
        
        # 데이터셋별 최적화된 기본값
        dataset_defaults = {
            "strategyqa": {
                "task_prefix": "strategy", 
                "answer_max_length": 8, 
                "num_epochs": 5,           # 작은 데이터셋(2.7K)
                "batch_size": 16,
                "learning_rate": 1e-4      # 작은 데이터셋엔 높은 lr
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
                "num_epochs": 15,
                "max_seq_len": 512,        # 수학 문제 설명이 길 수 있음
            },
            "multinli": {
                "task_prefix": "infer", 
                "answer_max_length": 16, 
                "num_epochs": 12,           # 큰 데이터셋(433K)
                "batch_size": 64,
                "learning_rate": 5e-5,     # 큰 데이터셋엔 낮은 lr
                "early_stopping_patience": 2
            },
            "eli5": {
                "task_prefix": "explain", 
                "answer_max_length": 200,   # 적절한 길이
                "max_seq_len": 320,
                "num_epochs": 6,
                "batch_size": 12,           # 메모리 고려
                "learning_rate": 8e-5       # 안정적인 학습률
            },
            "commongen": {
                "task_prefix": "connect", 
                "answer_max_length": 80,    # 간결한 생성
                "max_seq_len": 200,
                "num_epochs": 10,
                "batch_size": 32,
                "learning_rate": 1e-4
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
        """Connection과 Baseline 파라미터 자동 균형화"""
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