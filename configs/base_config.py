# configs/base_config.py
class BaseConfig:
    """깔끔한 기본 설정"""
    
    def __init__(self):
        # 모델 아키텍처
        self.d_model = 64
        self.num_slots = 16
        self.bilinear_rank = 4
        self.max_reasoning_steps = 2
        self.num_decoder_layers = 3
        self.num_heads = 4
        self.convergence_threshold = 0.1
        
        # 토크나이저 (실행 시 설정됨)
        self.tokenizer_name = "google-t5/t5-base"
        self.vocab_size = None
        self.pad_token_id = None
        
        # 시퀀스 길이
        self.max_seq_len = 128
        self.answer_max_length = 32
        
        # 훈련 설정
        self.learning_rate = 1e-4
        self.batch_size = 8
        self.num_epochs = 3
        self.dropout = 0.3
        self.weight_decay = 0.1
        
        # 정규화
        self.orthogonal_weight = 0.1
        self.label_smoothing = 0.1
        self.gradient_clip = 1.0
        
        # 기타
        self.bf16 = True
        self.early_stopping_patience = 3
        self.eval_every = 50
        
        # 데이터셋별 설정
        self.dataset_name = "unknown"
        self.task_prefix = "answer"
    
    def set_size(self, size):
        """모델 크기 설정"""
        sizes = {
            "micro": {
                "d_model": 128, "num_slots": 32, "bilinear_rank": 8,
                "max_reasoning_steps": 3, "num_decoder_layers": 4, "num_heads": 8,
                "max_seq_len": 256, "batch_size": 12, "learning_rate": 2e-4
            },
            "small": {
                "d_model": 192, "num_slots": 48, "bilinear_rank": 12,
                "max_reasoning_steps": 3, "num_decoder_layers": 5, "num_heads": 6,
                "max_seq_len": 384, "batch_size": 14, "learning_rate": 2.5e-4
            },
            "base": {
                "d_model": 256, "num_slots": 64, "bilinear_rank": 16,
                "max_reasoning_steps": 4, "num_decoder_layers": 6, "num_heads": 8,
                "max_seq_len": 512, "batch_size": 16, "learning_rate": 3e-4
            },
            "large": {
                "d_model": 512, "num_slots": 128, "bilinear_rank": 32,
                "max_reasoning_steps": 5, "num_decoder_layers": 8, "num_heads": 16,
                "max_seq_len": 1024, "batch_size": 32, "learning_rate": 4e-4
            }
        }
        
        if size in sizes:
            for key, value in sizes[size].items():
                setattr(self, key, value)
        
        return self
    
    def set_dataset(self, dataset_name, **kwargs):
        """데이터셋별 설정"""
        self.dataset_name = dataset_name
        
        # 데이터셋별 기본값
        dataset_defaults = {
            "strategyqa": {"task_prefix": "strategy", "answer_max_length": 8, "num_epochs": 2},
            "logiqa": {"task_prefix": "reason", "answer_max_length": 16, "num_epochs": 3},
            "gsm8k": {"task_prefix": "solve", "answer_max_length": 32, "num_epochs": 3},
            "multinli": {"task_prefix": "infer", "answer_max_length": 16, "num_epochs": 5}
        }
        
        # 기본값 적용
        if dataset_name in dataset_defaults:
            for key, value in dataset_defaults[dataset_name].items():
                setattr(self, key, value)
        
        # 사용자 정의 설정 적용
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        return self
    
    def update(self, **kwargs):
        """설정 업데이트"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self