# configs/base_config.py

import math

class BaseConfig:
    """데이터셋 크기에 최적화된 통합 기본 설정"""
    
    # 기본 데이터셋 이름
    dataset_name = "unknown"
    
    # 통일된 모델 사이즈 아키텍처
    MODEL_ARCHITECTURES = {
        "nano": {
            "d_model": 32,
            "num_slots": 8,
            "bilinear_rank": 2,
            "max_reasoning_steps": 1,
            "description": "극소형 - StrategyQA 전용"
        },
        "micro": {
            "d_model": 64, 
            "num_slots": 16,
            "bilinear_rank": 4,
            "max_reasoning_steps": 2,
            "description": "초소형 - 모든 데이터셋 안전"
        },
        "tiny": {
            "d_model": 128,
            "num_slots": 32,
            "bilinear_rank": 8,
            "max_reasoning_steps": 3,
            "description": "소형 - LogiQA/GSM8K 권장"
        },
        "small": {
            "d_model": 192,
            "num_slots": 48,
            "bilinear_rank": 12,
            "max_reasoning_steps": 4,
            "description": "중소형 - 실험용 (위험)"
        },
        "base": {
            "d_model": 256,
            "num_slots": 64,
            "bilinear_rank": 16,
            "max_reasoning_steps": 4,
            "description": "기본형 - 오버피팅 위험 높음"
        }
    }
    
    # 기본 아키텍처 (micro - 가장 안전)
    d_model = MODEL_ARCHITECTURES["micro"]["d_model"]
    num_slots = MODEL_ARCHITECTURES["micro"]["num_slots"]
    bilinear_rank = MODEL_ARCHITECTURES["micro"]["bilinear_rank"]
    max_reasoning_steps = MODEL_ARCHITECTURES["micro"]["max_reasoning_steps"]
    convergence_threshold = 0.01
    
    # 데이터셋 크기별 최적화된 훈련 설정
    learning_rate = 3e-5        # 매우 작은 LR (오버피팅 방지)
    weight_decay = 0.1          # 강한 weight decay
    num_epochs = 3              # 매우 적은 에폭
    warmup_ratio = 0.1
    batch_size = 8              # 작은 배치
    gradient_accumulation_steps = 8  # 실질적 batch = 64
    gradient_clip = 0.5         # 강한 gradient clipping
    
    # 강력한 정규화 (오버피팅 방지)
    dropout = 0.3               # 강한 dropout
    orthogonal_weight = 0.1     # 강한 orthogonal regularization
    reasoning_cost_weight = 0.01
    label_smoothing = 0.2       # 강한 label smoothing
    
    # 조기 종료 (필수)
    early_stopping_patience = 3  # 매우 짧은 patience
    eval_every = 50             # 자주 평가
    
    # 토크나이저 설정
    tokenizer_name = "t5-base"
    max_seq_len = 128           # 짧은 시퀀스 (메모리 절약)
    
    # T5 특화 설정
    decoder_start_token_id = 0
    
    # 데이터셋별 task prefix
    task_prefixes = {
        "logiqa": "reason",
        "gsm8k": "solve", 
        "strategyqa": "strategy"
    }
    
    # 하드웨어 설정 (메모리 절약)
    fp16 = True
    gradient_checkpointing = True
    num_workers = 4
    pin_memory = True
    empty_cache_every = 25      # 자주 캐시 정리
    
    # 로깅 설정
    save_every = 200
    log_every = 10
    
    def set_model_size(self, size="micro"):
        """모델 크기 설정"""
        if size not in self.MODEL_ARCHITECTURES:
            available = list(self.MODEL_ARCHITECTURES.keys())
            raise ValueError(f"Unknown model size: {size}. Available: {available}")
        
        arch = self.MODEL_ARCHITECTURES[size]
        self.d_model = arch["d_model"]
        self.num_slots = arch["num_slots"] 
        self.bilinear_rank = arch["bilinear_rank"]
        self.max_reasoning_steps = arch["max_reasoning_steps"]
        
        # 모델 크기에 따른 추가 조정
        if size == "nano":
            # 극소형 - 최대 정규화
            self.learning_rate = 1e-5
            self.dropout = 0.5
            self.weight_decay = 0.3
            self.num_epochs = 2
            self.batch_size = 4
            self.gradient_accumulation_steps = 16
        elif size == "micro":
            # 초소형 - 강한 정규화
            self.learning_rate = 3e-5
            self.dropout = 0.3
            self.weight_decay = 0.1
            self.num_epochs = 3
            self.batch_size = 8
            self.gradient_accumulation_steps = 8
        elif size in ["tiny", "small", "base"]:
            # 더 큰 모델 - 조금 완화하지만 여전히 강함
            self.learning_rate = 5e-5
            self.dropout = 0.2
            self.weight_decay = 0.05
            self.num_epochs = 5
            self.batch_size = 12
            self.gradient_accumulation_steps = 6
    
    def update(self, **kwargs):
        """설정 값들을 업데이트"""
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self
    
    def to_dict(self):
        """설정을 딕셔너리로 변환"""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v) and k != 'MODEL_ARCHITECTURES'}
    
    def get_task_prefix(self, dataset_name):
        """데이터셋에 따른 task prefix 반환"""
        return self.task_prefixes.get(dataset_name, "answer")
    
    def get_estimated_params(self):
        """예상 파라미터 수 계산"""
        N = self.num_slots
        D = self.d_model
        r = self.bilinear_rank
        V = 32128  # T5-base vocab size
        S = self.max_seq_len
        
        # Connection Transformer 파라미터
        bilinear_params = 2 * N * N * D * r
        cross_attn_params = 6 * D * D
        embedding_params = (V + S) * D
        output_params = D * V
        layer_norm_params = self.max_reasoning_steps * 2 * D
        
        total = bilinear_params + cross_attn_params + embedding_params + output_params + layer_norm_params
        
        return {
            'bilinear': bilinear_params,
            'cross_attention': cross_attn_params,
            'embeddings': embedding_params,
            'output': output_params,
            'layer_norms': layer_norm_params,
            'total': total
        }
    
    def print_model_info(self):
        """모델 정보 및 오버피팅 위험도 출력"""
        params = self.get_estimated_params()
        
        print(f"\n📊 Model Configuration:")
        print(f"   Architecture: d_model={self.d_model}, num_slots={self.num_slots}, bilinear_rank={self.bilinear_rank}")
        print(f"   Reasoning steps: {self.max_reasoning_steps}")
        print(f"   Total parameters: {params['total']:,} ({params['total']/1e6:.1f}M)")
        
        # 메모리 예상
        memory_gb = params['total'] * 4 / 1e9 * 2  # FP16 + gradients
        print(f"   Estimated GPU memory: ~{memory_gb:.1f}GB")
    
    def analyze_overfitting_risk(self, dataset_size):
        """오버피팅 위험도 분석"""
        params = self.get_estimated_params()['total']
        examples_per_param = dataset_size / params
        
        print(f"\n⚠️ Overfitting Risk Analysis:")
        print(f"   Dataset size: {dataset_size:,}")
        print(f"   Examples per parameter: {examples_per_param:.6f}")
        
        if examples_per_param >= 0.1:
            risk_level = "✅ LOW RISK"
            recommendation = "안전하게 사용 가능"
        elif examples_per_param >= 0.01:
            risk_level = "⚠️ MODERATE RISK"
            recommendation = "강한 정규화 필요"
        elif examples_per_param >= 0.001:
            risk_level = "🚨 HIGH RISK" 
            recommendation = "더 작은 모델 사용 권장"
        else:
            risk_level = "💀 EXTREME RISK"
            recommendation = "nano/micro 모델 필수"
        
        print(f"   Risk level: {risk_level}")
        print(f"   Recommendation: {recommendation}")
        
        return examples_per_param
