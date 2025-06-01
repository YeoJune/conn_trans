# configs/base_config.py

import math

class BaseConfig:
    """T5에 최적화된 통합 기본 설정"""
    
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
    convergence_threshold = 0.1
    
    # T5 최적화된 훈련 설정
    learning_rate = 1e-4        # T5는 더 높은 LR 필요 (HF 문서 권장)
    weight_decay = 0.1          
    num_epochs = 3              
    warmup_ratio = 0.1
    batch_size = 8              
    gradient_accumulation_steps = 8  
    gradient_clip = 1.0         # T5는 1.0 권장
    
    # 강력한 정규화 (오버피팅 방지)
    dropout = 0.3               
    orthogonal_weight = 0.1     
    reasoning_cost_weight = 0.01
    label_smoothing = 0.1       # T5는 label smoothing 효과적
    
    # 조기 종료 (필수)
    early_stopping_patience = 3  
    eval_every = 50             
    
    # T5 토크나이저 설정
    tokenizer_name = "google-t5/t5-base"  # 최신 T5 모델명
    max_seq_len = 128           
    
    # T5 특화 설정
    decoder_start_token_id = 0
    
    # 🔥 T5 중요 설정 (정확도 0 문제 해결)
    fp16 = False                # T5는 fp16에서 문제 발생, bf16 또는 fp32 사용
    bf16 = True                 # 가능한 경우 bfloat16 사용
    dataloader_pin_memory = True
    
    # 데이터셋별 task prefix
    task_prefixes = {
        "logiqa": "reason",
        "gsm8k": "solve", 
        "strategyqa": "strategy",
        "multinli": "infer"
    }
    
    # Answer 길이 설정
    answer_max_length = 32      # 답변 최대 길이
    
    # 하드웨어 설정 (메모리 절약)
    gradient_checkpointing = True
    num_workers = 2             # 데이터 로더 워커 수 줄임
    pin_memory = True
    empty_cache_every = 25      
    
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
        
        # 모델 크기에 따른 T5 최적화 조정
        if size == "nano":
            # 극소형 - 최대 정규화
            self.learning_rate = 5e-5   # T5 nano는 더 작은 LR
            self.dropout = 0.5
            self.weight_decay = 0.3
            self.num_epochs = 2
            self.batch_size = 4
            self.gradient_accumulation_steps = 16
            self.bf16 = False           # nano는 fp32 안전
            self.fp16 = False
        elif size == "micro":
            # 초소형 - 강한 정규화
            self.learning_rate = 1e-4   # T5 권장 범위
            self.dropout = 0.3
            self.weight_decay = 0.1
            self.num_epochs = 3
            self.batch_size = 8
            self.gradient_accumulation_steps = 8
            self.bf16 = True
        elif size in ["tiny", "small", "base"]:
            # 더 큰 모델 - 조금 완화하지만 여전히 강함
            self.learning_rate = 3e-4   # T5 큰 모델 권장
            self.dropout = 0.2
            self.weight_decay = 0.05
            self.num_epochs = 5
            self.batch_size = 12
            self.gradient_accumulation_steps = 6
            self.bf16 = True
    
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
        """모델 정보 및 T5 최적화 설정 출력"""
        params = self.get_estimated_params()
        
        print(f"\n📊 Model Configuration (T5 Optimized):")
        print(f"   Architecture: d_model={self.d_model}, num_slots={self.num_slots}, bilinear_rank={self.bilinear_rank}")
        print(f"   Reasoning steps: {self.max_reasoning_steps}")
        print(f"   Total parameters: {params['total']:,} ({params['total']/1e6:.1f}M)")
        
        # T5 특화 설정 출력
        print(f"\n🔧 T5 Training Settings:")
        print(f"   Learning rate: {self.learning_rate} (T5 optimized)")
        print(f"   Precision: {'bf16' if self.bf16 else 'fp16' if self.fp16 else 'fp32'}")
        print(f"   Batch size: {self.batch_size} (effective: {self.batch_size * self.gradient_accumulation_steps})")
        print(f"   Gradient clip: {self.gradient_clip}")
        print(f"   Label smoothing: {self.label_smoothing}")
        
        # 메모리 예상
        memory_gb = params['total'] * 4 / 1e9 * 2  # parameters + gradients
        if self.bf16:
            memory_gb *= 0.5  # bfloat16 절약
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
            recommendation = "강한 정규화 필요 (현재 설정 적절)"
        elif examples_per_param >= 0.001:
            risk_level = "🚨 HIGH RISK" 
            recommendation = "더 작은 모델 사용 권장"
        else:
            risk_level = "💀 EXTREME RISK"
            recommendation = "nano 모델 필수"
        
        print(f"   Risk level: {risk_level}")
        print(f"   Recommendation: {recommendation}")
        
        # T5 특화 권장사항
        if examples_per_param < 0.01:
            print(f"   T5 Tip: fp16=False, bf16=False로 설정 권장")
        
        return examples_per_param