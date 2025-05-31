# configs/base_config.py
import math

class BaseConfig:
    """RTX 4090 최적화 기본 설정 클래스"""
    
    # 🚀 RTX 4090 최적화 모델 아키텍처
    d_model = 256           # 유지 (적당한 크기)
    num_slots = 64          # 128 → 64 (4배 메모리 절약)
    bilinear_rank = 16      # 32 → 16 (4배 메모리 절약)
    max_reasoning_steps = 4 # 6 → 4 (빠른 수렴)
    convergence_threshold = 0.01
    
    # 🎯 훈련 설정 최적화
    learning_rate = 1e-4
    weight_decay = 0.01
    num_epochs = 12         # 20 → 12 (빠른 실험)
    warmup_ratio = 0.1
    batch_size = 16         # 32 → 16 (메모리 절약)
    gradient_clip = 1.0
    reasoning_cost_weight = 0.001
    gradient_accumulation_steps = 2  # 실질적 batch_size = 32
    
    # 토크나이저 설정
    tokenizer_name = "t5-base"
    max_seq_len = 256       # 512 → 256 (메모리 절약)
    
    # T5 특화 설정
    decoder_start_token_id = 0
    
    # 데이터셋별 task prefix
    task_prefixes = {
        "logiqa": "reason",
        "gsm8k": "solve", 
        "strategyqa": "strategy"
    }
    
    # 🔧 RTX 4090 최적화 하드웨어 설정
    fp16 = True             # Mixed precision 필수
    gradient_checkpointing = True  # 메모리 절약
    num_workers = 8         # 4090의 높은 처리 속도 활용
    pin_memory = True
    empty_cache_every = 50  # 자주 캐시 정리
    
    # 로깅 설정
    save_every = 500
    eval_every = 200        # 더 자주 평가
    log_every = 20          # 더 자주 로깅
    
    def update(self, **kwargs):
        """설정 값들을 업데이트"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: Unknown config parameter {k}")
        return self
    
    def to_dict(self):
        """설정을 딕셔너리로 변환"""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v)}
    
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
        bilinear_params = 2 * N * N * D * r  # W_source + W_target
        cross_attn_params = 6 * D * D        # 6개 projection matrices
        embedding_params = (V + S) * D       # token + pos embeddings
        output_params = D * V                # output projection
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
        """모델 정보 출력"""
        params = self.get_estimated_params()
        print(f"\n📊 Model Configuration:")
        print(f"   d_model: {self.d_model}")
        print(f"   num_slots: {self.num_slots}")
        print(f"   bilinear_rank: {self.bilinear_rank}")
        print(f"   max_reasoning_steps: {self.max_reasoning_steps}")
        print(f"\n🔢 Estimated Parameters:")
        print(f"   Bilinear connections: {params['bilinear']:,}")
        print(f"   Cross-attention: {params['cross_attention']:,}")
        print(f"   Embeddings: {params['embeddings']:,}")
        print(f"   Output projection: {params['output']:,}")
        print(f"   Layer norms: {params['layer_norms']:,}")
        print(f"   Total: {params['total']:,} ({params['total']/1e6:.1f}M)")
        
        # 메모리 예상
        memory_gb = params['total'] * 4 / 1e9 * 3  # FP32 기준 x3 (grads + optimizer)
        print(f"\n💾 Estimated GPU Memory: ~{memory_gb:.1f}GB")
