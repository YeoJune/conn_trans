# configs/base_config.py
import math

class BaseConfig:
    """기본 설정 클래스 - 모든 실험의 베이스"""
    
    # 모델 아키텍처
    d_model = 256
    num_slots = 128
    bilinear_rank = 32
    max_reasoning_steps = 6
    convergence_threshold = 0.01
    
    # 훈련 설정
    learning_rate = 1e-4
    weight_decay = 0.01
    num_epochs = 20
    warmup_ratio = 0.1
    batch_size = 32
    gradient_clip = 1.0
    reasoning_cost_weight = 0.001
    
    # 토크나이저 설정 (T5 고정)
    tokenizer_name = "t5-base"  # T5 전용
    max_seq_len = 512
    
    # T5 특화 설정
    decoder_start_token_id = 0  # T5의 pad_token_id
    
    # 데이터셋별 task prefix (T5의 핵심 특징)
    task_prefixes = {
        "logiqa": "reason",
        "gsm8k": "solve", 
        "strategyqa": "strategy"
    }
    
    # 하드웨어 설정
    fp16 = True
    gradient_checkpointing = True
    num_workers = 4
    pin_memory = True
    
    # 로깅 설정
    save_every = 1000
    eval_every = 500
    log_every = 100
    
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
    
    def calculate_baseline_config(self):
        """매칭되는 baseline transformer 설정 계산"""
        # Connection Transformer 파라미터 계산
        N = self.num_slots
        D = self.d_model
        r = self.bilinear_rank
        V = getattr(self, 'vocab_size', 32000)
        S = self.max_seq_len
        
        # Connection Transformer 파라미터 수
        bilinear_params = 2 * N * N * D * r
        cross_attn_params = 6 * D * D
        embedding_params = (V + S) * D
        output_params = D * V
        layer_norm_params = self.max_reasoning_steps * 2 * D  # LayerNorm has weight + bias
        
        conn_total = bilinear_params + cross_attn_params + embedding_params + output_params + layer_norm_params
        
        print(f"\nConnection Transformer parameters:")
        print(f"  Bilinear: {bilinear_params:,}")
        print(f"  Cross-attention: {cross_attn_params:,}")  
        print(f"  Embeddings: {embedding_params:,}")
        print(f"  Output: {output_params:,}")
        print(f"  LayerNorms: {layer_norm_params:,}")
        print(f"  Total: {conn_total:,}")
        
        # Baseline transformer - 공유 파라미터
        baseline_shared = embedding_params + output_params + 2 * D  # output LayerNorm
        available_for_layers = conn_total - baseline_shared
        
        # 최적 layer 수와 FFN 배수 찾기
        best_config = None
        best_diff = float('inf')
        
        for ffn_mult in [2, 3, 4, 6, 8]:
            ffn_dim = D * ffn_mult
            
            # Attention parameters
            attn_params = 4 * D * D  # q, k, v, out projections
            
            # FFN parameters  
            ffn_params = D * ffn_dim + ffn_dim * D + ffn_dim + D  # linear1 + linear2 + biases
            
            # LayerNorm parameters
            ln_params = 4 * D  # 2 LayerNorms * (weight + bias)
            
            params_per_layer = attn_params + ffn_params + ln_params
            
            num_layers = max(1, available_for_layers // params_per_layer)
            actual_layer_params = num_layers * params_per_layer
            total_baseline = baseline_shared + actual_layer_params
            
            diff = abs(total_baseline - conn_total)
            if diff < best_diff:
                best_diff = diff
                best_config = {
                    'num_layers': int(num_layers),
                    'ffn_multiplier': ffn_mult,
                    'total_params': total_baseline,
                    'param_diff': diff
                }
        
        print(f"\nMatched Baseline Transformer:")
        print(f"  Layers: {best_config['num_layers']}")
        print(f"  FFN multiplier: {best_config['ffn_multiplier']}")
        print(f"  Total params: {best_config['total_params']:,}")
        print(f"  Difference: {best_config['param_diff']:,} ({best_config['param_diff']/conn_total*100:.1f}%)")
        
        return best_config