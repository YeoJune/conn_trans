# configs/base_config.py
import math

class BaseConfig:
    """T5 Encoder-Decoder 모델에 최적화된 통합 기본 설정"""
    
    # 기본 데이터셋 이름
    dataset_name = "unknown"
    
    # 통일된 모델 사이즈 아키텍처 (Encoder-Decoder용)
    MODEL_ARCHITECTURES = {
        "nano": {
            "d_model": 32,
            "num_slots": 8,
            "bilinear_rank": 2,
            "max_reasoning_steps": 1,
            "num_decoder_layers": 2,
            "num_heads": 2,
            "description": "극소형 - StrategyQA 전용"
        },
        "micro": {
            "d_model": 64, 
            "num_slots": 16,
            "bilinear_rank": 4,
            "max_reasoning_steps": 2,
            "num_decoder_layers": 3,
            "num_heads": 4,
            "description": "초소형 - 모든 데이터셋 안전"
        },
        "tiny": {
            "d_model": 128,
            "num_slots": 32,
            "bilinear_rank": 8,
            "max_reasoning_steps": 3,
            "num_decoder_layers": 4,
            "num_heads": 8,
            "description": "소형 - LogiQA/GSM8K 권장"
        },
        "small": {
            "d_model": 192,
            "num_slots": 48,
            "bilinear_rank": 12,
            "max_reasoning_steps": 4,
            "num_decoder_layers": 6,
            "num_heads": 8,
            "description": "중소형 - 실험용"
        },
        "base": {
            "d_model": 256,
            "num_slots": 64,
            "bilinear_rank": 16,
            "max_reasoning_steps": 4,
            "num_decoder_layers": 6,
            "num_heads": 8,
            "description": "기본형 - 큰 데이터셋용"
        }
    }
    
    # 기본 아키텍처 (micro - 가장 안전)
    d_model = MODEL_ARCHITECTURES["micro"]["d_model"]
    num_slots = MODEL_ARCHITECTURES["micro"]["num_slots"]
    bilinear_rank = MODEL_ARCHITECTURES["micro"]["bilinear_rank"]
    max_reasoning_steps = MODEL_ARCHITECTURES["micro"]["max_reasoning_steps"]
    num_decoder_layers = MODEL_ARCHITECTURES["micro"]["num_decoder_layers"]
    num_heads = MODEL_ARCHITECTURES["micro"]["num_heads"]
    convergence_threshold = 0.1
    
    # Encoder-Decoder Vocabulary 설정 (토크나이저에서 자동 설정됨)
    src_vocab_size = None  # Will be set by tokenizer
    tgt_vocab_size = None  # Will be set by tokenizer
    src_pad_token_id = None  # Will be set by tokenizer
    tgt_pad_token_id = None  # Will be set by tokenizer
    vocab_size = None  # Compatibility - will be set by tokenizer
    
    # T5 최적화된 훈련 설정
    learning_rate = 1e-4        
    weight_decay = 0.1          
    num_epochs = 3              
    warmup_ratio = 0.1
    batch_size = 8              
    gradient_accumulation_steps = 8  
    gradient_clip = 1.0         
    
    # 강력한 정규화 (오버피팅 방지)
    dropout = 0.3               
    orthogonal_weight = 0.1     
    reasoning_cost_weight = 0.01
    label_smoothing = 0.1       
    
    # 조기 종료 (필수)
    early_stopping_patience = 3  
    eval_every = 50             
    
    # T5 토크나이저 설정
    tokenizer_name = "google-t5/t5-base"
    max_seq_len = 128           
    
    # Encoder-Decoder 특화 설정
    decoder_start_token_id = 0
    max_target_length = 32      # Target sequence 최대 길이
    
    # 🔥 T5 중요 설정
    fp16 = False                
    bf16 = True                 
    dataloader_pin_memory = True
    
    # 데이터셋별 task prefix
    task_prefixes = {
        "logiqa": "reason",
        "gsm8k": "solve", 
        "strategyqa": "strategy",
        "multinli": "infer"
    }
    
    # Answer 길이 설정 (Encoder-Decoder용)
    answer_max_length = 32      
    
    # 하드웨어 설정
    gradient_checkpointing = True
    num_workers = 2             
    pin_memory = True
    empty_cache_every = 25      
    
    # 로깅 설정
    save_every = 200
    log_every = 10
    
    def set_model_size(self, size="micro"):
        """Encoder-Decoder 모델 크기 설정"""
        if size not in self.MODEL_ARCHITECTURES:
            available = list(self.MODEL_ARCHITECTURES.keys())
            raise ValueError(f"Unknown model size: {size}. Available: {available}")
        
        arch = self.MODEL_ARCHITECTURES[size]
        self.d_model = arch["d_model"]
        self.num_slots = arch["num_slots"] 
        self.bilinear_rank = arch["bilinear_rank"]
        self.max_reasoning_steps = arch["max_reasoning_steps"]
        self.num_decoder_layers = arch["num_decoder_layers"]
        self.num_heads = arch["num_heads"]
        
        # 모델 크기에 따른 최적화 조정
        if size == "nano":
            self.learning_rate = 5e-5   
            self.dropout = 0.5
            self.weight_decay = 0.3
            self.num_epochs = 2
            self.batch_size = 4
            self.gradient_accumulation_steps = 16
            self.bf16 = False           
            self.fp16 = False
            self.max_seq_len = 96
            self.answer_max_length = 16
        elif size == "micro":
            self.learning_rate = 1e-4   
            self.dropout = 0.3
            self.weight_decay = 0.1
            self.num_epochs = 3
            self.batch_size = 8
            self.gradient_accumulation_steps = 8
            self.bf16 = True
            self.max_seq_len = 128
            self.answer_max_length = 32
        elif size == "tiny":
            self.learning_rate = 2e-4   
            self.dropout = 0.25
            self.weight_decay = 0.08
            self.num_epochs = 4
            self.batch_size = 12
            self.gradient_accumulation_steps = 6
            self.bf16 = True
            self.max_seq_len = 256
            self.answer_max_length = 48
        elif size in ["small", "base"]:
            self.learning_rate = 3e-4   
            self.dropout = 0.2
            self.weight_decay = 0.05
            self.num_epochs = 5
            self.batch_size = 16
            self.gradient_accumulation_steps = 4
            self.bf16 = True
            self.max_seq_len = 384 if size == "base" else 256
            self.answer_max_length = 64 if size == "base" else 48
    
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
        """Encoder-Decoder 모델 예상 파라미터 수 계산"""
        N = self.num_slots
        D = self.d_model
        r = self.bilinear_rank
        V_src = getattr(self, 'src_vocab_size', 32128)
        V_tgt = getattr(self, 'tgt_vocab_size', 32128)
        S = self.max_seq_len
        num_heads = self.num_heads
        ffn_mult = 4  # 표준 FFN multiplier
        
        # Encoder (Connection Transformer) 파라미터
        encoder_bilinear = 2 * N * N * D * r
        encoder_cross_attn = 3 * D * D  # W_q, W_k, W_v
        encoder_layer_norms = self.max_reasoning_steps * 2 * D
        
        # Embeddings
        src_embeddings = V_src * D + S * D
        tgt_embeddings = V_tgt * D + S * D
        
        # Decoder 파라미터
        decoder_params = 0
        for _ in range(self.num_decoder_layers):
            # Self-attention
            decoder_params += 4 * D * D  # q, k, v, out
            # Cross-attention  
            decoder_params += 4 * D * D  # q, k, v, out
            # FFN
            decoder_params += D * D * ffn_mult + D * ffn_mult * D + D * ffn_mult + D
            # LayerNorms (3개: self-attn, cross-attn, ffn)
            decoder_params += 6 * D
        
        # Output projection
        output_params = D * V_tgt
        output_norm = 2 * D
        
        encoder_total = encoder_bilinear + encoder_cross_attn + encoder_layer_norms
        embedding_total = src_embeddings + tgt_embeddings
        decoder_total = decoder_params
        output_total = output_params + output_norm
        
        total = encoder_total + embedding_total + decoder_total + output_total
        
        return {
            'encoder_bilinear': encoder_bilinear,
            'encoder_cross_attention': encoder_cross_attn,
            'encoder_layer_norms': encoder_layer_norms,
            'embeddings': embedding_total,
            'decoder': decoder_total,
            'output': output_total,
            'total': total
        }
    
    def print_model_info(self):
        """Encoder-Decoder 모델 정보 출력"""
        params = self.get_estimated_params()
        
        print(f"\n📊 Encoder-Decoder Model Configuration:")
        print(f"   Encoder: d_model={self.d_model}, num_slots={self.num_slots}, bilinear_rank={self.bilinear_rank}")
        print(f"   Decoder: {self.num_decoder_layers} layers, {self.num_heads} heads")
        print(f"   Reasoning steps: {self.max_reasoning_steps}")
        print(f"   Total parameters: {params['total']:,} ({params['total']/1e6:.1f}M)")
        
        print(f"\n🏗️ Architecture Breakdown:")
        print(f"   Encoder (slots + reasoning): {params['encoder_bilinear'] + params['encoder_cross_attention']:,}")
        print(f"   Decoder (standard transformer): {params['decoder']:,}")
        print(f"   Embeddings (src + tgt): {params['embeddings']:,}")
        print(f"   Output projection: {params['output']:,}")
        
        print(f"\n🔧 T5 Training Settings:")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Precision: {'bf16' if self.bf16 else 'fp16' if self.fp16 else 'fp32'}")
        print(f"   Batch size: {self.batch_size} (effective: {self.batch_size * self.gradient_accumulation_steps})")
        print(f"   Max sequence lengths: src={self.max_seq_len}, tgt={self.answer_max_length}")
        
        # 메모리 예상
        memory_gb = params['total'] * 4 / 1e9 * 2  # parameters + gradients
        if self.bf16:
            memory_gb *= 0.5
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
            recommendation = "현재 정규화 설정 적절"
        elif examples_per_param >= 0.001:
            risk_level = "🚨 HIGH RISK" 
            recommendation = "더 작은 모델 또는 더 강한 정규화 필요"
        else:
            risk_level = "💀 EXTREME RISK"
            recommendation = "nano 모델 또는 데이터 증강 필수"
        
        print(f"   Risk level: {risk_level}")
        print(f"   Recommendation: {recommendation}")
        
        # Encoder-Decoder 특화 권장사항
        if examples_per_param < 0.01:
            print(f"   Encoder-Decoder Tip: decoder layer 수 줄이기 고려")
        
        return examples_per_param
    
    def get_compatible_baseline_config(self):
        """BaselineTransformer와 호환되는 설정 계산"""
        conn_params = self.get_estimated_params()['total']
        
        # Baseline에서 사용할 설정들 시도
        best_config = None
        best_diff = float('inf')
        
        for enc_layers in range(2, 8):
            for dec_layers in range(2, 8):
                for ffn_mult in [2, 3, 4]:
                    # Baseline 파라미터 계산
                    D = self.d_model
                    V_src = getattr(self, 'src_vocab_size', 32128)
                    V_tgt = getattr(self, 'tgt_vocab_size', 32128)
                    S = self.max_seq_len
                    
                    # Shared params
                    embeddings = (V_src + V_tgt + 2 * S) * D
                    output_proj = D * V_tgt + 2 * D
                    
                    # Encoder params
                    enc_layer_params = (
                        4 * D * D +  # attention
                        D * D * ffn_mult + D * ffn_mult * D + D * ffn_mult + D +  # ffn
                        4 * D  # layer norms
                    )
                    
                    # Decoder params  
                    dec_layer_params = (
                        4 * D * D +  # self-attention
                        4 * D * D +  # cross-attention
                        D * D * ffn_mult + D * ffn_mult * D + D * ffn_mult + D +  # ffn
                        6 * D  # layer norms
                    )
                    
                    baseline_total = embeddings + output_proj + enc_layers * enc_layer_params + dec_layers * dec_layer_params
                    
                    diff = abs(baseline_total - conn_params)
                    if diff < best_diff:
                        best_diff = diff
                        best_config = {
                            'num_encoder_layers': enc_layers,
                            'num_decoder_layers': dec_layers,
                            'ffn_multiplier': ffn_mult,
                            'total_params': baseline_total,
                            'param_diff': diff
                        }
        
        return best_config
