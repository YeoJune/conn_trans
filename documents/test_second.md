# RTX 4090 기준 Connection Transformer 실험 셋업

## 하드웨어 스펙

- **GPU**: RTX 4090 (24GB VRAM)
- **메모리**: 충분한 RAM (32GB+ 권장)
- **저장공간**: SSD 500GB+ (데이터셋 + 모델 체크포인트)

---

## 실험 설정

### 모델 Configuration (현재 구현 기준)

```python
# micro 설정 (빠른 테스트)
micro_config = {
    "d_model": 128,
    "num_slots": 128,
    "bilinear_rank": 1,
    "max_reasoning_steps": 1,
    "num_decoder_layers": 4,
    "num_heads": 4,
    "max_seq_len": 128,
    "batch_size": 64
}

# small 설정 (균형)
small_config = {
    "d_model": 256,
    "num_slots": 256,          # N = D baseline
    "bilinear_rank": 1,
    "max_reasoning_steps": 1,
    "num_decoder_layers": 5,
    "num_heads": 4,
    "max_seq_len": 256,
    "batch_size": 48
}

# base 설정 (성능 중심)
base_config = {
    "d_model": 512,
    "num_slots": 512,          # N = D for full capacity
    "bilinear_rank": 1,
    "max_reasoning_steps": 1,
    "num_decoder_layers": 6,
    "num_heads": 8,
    "max_seq_len": 256,
    "batch_size": 32
}

# large 설정 (최대 성능)
large_config = {
    "d_model": 768,
    "num_slots": 768,
    "bilinear_rank": 1,
    "max_reasoning_steps": 1,
    "num_decoder_layers": 6,
    "num_heads": 8,
    "max_seq_len": 256,
    "batch_size": 24
}
```

### 배치 크기 최적화 (Encoder-Decoder 구조)

```python
# RTX 4090 (24GB) 기준 권장 배치 크기
batch_sizes = {
    "micro": {
        "train_batch": 64,      # 메모리 사용량: ~12GB
        "eval_batch": 128,
        "gradient_accumulation": 1
    },
    "small": {
        "train_batch": 48,      # 메모리 사용량: ~16GB
        "eval_batch": 64,
        "gradient_accumulation": 2
    },
    "base": {
        "train_batch": 32,      # 메모리 사용량: ~20GB
        "eval_batch": 48,
        "gradient_accumulation": 2
    },
    "large": {
        "train_batch": 24,      # 메모리 사용량: ~22GB
        "eval_batch": 32,
        "gradient_accumulation": 3
    }
}
```

### 현재 구현 최적화 설정

```python
optimization_config = {
    # Mixed precision training (trainer.py 기준)
    "bf16": True,               # BF16 지원시 사용
    "fp16": False,              # fallback

    # Gradient 설정
    "gradient_clip": 1.0,
    "weight_decay": 0.01,

    # 정규화 (현재 구현)
    "orthogonal_weight": 0.01,  # Connection regularization
    "label_smoothing": 0.1,

    # DataLoader 설정
    "num_workers": 0,           # 현재 기본값
    "pin_memory": True,

    # 학습률 스케줄
    "warmup_ratio": 0.1,
}
```

---

## 데이터셋 구성 (현재 지원 데이터셋)

### Primary 데이터셋

```python
primary_datasets = {
    "strategyqa": {
        "task_prefix": "strategy",
        "answer_max_length": 8,
        "num_epochs": 5,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "description": "Yes/No reasoning (2.7K examples)"
    },
    "logiqa": {
        "task_prefix": "reason",
        "answer_max_length": 16,
        "num_epochs": 3,
        "max_seq_len": 384,
        "learning_rate": 1.5e-4,
        "description": "Multiple choice reasoning"
    },
    "gsm8k": {
        "task_prefix": "solve",
        "answer_max_length": 128,
        "num_epochs": 15,
        "max_seq_len": 512,
        "description": "Math word problems (8.8K examples)"
    }
}
```

### Secondary 데이터셋

```python
secondary_datasets = {
    "multinli": {
        "task_prefix": "infer",
        "answer_max_length": 16,
        "num_epochs": 12,
        "batch_size": 64,
        "learning_rate": 5e-5,
        "early_stopping_patience": 2,
        "description": "Natural language inference (433K examples)"
    },
    "eli5": {
        "task_prefix": "explain",
        "answer_max_length": 200,
        "max_seq_len": 320,
        "num_epochs": 6,
        "batch_size": 12,
        "learning_rate": 8e-5,
        "description": "Explain Like I'm 5"
    },
    "commongen": {
        "task_prefix": "connect",
        "answer_max_length": 80,
        "max_seq_len": 200,
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "description": "Concept-to-text generation"
    }
}
```

---

## 실험 계획

### Phase 1: 기본 검증 (현재 구현 테스트)

```python
baseline_experiments = [
    {
        "name": "Connection_Micro",
        "model": "connection",
        "size": "micro",
        "datasets": ["strategyqa", "logiqa"],
        "purpose": "빠른 기능 검증"
    },
    {
        "name": "Baseline_Micro",
        "model": "baseline",
        "size": "micro",
        "datasets": ["strategyqa", "logiqa"],
        "purpose": "대조군 성능 확인"
    },
    {
        "name": "Connection_Small",
        "model": "connection",
        "size": "small",
        "datasets": ["strategyqa", "logiqa", "gsm8k"],
        "purpose": "중간 크기 성능 확인"
    }
]
```

### Phase 2: 모델 크기별 비교

```python
size_comparison = [
    # Connection Transformer 스케일링
    {"model": "connection", "size": "micro", "focus": "efficiency"},
    {"model": "connection", "size": "small", "focus": "balance"},
    {"model": "connection", "size": "base", "focus": "performance"},

    # Baseline 비교
    {"model": "baseline", "size": "micro", "focus": "efficiency"},
    {"model": "baseline", "size": "small", "focus": "balance"},
    {"model": "baseline", "size": "base", "focus": "performance"}
]
```

### Phase 3: 데이터셋별 특성 분석

```python
dataset_analysis = {
    "reasoning_complexity": ["strategyqa", "logiqa"],
    "mathematical": ["gsm8k"],
    "large_scale": ["multinli"],
    "generation": ["eli5", "commongen"]
}
```

---

## 학습 설정 (현재 Trainer 클래스 기준)

### 훈련 파라미터

```python
training_config = {
    # 옵티마이저 (Trainer._setup_optimizer)
    "optimizer": "AdamW",
    "learning_rate": 1e-4,      # 데이터셋별 자동 조정
    "weight_decay": 0.01,

    # 스케줄러 (get_linear_schedule_with_warmup)
    "warmup_ratio": 0.1,

    # 정규화
    "gradient_clip": 1.0,
    "orthogonal_weight": 0.01,   # Connection specific
    "label_smoothing": 0.1,

    # 조기 종료
    "early_stopping_patience": 3,
    "eval_every": 100,

    # 정밀도
    "bf16": True,               # 자동 감지
    "fp16": False               # fallback
}
```

### 모니터링 메트릭 (현재 구현 기준)

```python
metrics_to_track = {
    "performance": {
        "accuracy": "calculate_accuracy from utils.metrics",
        "dataset_specific": "extract_final_answer per dataset type"
    },
    "efficiency": {
        "reasoning_steps": "avg_reasoning_steps from trainer",
        "training_time": "epoch duration",
        "memory_usage": "GPU memory tracking"
    },
    "connection_analysis": {
        "sparsity_ratio": "model.get_connection_analysis()",
        "max_connection": "connection strength analysis",
        "orthogonality_quality": "regularization metrics"
    },
    "training_dynamics": {
        "loss": "train/eval loss curves",
        "grad_norm": "gradient clipping tracking",
        "lr": "learning rate schedule"
    }
}
```

---

## 실험 실행 스크립트

### 단일 실험 실행

```bash
# 기본 Connection Transformer
python main.py --dataset strategyqa --model connection --model_size micro --output_dir ./outputs

# Baseline 비교
python main.py --dataset strategyqa --model baseline --model_size micro --output_dir ./outputs

# 다른 크기 테스트
python main.py --dataset gsm8k --model connection --model_size small --output_dir ./outputs
```

### 배치 실험 (run_experiments.sh 사용)

```bash
# 모든 데이터셋 기본 크기로 실행
./run_experiments.sh all

# 특정 데이터셋 특정 크기
./run_experiments.sh strategyqa micro

# 모든 데이터셋을 small 크기로
./run_experiments.sh all small
```

### 체계적 비교 실험

```bash
#!/bin/bash
# systematic_comparison.sh

echo "🚀 Systematic Connection Transformer Evaluation"

# Phase 1: Micro scale validation
echo "📊 Phase 1: Micro Scale Validation"
./run_experiments.sh strategyqa micro
./run_experiments.sh logiqa micro

# Phase 2: Small scale comparison
echo "📊 Phase 2: Small Scale Comparison"
for dataset in strategyqa logiqa gsm8k; do
    python main.py --dataset $dataset --model connection --model_size small --output_dir ./outputs
    python main.py --dataset $dataset --model baseline --model_size small --output_dir ./outputs
done

# Phase 3: Base scale performance
echo "📊 Phase 3: Base Scale Performance"
for dataset in gsm8k multinli; do
    python main.py --dataset $dataset --model connection --model_size base --output_dir ./outputs
    python main.py --dataset $dataset --model baseline --model_size base --output_dir ./outputs
done

# Phase 4: Generation tasks
echo "📊 Phase 4: Generation Tasks"
for dataset in eli5 commongen; do
    python main.py --dataset $dataset --model connection --model_size base --output_dir ./outputs
    python main.py --dataset $dataset --model baseline --model_size base --output_dir ./outputs
done

# Final analysis
echo "📈 Running comprehensive analysis..."
python analyze_results.py --output_dir ./outputs
```

---

## 실험 일정 및 예상 시간

### 전체 타임라인 (RTX 4090 기준)

```
Phase 1 (1일): Micro scale validation
- strategyqa micro: ~30분 × 2모델 = 1시간
- logiqa micro: ~45분 × 2모델 = 1.5시간

Phase 2 (2일): Small scale comparison
- 3개 데이터셋 × 2모델 × ~2시간 = 12시간

Phase 3 (2일): Base scale performance
- 2개 데이터셋 × 2모델 × ~4시간 = 16시간

Phase 4 (2일): Generation tasks
- 2개 데이터셋 × 2모델 × ~6시간 = 24시간

Phase 5 (1일): Analysis and documentation
- 결과 분석 및 리포트 생성
```

### 일일 체크리스트

```
□ GPU 온도 및 메모리 사용량 확인
□ 실험 로그 정상 기록 확인
□ 체크포인트 저장 확인
□ compare analysis 자동 실행 확인
□ 결과 백업 (experiments/, analysis/, comparisons/)
```

---

## 메모리 최적화 팁

### 현재 구현 기준 최적화

```python
# trainer.py에서 자동 적용되는 최적화들
memory_optimizations = {
    "autocast": "자동 mixed precision",
    "gradient_scaling": "FP16 사용시 자동 스케일링",
    "OOM_handling": "배치 스킵으로 안정성 확보",
    "periodic_cleanup": "torch.cuda.empty_cache() 주기적 실행"
}

# 추가 최적화 옵션
additional_optimizations = {
    "gradient_checkpointing": "메모리 절약 (약 30% 감소)",
    "smaller_batch_accumulation": "작은 배치 + accumulation",
    "eval_batch_larger": "평가시 더 큰 배치 사용 가능"
}
```

### RTX 4090 활용도 극대화

```python
rtx4090_optimization = {
    "tensor_cores": "BF16/FP16 자동 활용",
    "memory_bandwidth": "24GB VRAM 최대 활용",
    "compute_capability": "8.9 - 모든 최신 기능 지원",
    "nvcc_arch": "sm_89 컴파일 최적화"
}
```
