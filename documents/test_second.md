# RTX 4090 기준 Connection Transformer 실험 셋업

## 하드웨어 스펙

- **GPU**: RTX 4090 (24GB VRAM)
- **메모리**: 충분한 RAM (32GB+ 권장)
- **저장공간**: SSD 500GB+ (데이터셋 + 모델 체크포인트)

---

## 실험 설정

### 모델 Configuration

```python
# 기본 설정 (메모리 효율적)
base_config = {
    "d_model": 256,
    "num_slots": 128,           # N < D for compression
    "bilinear_rank": 32,        # 적당한 복잡도
    "max_reasoning_steps": 6,
    "convergence_threshold": 0.01,
    "vocab_size": 32000,        # T5 tokenizer 기준
    "max_seq_len": 512
}

# 큰 모델 설정 (성능 중심)
large_config = {
    "d_model": 512,
    "num_slots": 256,
    "bilinear_rank": 64,
    "max_reasoning_steps": 8,
    "convergence_threshold": 0.005,
    "vocab_size": 32000,
    "max_seq_len": 512
}
```

### 배치 크기 최적화

```python
# RTX 4090 (24GB) 기준 권장 배치 크기
batch_sizes = {
    "base_config": {
        "train_batch": 32,      # 메모리 사용량: ~18GB
        "eval_batch": 64,       # 추론시 더 큰 배치 가능
        "gradient_accumulation": 2  # 효과적 배치 = 64
    },
    "large_config": {
        "train_batch": 16,      # 메모리 사용량: ~20GB
        "eval_batch": 32,
        "gradient_accumulation": 4  # 효과적 배치 = 64
    }
}
```

### 메모리 최적화 전략

```python
optimization_config = {
    # Mixed precision training
    "fp16": True,
    "bf16": False,  # RTX 4090에서는 fp16이 더 안정적

    # Gradient checkpointing
    "gradient_checkpointing": True,  # 메모리 절약

    # DataLoader 설정
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,

    # 메모리 효율성
    "empty_cache_steps": 100,  # 주기적 캐시 정리
}
```

---

## 데이터셋 구성

### Primary 데이터셋 (논문용)

```python
primary_datasets = {
    "LogiQA": {
        "size": "8,678 examples",
        "split": "train(6,678) / dev(1,000) / test(1,000)",
        "preprocessing": "T5 format: 'reason: <premise> question: <question>'",
        "max_length": 256
    },
    "GSM8K": {
        "size": "8,792 examples",
        "split": "train(7,473) / test(1,319)",
        "preprocessing": "T5 format: 'solve: <problem>'",
        "max_length": 512
    },
    "StrategyQA": {
        "size": "2,780 examples",
        "split": "train(2,290) / test(490)",
        "preprocessing": "T5 format: 'strategy: <question>'",
        "max_length": 256
    }
}
```

### Secondary 데이터셋 (추가 검증)

```python
secondary_datasets = {
    "CommonsenseQA": {
        "size": "12,247 examples",
        "split": "train(9,741) / dev(1,221) / test(1,285)",
        "max_length": 128
    },
    "SST-2": {  # 단순 태스크 대조군
        "size": "67,349 examples",
        "split": "train(67,349) / dev(872) / test(1,821)",
        "max_length": 64
    }
}
```

---

## 실험 계획

### Phase 1: 기본 검증 (1-2일)

```python
baseline_experiments = [
    {
        "name": "Linear_N=D",
        "config": "N=256, D=256, Linear connections",
        "purpose": "기존 방식 재현"
    },
    {
        "name": "Bilinear_Base",
        "config": base_config,
        "purpose": "기본 bilinear 성능 확인"
    },
    {
        "name": "Bilinear_Large",
        "config": large_config,
        "purpose": "스케일업 효과 확인"
    }
]

# 각 실험당 예상 시간
training_time_estimates = {
    "LogiQA": "2-3시간 (base) / 4-5시간 (large)",
    "GSM8K": "3-4시간 (base) / 6-7시간 (large)",
    "StrategyQA": "1-2시간 (base) / 2-3시간 (large)"
}
```

### Phase 2: N/D 비율 탐색 (2-3일)

```python
nd_ratio_experiments = [
    {"N": 64,  "D": 256, "name": "4to1_compression"},
    {"N": 128, "D": 256, "name": "2to1_compression"},
    {"N": 256, "D": 256, "name": "1to1_baseline"},
    {"N": 512, "D": 256, "name": "2to1_expansion"}
]

# bilinear_rank=32 고정, 3개 데이터셋에서 테스트
```

### Phase 3: Rank 최적화 (1-2일)

```python
rank_experiments = [
    {"rank": 8,  "name": "minimal_rank"},
    {"rank": 16, "name": "small_rank"},
    {"rank": 32, "name": "medium_rank"},
    {"rank": 64, "name": "large_rank"}
]

# 최적 N/D 비율에서 rank만 변경
```

### Phase 4: Dynamic Reasoning 분석 (1일)

```python
dynamic_experiments = [
    {"threshold": 0.001, "name": "sensitive_termination"},
    {"threshold": 0.01,  "name": "medium_termination"},
    {"threshold": 0.1,   "name": "loose_termination"},
    {"fixed_steps": 4,   "name": "no_dynamic_baseline"}
]
```

---

## 학습 설정

### 최적화 파라미터

```python
training_config = {
    # 옵티마이저
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "betas": (0.9, 0.999),

    # 스케줄러
    "scheduler": "linear_warmup_cosine",
    "warmup_ratio": 0.1,
    "min_lr_ratio": 0.1,

    # 정규화
    "gradient_clip": 1.0,
    "reasoning_cost_weight": 0.001,

    # 에폭
    "max_epochs": 20,
    "patience": 5,  # Early stopping
    "eval_every": 500,  # steps
}
```

### 모니터링 메트릭

```python
metrics_to_track = {
    "performance": ["accuracy", "f1", "exact_match"],
    "efficiency": ["avg_reasoning_steps", "early_termination_rate"],
    "interpretability": ["connection_sparsity", "slot_specialization"],
    "training": ["loss", "grad_norm", "lr", "memory_usage"]
}
```

---

## 실험 스크립트 예시

### 메인 훈련 스크립트

```python
#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, AdamW, get_linear_schedule_with_warmup
import wandb
import os

def setup_experiment():
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # 최적화

    # 메모리 관리
    torch.cuda.empty_cache()

    return device

def train_loop(model, train_loader, eval_loader, config):
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    optimizer = AdamW(model.parameters(),
                     lr=config['learning_rate'],
                     weight_decay=config['weight_decay'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_loader) * config['warmup_ratio']),
        num_training_steps=len(train_loader) * config['max_epochs']
    )

    for epoch in range(config['max_epochs']):
        for step, batch in enumerate(train_loader):
            with torch.cuda.amp.autocast():
                outputs = model(batch['input_ids'], return_reasoning_trace=True)
                logits, reasoning_info = outputs

                # 기본 손실
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)),
                                           batch['labels'].view(-1))

                # 추론 비용 정규화
                reasoning_cost = model.reasoning_cost_loss(
                    reasoning_info['actual_steps'],
                    target_steps=4,
                    weight=config['reasoning_cost_weight']
                )

                total_loss = loss + reasoning_cost

            # Backward pass
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            # 로깅
            if step % 100 == 0:
                wandb.log({
                    "train_loss": total_loss.item(),
                    "reasoning_steps": reasoning_info['actual_steps'],
                    "learning_rate": scheduler.get_last_lr()[0],
                    "memory_used": torch.cuda.memory_allocated() / 1e9
                })

            # 메모리 정리
            if step % config.get('empty_cache_steps', 100) == 0:
                torch.cuda.empty_cache()
```

### 배치 실험 스크립트

```bash
#!/bin/bash
# run_experiments.sh

# Phase 1: 기본 검증
python train.py --config configs/linear_baseline.yaml --dataset LogiQA
python train.py --config configs/bilinear_base.yaml --dataset LogiQA
python train.py --config configs/bilinear_large.yaml --dataset LogiQA

# Phase 2: N/D 비율 (bilinear_base 기준)
for nd_ratio in "64_256" "128_256" "256_256" "512_256"; do
    python train.py --config configs/nd_${nd_ratio}.yaml --dataset LogiQA
done

# Phase 3: Rank 최적화 (최적 N/D에서)
for rank in 8 16 32 64; do
    python train.py --config configs/rank_${rank}.yaml --dataset LogiQA
done

# 모든 실험을 3개 데이터셋에서 반복
for dataset in "LogiQA" "GSM8K" "StrategyQA"; do
    python train.py --config configs/final_best.yaml --dataset $dataset
done
```

---

## 실험 일정

### 전체 타임라인 (7-10일)

```
Day 1-2: Phase 1 (기본 검증)
Day 3-4: Phase 2 (N/D 비율)
Day 5-6: Phase 3 (Rank 최적화)
Day 7: Phase 4 (Dynamic 분석)
Day 8-9: 최적 설정으로 모든 데이터셋 재실험
Day 10: 결과 정리 및 분석
```

### 일일 체크리스트

```
□ 실험 시작 전 GPU 메모리 확인
□ wandb 로깅 정상 작동 확인
□ 중간 체크포인트 저장 확인
□ 메모리 사용량 모니터링
□ 실험 결과 백업
```

이 설정으로 RTX 4090 한 장으로도 충분히 의미있는 실험들을 돌릴 수 있을 것 같아요! 🚀
