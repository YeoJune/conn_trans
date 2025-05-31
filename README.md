# Connection Transformer: Bilinear Connections for Adaptive Reasoning

이 프로젝트는 **bilinear connections**와 **adaptive reasoning**을 도입한 Connection Transformer의 구현입니다. 논리적 추론 능력을 향상시키기 위해 고정된 semantic slots 간의 학습 가능한 연결을 통해 반복적 추론을 수행합니다.

## 🏗️ 아키텍처 개요

### 핵심 혁신사항

1. **Bilinear Connections**: 기존 선형 연결을 bilinear transformation으로 확장
2. **Adaptive Reasoning**: 수렴 기준에 따른 동적 추론 단계 조절
3. **Parameter Efficiency**: 공정한 비교를 위한 parameter-matched baseline

### 모델 구조

```
Input → Embedding → Compression → Adaptive Bilinear Reasoning → Expansion → Output
[B,S]     [B,S,D]      [B,N,D]           [B,N,D] (variable steps)    [B,S,D]   [B,S,V]
```

## 📁 프로젝트 구조

```
connection_transformer/
├── main.py                     # 실험 실행 스크립트
├── configs/                    # 설정 파일들
│   ├── base_config.py
│   ├── logiqa_config.py
│   ├── gsm8k_config.py
│   └── strategyqa_config.py
├── models/                     # 모델 구현
│   ├── connection_transformer.py
│   └── baseline_transformer.py
├── data/                       # 데이터 처리
│   ├── tokenizer_utils.py
│   ├── logiqa_dataset.py
│   ├── gsm8k_dataset.py
│   └── strategyqa_dataset.py
├── training/                   # 훈련 코드
│   └── trainer.py
├── utils/                      # 유틸리티
│   ├── metrics.py
│   └── visualization.py
└── experiments/                # 실험 결과
    ├── results/
    ├── checkpoints/
    └── logs/
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성
conda create -n conn_trans python=3.9
conda activate conn_trans

# 의존성 설치
pip install -r requirements.txt
```

### 2. 기본 실험 실행

```bash
# Connection Transformer 훈련 (LogiQA)
python main.py --dataset logiqa --model connection --model_size base

# Baseline Transformer 훈련 (비교용)
python main.py --dataset logiqa --model baseline --model_size base

# 다른 데이터셋에서 실험
python main.py --dataset gsm8k --model connection --model_size base
python main.py --dataset strategyqa --model connection --model_size base
```

### 3. 배치 실험 실행

```bash
# 모든 실험 자동 실행
chmod +x run_experiments.sh
./run_experiments.sh

# 결과 분석
python analyze_results.py --results_dir experiments/results --output_dir experiments/analysis
```

## 📊 지원 데이터셋

| 데이터셋       | 태스크 유형 | 샘플 수 | 설명                |
| -------------- | ----------- | ------- | ------------------- |
| **LogiQA**     | 논리적 추론 | ~8K     | 다중 선택 논리 문제 |
| **GSM8K**      | 수학 추론   | ~8K     | 초등학교 수학 문제  |
| **StrategyQA** | 전략적 추론 | ~2K     | Yes/No 전략 질문    |

## ⚙️ 주요 설정

### Connection Transformer 설정

```python
# 기본 설정
config = {
    "d_model": 256,
    "num_slots": 128,           # Semantic slots 수
    "bilinear_rank": 32,        # Bilinear connection rank
    "max_reasoning_steps": 6,   # 최대 추론 단계
    "convergence_threshold": 0.01,  # 수렴 임계값
    "learning_rate": 1e-4,
    "batch_size": 32
}

# 큰 모델 설정
large_config = {
    "d_model": 512,
    "num_slots": 256,
    "bilinear_rank": 64,
    "max_reasoning_steps": 8
}
```

### RTX 4090 최적화 설정

```python
# 메모리 효율성
optimization = {
    "fp16": True,                    # Mixed precision
    "gradient_checkpointing": True,  # 메모리 절약
    "batch_size": 32,               # Base model
    "batch_size_large": 16,         # Large model
}
```

## 📈 실험 결과 분석

### 자동 생성되는 분석

1. **성능 비교**: Connection vs Baseline Transformer
2. **추론 효율성**: 평균 추론 단계, 수렴 패턴
3. **Connection 패턴**: Bilinear connection 시각화
4. **훈련 곡선**: Loss, accuracy, reasoning steps

### 주요 메트릭

- **Accuracy**: 정확한 답변 비율
- **Reasoning Steps**: 평균 추론 단계 수
- **Connection Sparsity**: 희소한 연결 패턴 비율
- **Parameter Efficiency**: 동일 파라미터 수 대비 성능

## 🔬 모델 분석 도구

### Connection Matrix 시각화

```python
from utils.visualization import visualize_connection_matrix

# 훈련된 모델의 connection pattern 분석
visualize_connection_matrix(model, save_path="connection_analysis.png")
```

### 추론 과정 분석

```python
from utils.visualization import analyze_reasoning_patterns

# 추론 패턴과 수렴 과정 분석
analyze_reasoning_patterns(model, save_path="reasoning_patterns.png")
```

## 📊 성능 벤치마크

### 예상 결과 (RTX 4090 기준)

| 모델                   | LogiQA | GSM8K | StrategyQA | 평균 추론 단계 |
| ---------------------- | ------ | ----- | ---------- | -------------- |
| **Connection (base)**  | 0.752  | 0.681 | 0.734      | 4.2            |
| **Baseline (matched)** | 0.731  | 0.663 | 0.718      | N/A            |
| **Connection (large)** | 0.784  | 0.723 | 0.761      | 5.1            |

### 훈련 시간 (추정)

- **Base model**: 2-3시간/데이터셋
- **Large model**: 4-6시간/데이터셋
- **전체 실험**: 24-30시간

## 🛠️ 고급 사용법

### 커스텀 데이터셋 추가

1. `data/` 폴더에 새 데이터셋 클래스 생성
2. `configs/` 폴더에 설정 파일 추가
3. `tokenizer_utils.py`에 데이터셋 등록

```python
# data/custom_dataset.py
class CustomDataset(Dataset):
    def __init__(self, tokenizer, config, split="train"):
        # 구현
        pass
```

### 하이퍼파라미터 튜닝

```bash
# N/D 비율 실험
python main.py --dataset logiqa --model connection --config custom_nd_ratio.yaml

# Bilinear rank 실험
python main.py --dataset logiqa --model connection --config custom_rank.yaml
```

### 체크포인트에서 재시작

```bash
python main.py --dataset logiqa --model connection --resume best_connection_logiqa.pt
```

## 🐛 문제 해결

### 메모리 부족 오류

```bash
# 배치 크기 줄이기
python main.py --dataset gsm8k --model connection --batch_size 16

# Gradient checkpointing 활성화
# (기본적으로 활성화됨)
```

### 데이터셋 로딩 오류

```python
# HuggingFace 로그인 (필요한 경우)
huggingface-cli login

# 캐시 초기화
rm -rf ~/.cache/huggingface/datasets
```

## 📚 참고 문헌

- **Connection Transformer**: [원본 논문 링크]
- **Bilinear Connections**: Enhanced semantic slot interactions
- **Adaptive Reasoning**: Dynamic computation
- **Transformer Architecture**: Vaswani et al., "Attention Is All You Need"

## 🤝 기여 가이드

### 새로운 모델 변형 추가

1. `models/` 폴더에 새 모델 클래스 생성
2. `ConnectionTransformer`를 상속하여 구현
3. `main.py`에 모델 선택 옵션 추가

### 새로운 평가 메트릭 추가

1. `utils/metrics.py`에 메트릭 함수 추가
2. `training/trainer.py`에서 메트릭 계산 통합
3. 시각화 함수도 `utils/visualization.py`에 추가

## 🔧 개발자 도구

### 디버깅 모드

```bash
# 작은 데이터셋으로 빠른 테스트
python main.py --dataset logiqa --model connection --debug --max_samples 100

# 상세 로깅
python main.py --dataset logiqa --model connection --log_level debug
```

### 프로파일링

```python
# 메모리 사용량 모니터링
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# 훈련 속도 측정
import time
start_time = time.time()
# 훈련 코드
print(f"Training time: {time.time() - start_time:.2f}s")
```

## 📋 TODO 및 향후 계획

### 단기 목표

- [ ] Multi-GPU 훈련 지원
- [ ] FSDP (Fully Sharded Data Parallel) 통합
- [ ] 더 많은 추론 데이터셋 추가 (CommonsenseQA, ARC, etc.)
- [ ] Bilinear rank 자동 조정

### 장기 목표

- [ ] 다중 모달 입력 지원
- [ ] 대화형 추론 태스크
- [ ] 설명 가능한 추론 경로 생성
- [ ] 온라인 학습 및 적응

## 🎯 실험 가이드라인

### Phase 1: 기본 검증 (1-2일)

```bash
# 기본 성능 확인
./run_basic_experiments.sh
```

### Phase 2: 하이퍼파라미터 최적화 (2-3일)

```bash
# N/D 비율 실험
python sweep_nd_ratio.py

# Bilinear rank 실험
python sweep_bilinear_rank.py
```

### Phase 3: 심층 분석 (1-2일)

```bash
# Connection 패턴 분석
python analyze_connections.py

# 추론 과정 시각화
python visualize_reasoning.py
```

## 💡 팁과 트릭

### 효율적인 실험 관리

```bash
# tmux 세션으로 장시간 실험 관리
tmux new-session -d -s experiments
tmux send-keys -t experiments './run_experiments.sh' C-m

# wandb로 실험 추적
export WANDB_PROJECT="connection_transformer"
python main.py --dataset logiqa --model connection --use_wandb
```

### 메모리 최적화

```python
# Gradient accumulation으로 큰 배치 효과
effective_batch_size = batch_size * gradient_accumulation_steps

# Mixed precision으로 메모리 절약
with torch.cuda.amp.autocast():
    outputs = model(inputs)
```

### 빠른 프로토타이핑

```python
# 작은 모델로 빠른 테스트
quick_config = {
    "d_model": 128,
    "num_slots": 64,
    "bilinear_rank": 16,
    "max_reasoning_steps": 3
}
```

## 🚨 알려진 이슈

### 메모리 관련

- Large model + 큰 배치 크기 시 OOM 가능
- 해결: `--gradient_checkpointing` 사용

### 데이터셋 관련

- 일부 HuggingFace 데이터셋 접근 제한
- 해결: 대체 데이터셋 자동 로드

### 훈련 안정성

- 매우 큰 bilinear rank에서 gradient explosion 가능
- 해결: Gradient clipping 및 적절한 learning rate

## 📞 지원 및 문의

- **이슈 리포트**: GitHub Issues 사용
- **기능 요청**: GitHub Discussions 사용
- **버그 제보**: 재현 가능한 예제와 함께 제보

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

## 🙏 감사의 말

- HuggingFace Transformers 라이브러리
- PyTorch 팀
- 오픈소스 커뮤니티의 모든 기여자들

---

**Happy Reasoning! 🧠✨**
