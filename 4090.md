## 🎯 **RTX 4090 최적화 설정**

### **하이퍼파라미터 (4090 24GB)**

```python
# 모델 크기 (4090 최적화 - 더 큰 모델 가능)
d_model = 512        # 384 → 512로 증가
num_ir = 1024        # 2 * d_model
num_steps = 4        # 3 → 4로 증가 (더 깊은 추론)
num_heads = 8        # 6 → 8로 증가
ffn_dim = 2048       # 4 * d_model

# 학습 설정 (4090의 높은 성능 활용)
batch_size = 32      # 24 → 32로 증가
max_seq_len = 128
learning_rate = 1e-4
gradient_clip = 1.0
```

## 📊 **기존 데이터셋 추천**

### **1. bAbI Tasks (1차 추천 🏆)**

```
크기: 20개 태스크, 각 1K 샘플
다운로드: HuggingFace datasets
예시:
- Task 1: "Mary went to the bathroom. John moved to the hallway. Where is Mary?" → "bathroom"
- Task 16: "If Jeff is a frog then Jeff is green. Jeff is a frog. What color is Jeff?" → "green"

장점:
✅ 논리 추론에 최적화
✅ 다양한 추론 깊이 (1-3 단계)
✅ 작은 어휘 사이즈
✅ 명확한 정답
```

### **2. LogiQA (2차 추천)**

```
크기: 8K 샘플
타입: 중국어 → 영어 논리 추론
예시: 조건부 추론, 귀납/연역
다운로드: 논문 저자 공개 데이터
```

### **3. CLUTRR (3차 추천)**

```
크기: 가변 (생성 가능)
타입: 가족 관계 추론
예시: "A is B's father. B is C's mother. What is A to C?" → "grandfather"
```

## ⏱️ **예상 시간 (RTX 4090)**

### **bAbI Tasks**

```
Pure Conn-Trans (20M params):
- 에폭당: ~4분 (Task 1) - 4090이 3090보다 50% 빠름
- 전체 학습: ~1시간 (15 epochs)
- 메모리: ~12GB

With FFN (30M params):
- 에폭당: ~6분
- 전체 학습: ~1.5시간
- 메모리: ~16GB

Standard Transformer (25M params):
- 에폭당: ~5분
- 전체 학습: ~1.3시간
- 메모리: ~14GB

전체 3개 모델 실험: ~4시간
전체 20개 태스크: 1일
```

## 🚀 **실험 로드맵**

### **Day 1: 빠른 검증 (4090 고속 실험)**

```
bAbI Task 1 (단순 팩트 기억)
→ 20분 만에 오버피팅 확인
→ 3개 모델 모두 빠른 테스트
→ 아키텍처 버그 체크
```

### **Day 2: 핵심 추론 태스크**

```
bAbI Task 16 (기본 추론)
→ Pure vs FFN vs Standard 성능 비교
→ Connection Matrix 시각화
→ 추론 과정 step-by-step 분석
```

### **Day 3: 심화 분석**

```
bAbI Task 17 (위치 추론)
bAbI Task 19 (경로 찾기)
→ 복잡한 추론에서의 성능 차이
→ 파라미터 효율성 분석
```

## 💻 **구체적 사용법**

### **데이터 로드**

```python
from datasets import load_dataset

# bAbI 태스크 로드
dataset = load_dataset("facebook/babi_qa", "en-valid-10k")
task_1 = dataset.filter(lambda x: x['task'] == 1)

# 간단한 전처리만 하면 됨
```

### **메모리 최적화 (4090용)**

```python
# 4090용 설정 (더 큰 모델 지원)
torch.backends.cudnn.benchmark = True
# model = model.half()  # FP16 선택적 사용 (충분한 메모리)
torch.cuda.empty_cache()  # 주기적 정리
```

## 📈 **예상 성능 (4090 더 큰 모델)**

### **bAbI Tasks**

```
Task 1 (팩트): 98%+ (더 큰 모델로 향상)
Task 16 (추론): 85-95% (핵심, 향상 기대)
Task 19 (복잡): 70-85% (도전, 큰 폭 향상)

Pure vs FFN: 3-10% 차이 (큰 모델로 격차 축소)
Conn-Trans vs Standard: 경쟁력 있을 것으로 예상
```

## 🎯 **성공 기준**

### **최소 목표**

- bAbI Task 16에서 85% 달성 (3090 대비 5% 향상)
- Connection Matrix에서 명확한 추론 패턴 발견
- Standard Transformer와 경쟁력 있는 성능

### **우수 목표**

- 여러 bAbI 태스크에서 SOTA 근접 또는 달성
- Pure 버전이 Standard Transformer 90% 이상 성능
- 파라미터 효율성에서 명확한 우위

### **도전 목표**

- 20개 bAbI 태스크 중 15개 이상에서 90%+ 달성
- Connection Matrix 해석을 통한 추론 메커니즘 규명

**4090 장점 활용:** 더 큰 모델 + 빠른 실험 + 다양한 태스크 동시 테스트 가능! 🚀
