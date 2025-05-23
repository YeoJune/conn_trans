# configs/squad_config.py
from .base_config import get_base_config

SQUAD_CONFIG = get_base_config()  # 기본 설정 상속

SQUAD_CONFIG.update({
    "task_name": "squad_v1.1",
    "dataset_name_hf": "squad",  # HuggingFace 데이터셋 이름 (SQuAD 1.1)
    # SQuAD는 name 파라미터 없이 split만 지정

    "max_seq_len": 384,  # SQuAD는 문맥이 길 수 있음
    "max_query_length": 64,  # SQuAD 질문 최대 길이
    "doc_stride": 128,  # SQuAD 슬라이딩 윈도우 stride

    "batch_size": 16,  # 입력이 길어서 배치 크기 줄임
    "learning_rate": 3e-5,  # BERT 스타일의 작은 학습률
    "warmup_steps": 500,  # 데이터셋 크기에 따라 조절
    "max_epochs": 15,  # SQuAD는 데이터가 커서 에폭 줄임 (실제로는 더 필요)

    "tokenizer_name": "bert-base-uncased",  # SQuAD에 사용할 토크나이저
})


def get_squad_config():
    return SQUAD_CONFIG.copy()