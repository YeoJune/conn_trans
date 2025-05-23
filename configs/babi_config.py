# configs/babi_config.py
from .base_config import get_base_config

BABI_CONFIG = get_base_config()  # 기본 설정 상속

BABI_CONFIG.update({
    "task_name": "babi_qa1",  # 예시: 'babi_qa1', 'babi_qa16' 등 main_babi.py에서 설정 가능
    "dataset_name_hf": "facebook/babi_qa",  # HuggingFace 데이터셋 이름
    "babi_hf_config_name": "en-10k-qa1",  # task_id=1일 때 사용할 name (main_babi.py에서 task_id에 따라 동적 생성)
    # 또는 "en-10k"로 하고 task_no를 사용

    "max_seq_len": 128,  # bAbI는 문장이 짧음
    "batch_size": 32,
    "learning_rate": 1e-4,  # bAbI에 적합한 학습률
    "warmup_steps": 500,
    "max_epochs": 15,  # bAbI는 더 많은 에폭 필요할 수 있음

    # bAbI는 자체 어휘 구축하므로 tokenizer_name은 사용 안 함 (또는 None)
    "tokenizer_name": None,
})


def get_babi_config():
    return BABI_CONFIG.copy()