# dataset/tokenizer_utils.py

from transformers import T5Tokenizer
from .logiqa_dataset import LogiQADataset
from .gsm8k_dataset import GSM8KDataset
from .strategyqa_dataset import StrategyQADataset
from .multinli_dataset import MultiNLIDataset

def get_tokenizer_and_dataset(dataset_name, config):
    """토크나이저와 데이터셋을 함께 반환"""
    
    # 토크나이저 로딩 (기존과 동일)
    print(f"🔄 Loading tokenizer: {config.tokenizer_name}")
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(
            config.tokenizer_name, 
            legacy=False
        )
        print(f"✅ Using modern T5Tokenizer (legacy=False)")
    except Exception as e:
        print(f"⚠️ Modern tokenizer failed, falling back to legacy mode: {e}")
        tokenizer = T5Tokenizer.from_pretrained(config.tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   ⚠️ pad_token이 None이어서 eos_token으로 설정")
    
    print(f"✅ Tokenizer loaded. Vocab size: {tokenizer.vocab_size:,}")
    print(f"   Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"   EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    
    # ✅ 여기가 실제 변경 부분 - MultiNLI 추가
    dataset_classes = {
        "logiqa": LogiQADataset,
        "gsm8k": GSM8KDataset,
        "strategyqa": StrategyQADataset,
        "multinli": MultiNLIDataset  # 이 한 줄만 추가하면 됨
    }
    
    # 나머지는 기존과 동일
    if dataset_name not in dataset_classes:
        available = list(dataset_classes.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    dataset_class = dataset_classes[dataset_name]
    
    print(f"🔄 Loading {dataset_name} dataset...")
    train_dataset = dataset_class(tokenizer, config, split="train")
    
    try:
        eval_dataset = dataset_class(tokenizer, config, split="validation")
    except:
        try:
            eval_dataset = dataset_class(tokenizer, config, split="test")
            print("⚠️ Validation split not found, using test split")
        except:
            print("⚠️ No validation/test split, creating validation from train")
            eval_dataset = dataset_class(tokenizer, config, split="train")
    
    print(f"✅ Dataset loaded. Train: {len(train_dataset):,}, Eval: {len(eval_dataset):,}")
    
    return tokenizer, train_dataset, eval_dataset