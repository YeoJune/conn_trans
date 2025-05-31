# dataset/tokenizer_utils.py
from transformers import T5Tokenizer
from .logiqa_dataset import LogiQADataset
from .gsm8k_dataset import GSM8KDataset
from .strategyqa_dataset import StrategyQADataset

def get_tokenizer_and_dataset(dataset_name, config):
    """
    토크나이저와 데이터셋을 함께 반환
    """
    
    # T5 토크나이저 생성 (legacy=False로 설정하여 최신 방식 사용)
    print(f"🔄 Loading tokenizer: {config.tokenizer_name}")
    
    try:
        # 최신 T5 토크나이저 사용 (legacy=False)
        tokenizer = T5Tokenizer.from_pretrained(
            config.tokenizer_name, 
            legacy=False  # 최신 동작 방식 사용
        )
        print(f"✅ Using modern T5Tokenizer (legacy=False)")
    except Exception as e:
        print(f"⚠️ Modern tokenizer failed, falling back to legacy mode: {e}")
        tokenizer = T5Tokenizer.from_pretrained(config.tokenizer_name)
    
    # 패딩 토큰 확인 및 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   ⚠️ pad_token이 None이어서 eos_token으로 설정")
    
    print(f"✅ Tokenizer loaded. Vocab size: {tokenizer.vocab_size:,}")
    print(f"   Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"   EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    
    # 나머지 코드는 동일...
    dataset_classes = {
        "logiqa": LogiQADataset,
        "gsm8k": GSM8KDataset,
        "strategyqa": StrategyQADataset
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_classes.keys())}")
    
    dataset_class = dataset_classes[dataset_name]
    
    print(f"🔄 Loading {dataset_name} dataset...")
    train_dataset = dataset_class(tokenizer, config, split="train")
    
    # validation split이 없는 경우 test 사용
    try:
        eval_dataset = dataset_class(tokenizer, config, split="validation")
    except:
        try:
            eval_dataset = dataset_class(tokenizer, config, split="test")
            print("⚠️ Validation split not found, using test split")
        except:
            # train의 일부를 validation으로 사용
            print("⚠️ No validation/test split, creating validation from train")
            eval_dataset = dataset_class(tokenizer, config, split="train")
    
    print(f"✅ Dataset loaded. Train: {len(train_dataset):,}, Eval: {len(eval_dataset):,}")
    
    return tokenizer, train_dataset, eval_dataset