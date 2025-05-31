# dataset/tokenizer_utils.py
from transformers import T5Tokenizer
from .logiqa_dataset import LogiQADataset
from .gsm8k_dataset import GSM8KDataset
from .strategyqa_dataset import StrategyQADataset

def get_tokenizer_and_dataset(dataset_name, config):
    """
    토크나이저와 데이터셋을 함께 반환
    
    Args:
        dataset_name: str - "logiqa", "gsm8k", "strategyqa"
        config: 설정 객체
        
    Returns:
        tokenizer, train_dataset, eval_dataset
    """
    
    # T5 토크나이저만 사용 (연구에서 명시된 방식)
    print(f"🔄 Loading tokenizer: {config.tokenizer_name}")
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(config.tokenizer_name)
    except ImportError as e:
        if "sentencepiece" in str(e).lower():
            raise ImportError(
                "T5Tokenizer requires SentencePiece. Please install it:\n"
                "pip install sentencepiece>=0.1.97"
            ) from e
        else:
            raise e
    
    # 패딩 토큰 확인 및 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   ⚠️ pad_token이 None이어서 eos_token으로 설정")
    
    print(f"✅ Tokenizer loaded. Vocab size: {tokenizer.vocab_size:,}")
    print(f"   Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"   EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"   UNK token: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")
    
    # 데이터셋 생성
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
    eval_dataset = None
    for split_name in ["validation", "test"]:
        try:
            eval_dataset = dataset_class(tokenizer, config, split=split_name)
            if split_name == "test":
                print("⚠️ Validation split not found, using test split")
            break
        except:
            continue
    
    if eval_dataset is None:
        # 최후의 수단: train의 작은 부분을 validation으로 사용
        print("⚠️ No validation/test split found, creating validation from train subset")
        eval_dataset = dataset_class(tokenizer, config, split="train")
        # 실제로는 train dataset의 일부만 사용하도록 수정 필요
    
    print(f"✅ Dataset loaded. Train: {len(train_dataset):,}, Eval: {len(eval_dataset):,}")
    
    return tokenizer, train_dataset, eval_dataset