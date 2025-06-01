# dataset/tokenizer_utils.py

from transformers import T5Tokenizer
from .logiqa_dataset import LogiQADataset
from .gsm8k_dataset import GSM8KDataset
from .strategyqa_dataset import StrategyQADataset
from .multinli_dataset import MultiNLIDataset

def get_tokenizer_and_dataset(dataset_name, config):
    """토크나이저와 데이터셋 로딩"""
    
    # 토크나이저 로드
    tokenizer = T5Tokenizer.from_pretrained(config.tokenizer_name)
    
    # Config 업데이트
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    
    print(f"✅ Tokenizer loaded: vocab_size={tokenizer.vocab_size:,}")
    
    # 데이터셋 로드
    dataset_map = {
        "logiqa": LogiQADataset,
        "gsm8k": GSM8KDataset,
        "strategyqa": StrategyQADataset,
        "multinli": MultiNLIDataset
    }
    
    dataset_class = dataset_map[dataset_name]
    train_dataset = dataset_class(tokenizer, config, split="train")
    
    try:
        eval_dataset = dataset_class(tokenizer, config, split="validation")
    except:
        # 간단한 train/eval 분할
        train_size = len(train_dataset)
        split_idx = int(train_size * 0.9)
        
        class SimpleSubset:
            def __init__(self, dataset, start, end):
                self.dataset = dataset
                self.indices = list(range(start, end))
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
        
        eval_dataset = SimpleSubset(train_dataset, split_idx, train_size)
        train_dataset = SimpleSubset(train_dataset, 0, split_idx)
    
    print(f"✅ Dataset loaded: train={len(train_dataset)}, eval={len(eval_dataset)}")
    
    return tokenizer, train_dataset, eval_dataset
