# dataset/tokenizer_utils.py (simplified)
from transformers import T5Tokenizer
from .gsm8k_dataset import GSM8KDataset
from .logiqa_dataset import LogiQADataset
from .multinli_dataset import MultiNLIDataset
from .strategyqa_dataset import StrategyQADataset

def get_tokenizer_and_dataset(dataset_name, config):
    """Load tokenizer and dataset"""
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config.tokenizer_name)
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    
    # Dataset mapping
    datasets = {
        "gsm8k": GSM8KDataset,
        "logiqa": LogiQADataset,
        "multinli": MultiNLIDataset,
        "strategyqa": StrategyQADataset
    }
    
    dataset_class = datasets[dataset_name]
    
    # Load train/eval datasets
    train_dataset = dataset_class(tokenizer, config, split="train")
    
    try:
        eval_dataset = dataset_class(tokenizer, config, split="validation")
    except:
        # Simple train/eval split
        train_size = len(train_dataset)
        split_idx = int(train_size * 0.9)
        
        eval_dataset = _SimpleSubset(train_dataset, split_idx, train_size)
        train_dataset = _SimpleSubset(train_dataset, 0, split_idx)
    
    print(f"✅ Loaded: train={len(train_dataset)}, eval={len(eval_dataset)}")
    return tokenizer, train_dataset, eval_dataset


class _SimpleSubset:
    """Simple dataset subset for train/eval split"""
    def __init__(self, dataset, start, end):
        self.dataset = dataset
        self.indices = list(range(start, end))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]