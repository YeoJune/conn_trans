# dataset/tokenizer_utils.py
from transformers import T5Tokenizer
from .gsm8k_dataset import GSM8KDataset
from .logiqa_dataset import LogiQADataset
from .multinli_dataset import MultiNLIDataset
from .strategyqa_dataset import StrategyQADataset
from .eli5_dataset import ELI5Dataset
from .commongen_dataset import CommonGenDataset
import torch
from torch.utils.data import Subset
import random

def get_tokenizer_and_dataset(dataset_name, config):
    """Load tokenizer and dataset with proper validation handling"""
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config.tokenizer_name)
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    
    # Dataset mapping
    datasets = {
        "gsm8k": GSM8KDataset,
        "logiqa": LogiQADataset,
        "multinli": MultiNLIDataset,
        "strategyqa": StrategyQADataset,
        "eli5": ELI5Dataset,
        "commongen": CommonGenDataset
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_class = datasets[dataset_name]
    
    # Load train dataset
    train_dataset = dataset_class(tokenizer, config, split="train")
    eval_dataset = None
    
    # Try to load validation set
    try:
        eval_dataset = dataset_class(tokenizer, config, split="validation")
        print(f"✅ Found validation split: {len(eval_dataset)} samples")
    except:
        print(f"📋 No validation split found for {dataset_name}")
    
    # If no validation, try test split for validation
    if eval_dataset is None:
        try:
            eval_dataset = dataset_class(tokenizer, config, split="test")
            print(f"⚠️ Using test split as validation: {len(eval_dataset)} samples")
            print(f"   (Note: This should only be used for development, not final evaluation)")
        except:
            print(f"📋 No test split found either")
    
    # Last resort: split train data
    if eval_dataset is None:
        print(f"🔀 Auto-splitting train data for validation")
        train_dataset, eval_dataset = _create_train_val_split(
            train_dataset, 
            val_ratio=getattr(config, 'val_split_ratio', 0.1),
            seed=getattr(config, 'seed', 42)
        )
    
    print(f"✅ Final split: train={len(train_dataset)}, eval={len(eval_dataset)}")
    
    # 데이터 검증
    _verify_datasets(train_dataset, eval_dataset, dataset_name)
    
    return tokenizer, train_dataset, eval_dataset


def _create_train_val_split(dataset, val_ratio=0.1, seed=42):
    """Create train/validation split from train dataset"""
    
    # Set seeds for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    # Create random indices
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    print(f"   → Split: {train_size} train / {val_size} validation")
    
    return train_subset, val_subset


def _verify_datasets(train_dataset, eval_dataset, dataset_name):
    """Verify dataset integrity"""
    try:
        # Test loading a few samples
        train_sample = train_dataset[0]
        eval_sample = eval_dataset[0]
        
        # Check required fields
        required_fields = ['input_text', 'target_text']
        for field in required_fields:
            if field not in train_sample:
                print(f"⚠️ Missing field '{field}' in train dataset")
            if field not in eval_sample:
                print(f"⚠️ Missing field '{field}' in eval dataset")
        
        # Show sample data
        print(f"📋 Sample data from {dataset_name}:")
        print(f"   Train input: {train_sample['input_text'][:50]}...")
        print(f"   Train target: {train_sample['target_text']}")
        print(f"   Eval input: {eval_sample['input_text'][:50]}...")
        print(f"   Eval target: {eval_sample['target_text']}")
        
    except Exception as e:
        print(f"⚠️ Dataset verification failed: {e}")


def get_test_dataset(dataset_name, config, tokenizer):
    """
    Separate function to get test dataset for final evaluation
    (only use this after training is complete)
    """
    datasets = {
        "gsm8k": GSM8KDataset,
        "logiqa": LogiQADataset,
        "multinli": MultiNLIDataset,
        "strategyqa": StrategyQADataset,
        "eli5": ELI5Dataset,
        "commongen": CommonGenDataset
    }
    
    dataset_class = datasets[dataset_name]
    
    try:
        test_dataset = dataset_class(tokenizer, config, split="test")
        print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
        return test_dataset
    except Exception as e:
        print(f"❌ Failed to load test dataset: {e}")
        return None