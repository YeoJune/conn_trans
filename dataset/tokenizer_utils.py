# dataset/tokenizer_utils.py

from transformers import T5Tokenizer
from .logiqa_dataset import LogiQADataset
from .gsm8k_dataset import GSM8KDataset
from .strategyqa_dataset import StrategyQADataset
from .multinli_dataset import MultiNLIDataset

def get_tokenizer_and_dataset(dataset_name, config):
    """
    T5에 최적화된 토크나이저와 데이터셋 로딩
    
    주요 수정 사항:
    1. as_target_tokenizer() 사용
    2. labels에서 pad_token_id -> -100 변환
    3. 안정적인 데이터셋 소스 사용
    """
    
    print(f"🔄 Loading T5 tokenizer: {config.tokenizer_name}")
    
    try:
        # T5 토크나이저 (최신 방식)
        tokenizer = T5Tokenizer.from_pretrained(
            config.tokenizer_name,
            legacy=False,
            use_fast=True  # 빠른 토크나이저 사용
        )
        print(f"✅ T5Tokenizer loaded successfully")
    except Exception as e:
        print(f"⚠️ Fast tokenizer failed, falling back: {e}")
        tokenizer = T5Tokenizer.from_pretrained(config.tokenizer_name)
    
    # T5는 기본적으로 pad_token이 설정되어 있음
    print(f"✅ Tokenizer loaded. Vocab size: {tokenizer.vocab_size:,}")
    print(f"   Pad token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
    print(f"   EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
    
    # 데이터셋 클래스 매핑
    dataset_classes = {
        "logiqa": LogiQADataset,
        "gsm8k": GSM8KDataset,
        "strategyqa": StrategyQADataset,
        "multinli": MultiNLIDataset
    }
    
    if dataset_name not in dataset_classes:
        available = list(dataset_classes.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    dataset_class = dataset_classes[dataset_name]
    
    print(f"🔄 Loading {dataset_name} dataset with T5 optimization...")
    
    # 훈련 데이터셋
    train_dataset = dataset_class(tokenizer, config, split="train")
    
    # 검증 데이터셋
    try:
        eval_dataset = dataset_class(tokenizer, config, split="validation")
    except:
        try:
            eval_dataset = dataset_class(tokenizer, config, split="test")
            print("⚠️ Using test split as validation")
        except:
            # StrategyQA의 경우 train에서 분할
            print("⚠️ Creating validation from train split")
            total_size = len(train_dataset)
            eval_size = min(500, total_size // 5)  # 최대 500개 또는 20%
            
            # 간단한 분할
            eval_indices = list(range(total_size - eval_size, total_size))
            train_indices = list(range(total_size - eval_size))
            
            # Subset 생성 (간단한 구현)
            class DatasetSubset:
                def __init__(self, dataset, indices):
                    self.dataset = dataset
                    self.indices = indices
                
                def __len__(self):
                    return len(self.indices)
                
                def __getitem__(self, idx):
                    return self.dataset[self.indices[idx]]
            
            eval_dataset = DatasetSubset(train_dataset, eval_indices)
            train_dataset = DatasetSubset(train_dataset, train_indices)
    
    print(f"✅ Dataset loaded:")
    print(f"   Train: {len(train_dataset):,} examples")
    print(f"   Eval: {len(eval_dataset):,} examples")
    
    # 데이터 샘플 확인
    print(f"\n🔍 Sample data check:")
    sample = train_dataset[0]
    print(f"   Input shape: {sample['input_ids'].shape}")
    print(f"   Labels shape: {sample['labels'].shape}")
    print(f"   Input text: {sample.get('target_text', 'N/A')[:50]}...")
    
    # Labels에 -100이 제대로 있는지 확인
    labels = sample['labels']
    mask_count = (labels == -100).sum().item()
    print(f"   Labels masked tokens: {mask_count}")
    
    return tokenizer, train_dataset, eval_dataset