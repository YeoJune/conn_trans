# dataset/tokenizer_utils.py

from transformers import T5Tokenizer
from .logiqa_dataset import LogiQADataset
from .gsm8k_dataset import GSM8KDataset
from .strategyqa_dataset import StrategyQADataset
from .multinli_dataset import MultiNLIDataset

def get_tokenizer_and_dataset(dataset_name, config):
    """
    Encoder-Decoder 모델용 토크나이저와 데이터셋 로딩
    """
    
    print(f"🔄 Loading T5 tokenizer: {config.tokenizer_name}")
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(
            config.tokenizer_name,
            legacy=False
        )
        print(f"✅ T5Tokenizer loaded successfully")
    except Exception as e:
        print(f"⚠️ Latest tokenizer failed, using fallback: {e}")
        tokenizer = T5Tokenizer.from_pretrained(config.tokenizer_name)
    
    # Config에 실제 tokenizer vocab_size 설정
    config.src_vocab_size = tokenizer.vocab_size
    config.tgt_vocab_size = tokenizer.vocab_size
    config.src_pad_token_id = tokenizer.pad_token_id
    config.tgt_pad_token_id = tokenizer.pad_token_id
    config.vocab_size = tokenizer.vocab_size
    
    print(f"✅ Tokenizer info:")
    print(f"   - Vocab size: {tokenizer.vocab_size:,}")
    print(f"   - Pad token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
    print(f"   - EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")

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
            # 훈련 데이터에서 분할 (더 정교한 방법)
            print("⚠️ Creating validation from train split")
            train_size = len(train_dataset)
            
            # 데이터셋 크기에 따른 분할 비율 조정
            if train_size < 1000:
                eval_ratio = 0.2  # 작은 데이터셋: 20%
            elif train_size < 10000:
                eval_ratio = 0.15  # 중간 데이터셋: 15%
            else:
                eval_ratio = 0.1   # 큰 데이터셋: 10%
            
            eval_size = int(train_size * eval_ratio)
            train_size_new = train_size - eval_size
            
            # 더 안전한 Subset 구현
            class SafeDatasetSubset:
                def __init__(self, dataset, indices):
                    self.dataset = dataset
                    self.indices = list(indices)
                
                def __len__(self):
                    return len(self.indices)
                
                def __getitem__(self, idx):
                    if idx >= len(self.indices):
                        raise IndexError(f"Index {idx} out of range for subset of size {len(self.indices)}")
                    return self.dataset[self.indices[idx]]
            
            # 인덱스 분할 (뒤쪽을 eval로)
            eval_indices = list(range(train_size_new, train_size))
            train_indices = list(range(train_size_new))
            
            eval_dataset = SafeDatasetSubset(train_dataset, eval_indices)
            train_dataset = SafeDatasetSubset(train_dataset, train_indices)
    
    print(f"✅ Dataset loaded:")
    print(f"   Train: {len(train_dataset):,} examples")
    print(f"   Eval: {len(eval_dataset):,} examples")
    
    # 🔍 Encoder-Decoder 데이터 샘플 확인
    print(f"\n🔍 Encoder-Decoder data sample verification:")
    try:
        sample = train_dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Source input shape: {sample['input_ids'].shape}")
        print(f"   Target labels shape: {sample['labels'].shape}")
        
        # Source text 확인
        src_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        print(f"   Source text: '{src_text[:80]}...'")
        print(f"   Target text: '{sample['target_text']}'")
        
        # Target sequence 길이 확인
        tgt_labels = sample['labels']
        valid_tgt_tokens = (tgt_labels != -100).sum().item()
        print(f"   Target sequence: {valid_tgt_tokens} valid tokens")
        
        # Encoder-Decoder 모델에서는 target이 짧아도 괜찮음
        if valid_tgt_tokens < 2:
            print("   ℹ️ Short target sequence - normal for classification tasks")
        
    except Exception as e:
        print(f"   ❌ Sample verification failed: {e}")
    
    return tokenizer, train_dataset, eval_dataset

def verify_tokenizer_setup(tokenizer, sample_texts=None):
    """Encoder-Decoder 토크나이저 검증 함수"""
    print("\n🔍 Encoder-Decoder Tokenizer verification:")
    
    if sample_texts is None:
        sample_texts = [
            ("solve: 2 + 2 = ?", "4"),
            ("strategy: Is the sky blue?", "Yes"),
            ("reason: All birds fly. Question: Do penguins fly? A) Yes B) No", "B"),
            ("infer: Premise: It's sunny. Hypothesis: It's bright.", "entailment")
        ]
    
    for input_text, target_text in sample_texts:
        print(f"\n   Testing Encoder-Decoder: '{input_text[:40]}...' -> '{target_text}'")
        
        # Source (encoder) 토크나이징
        src_inputs = tokenizer(
            input_text, 
            max_length=128, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Target (decoder) 토크나이징
        with tokenizer.as_target_tokenizer():
            tgt_inputs = tokenizer(
                target_text,
                max_length=32,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        # Target labels 처리
        labels = tgt_inputs.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        print(f"      Source input shape: {src_inputs.input_ids.shape}")
        print(f"      Target labels shape: {labels.shape}")
        print(f"      Source tokens: {(src_inputs.attention_mask == 1).sum().item()}")
        print(f"      Target tokens: {(labels != -100).sum().item()}")
        
        # 디코딩 확인
        decoded_src = tokenizer.decode(src_inputs.input_ids[0], skip_special_tokens=True)
        valid_labels = labels[0][labels[0] != -100]
        if len(valid_labels) > 0:
            decoded_tgt = tokenizer.decode(valid_labels, skip_special_tokens=True)
            print(f"      Decoded source: '{decoded_src}'")
            print(f"      Decoded target: '{decoded_tgt}'")
            
    print("\n✅ Encoder-Decoder tokenizer verification completed")