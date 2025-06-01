# dataset/tokenizer_utils.py

from transformers import T5Tokenizer
from .logiqa_dataset import LogiQADataset
from .gsm8k_dataset import GSM8KDataset
from .strategyqa_dataset import StrategyQADataset
from .multinli_dataset import MultiNLIDataset

def get_tokenizer_and_dataset(dataset_name, config):
    """
    최신 T5 토크나이저와 데이터셋 로딩
    
    핵심 수정사항:
    1. 최신 T5 토크나이저 사용법 확인
    2. as_target_tokenizer() 여전히 유효 (2024년 기준)
    3. DataCollatorForSeq2Seq와 호환성 확보
    """
    
    print(f"🔄 Loading T5 tokenizer: {config.tokenizer_name}")
    
    try:
        # 최신 T5 토크나이저 로딩
        tokenizer = T5Tokenizer.from_pretrained(
            config.tokenizer_name,
            legacy=False  # 최신 방식 사용
        )
        print(f"✅ T5Tokenizer loaded successfully")
    except Exception as e:
        print(f"⚠️ Latest tokenizer failed, using fallback: {e}")
        tokenizer = T5Tokenizer.from_pretrained(config.tokenizer_name)
    
    # T5 토크나이저 정보 출력
    print(f"✅ Tokenizer info:")
    print(f"   - Vocab size: {tokenizer.vocab_size:,}")
    print(f"   - Pad token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
    print(f"   - EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
    print(f"   - Extra IDs: {getattr(tokenizer, 'extra_ids', 100)} (for T5 special tokens)")
    
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
    
    # 🔍 데이터 샘플 확인 (중요!)
    print(f"\n🔍 Data sample verification:")
    try:
        sample = train_dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Input shape: {sample['input_ids'].shape}")
        print(f"   Labels shape: {sample['labels'].shape}")
        
        # 실제 텍스트 확인
        input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        print(f"   Input text: '{input_text[:80]}...'")
        print(f"   Target text: '{sample['target_text']}'")
        
        # Labels 검증 (중요!)
        labels = sample['labels']
        mask_count = (labels == -100).sum().item()
        valid_count = (labels != -100).sum().item()
        print(f"   Labels: {valid_count} valid tokens, {mask_count} masked tokens")
        
        # 잠재적 문제 감지
        if mask_count == 0:
            print("   ⚠️ WARNING: No masked tokens! This will cause training issues.")
        if valid_count == 0:
            print("   ⚠️ WARNING: No valid tokens! This will cause training issues.")
        if valid_count < 2:
            print("   ⚠️ WARNING: Very few valid tokens! Consider longer targets.")
        
        # 토큰 ID 분포 확인
        unique_tokens = torch.unique(labels[labels != -100])
        print(f"   Unique token IDs in labels: {len(unique_tokens)} (sample: {unique_tokens[:5].tolist()})")
        
        if len(unique_tokens) < 2:
            print("   ⚠️ WARNING: Very few unique tokens in labels!")
            
    except Exception as e:
        print(f"   ❌ Sample verification failed: {e}")
        print("   This might indicate a data preprocessing issue.")
    
    return tokenizer, train_dataset, eval_dataset

def verify_tokenizer_setup(tokenizer, sample_texts=None):
    """토크나이저 설정 검증 함수"""
    print("\n🔍 Tokenizer verification:")
    
    if sample_texts is None:
        sample_texts = [
            ("solve: 2 + 2 = ?", "4"),
            ("strategy: Is the sky blue?", "Yes"),
            ("reason: All birds fly. Question: Do penguins fly? A) Yes B) No", "B"),
            ("infer: Premise: It's sunny. Hypothesis: It's bright.", "entailment")
        ]
    
    for input_text, target_text in sample_texts:
        print(f"\n   Testing: '{input_text[:40]}...' -> '{target_text}'")
        
        # 입력 토크나이징
        inputs = tokenizer(
            input_text, 
            max_length=128, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        # 타겟 토크나이징 (as_target_tokenizer 사용)
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                target_text,
                max_length=32,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        # Labels 처리
        labels = targets.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        print(f"      Input IDs shape: {inputs.input_ids.shape}")
        print(f"      Labels shape: {labels.shape}")
        print(f"      Masked tokens: {(labels == -100).sum().item()}")
        print(f"      Valid tokens: {(labels != -100).sum().item()}")
        
        # 디코딩 확인
        decoded_input = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        valid_labels = labels[0][labels[0] != -100]
        if len(valid_labels) > 0:
            decoded_target = tokenizer.decode(valid_labels, skip_special_tokens=True)
            print(f"      Decoded input: '{decoded_input}'")
            print(f"      Decoded target: '{decoded_target}'")
        else:
            print(f"      ⚠️ No valid tokens to decode!")
            
    print("\n✅ Tokenizer verification completed")