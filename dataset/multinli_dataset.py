# dataset/multinli_dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class MultiNLIDataset(Dataset):
    """MultiNLI Dataset - 자연어 추론 (호환성 개선)"""
    
    def __init__(self, tokenizer, config, split="train"):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_seq_len
        self.answer_max_length = getattr(config, 'answer_max_length', 32)
        self.task_prefix = getattr(config, 'task_prefix', 'infer')
        
        print(f"📦 Loading MultiNLI dataset ({split} split)...")
        
        try:
            # 표준 MultiNLI 데이터셋
            if split == "validation":
                dataset = load_dataset("multi_nli", split="validation_matched")
                print(f"✅ Successfully loaded MultiNLI validation_matched")
            elif split == "test":
                dataset = load_dataset("multi_nli", split="test_matched") 
                print(f"✅ Successfully loaded MultiNLI test_matched")
            else:
                dataset = load_dataset("multi_nli", split=split)
                print(f"✅ Successfully loaded MultiNLI {split}")
            
            print(f"   Raw dataset size: {len(dataset):,}")
            
        except Exception as e:
            print(f"❌ MultiNLI loading failed: {e}")
            raise RuntimeError("Failed to load MultiNLI dataset")
        
        # 전처리 및 검증
        self.data = self._preprocess_and_validate(dataset)
        print(f"MultiNLI {split}: {len(self.data):,} examples (after validation)")
    
    def _preprocess_and_validate(self, dataset):
        """전처리 및 데이터 검증"""
        processed = []
        skipped = 0
        
        # 라벨 매핑
        label_map = {
            0: "entailment",
            1: "neutral", 
            2: "contradiction"
        }
        
        for i, item in enumerate(dataset):
            try:
                premise = item.get('premise', '').strip()
                hypothesis = item.get('hypothesis', '').strip()
                label = item.get('label', -1)
                
                # 검증: 빈 텍스트 건너뛰기
                if not premise or not hypothesis:
                    skipped += 1
                    continue
                
                # 너무 긴 텍스트 건너뛰기 (T5는 메모리 제약이 있음)
                total_length = len(premise) + len(hypothesis)
                if total_length > 1000:
                    skipped += 1
                    continue
                
                # 입력 구성
                input_text = f"{self.task_prefix}: Premise: {premise} Hypothesis: {hypothesis}"
                
                # 출력: entailment/neutral/contradiction
                if label in label_map:
                    target_text = label_map[label]
                else:
                    skipped += 1
                    continue  # 라벨이 없으면 스킵
                
                processed.append({
                    'input_text': input_text,
                    'target_text': target_text,
                    'premise': premise,
                    'hypothesis': hypothesis,
                    'original_label': label,
                    'genre': item.get('genre', 'unknown'),
                    'index': i
                })
                
            except Exception as e:
                print(f"   ⚠️ Error processing item {i}: {e}")
                skipped += 1
                continue
        
        if skipped > 0:
            print(f"   ⚠️ Skipped {skipped} invalid examples")
        
        # 샘플 검증
        if len(processed) > 0:
            self._validate_samples(processed[:3])
        
        return processed
    
    def _validate_samples(self, samples):
        """샘플 데이터 검증"""
        print(f"   🔍 Validating {len(samples)} samples:")
        
        for i, sample in enumerate(samples):
            print(f"      Sample {i+1}:")
            print(f"         Input: '{sample['input_text'][:60]}...'")
            print(f"         Target: '{sample['target_text']}'")
            
            # 토크나이징 테스트
            try:
                inputs = self.tokenizer(
                    sample['input_text'],
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                with self.tokenizer.as_target_tokenizer():
                    targets = self.tokenizer(
                        sample['target_text'],
                        max_length=self.answer_max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                
                labels = targets.input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                valid_tokens = (labels != -100).sum().item()
                print(f"         Tokenization: ✅ ({valid_tokens} valid tokens)")
                
                if valid_tokens == 0:
                    print(f"         ⚠️ WARNING: No valid tokens!")
                
            except Exception as e:
                print(f"         Tokenization: ❌ {e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """호환성 개선된 데이터 반환"""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        item = self.data[idx]
        
        try:
            # 입력 토크나이징
            inputs = self.tokenizer(
                item['input_text'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # T5 타겟 토크나이징
            with self.tokenizer.as_target_tokenizer():
                targets = self.tokenizer(
                    item['target_text'],
                    max_length=self.answer_max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
            
            # Labels 처리: -100으로 마스킹
            labels = targets.input_ids.squeeze().clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                'input_ids': inputs.input_ids.squeeze(),
                'attention_mask': inputs.attention_mask.squeeze(),
                'labels': labels,
                'target_text': item['target_text'],
                'premise': item['premise'],
                'hypothesis': item['hypothesis'],
                'genre': item['genre']
            }
            
        except Exception as e:
            print(f"⚠️ Error in __getitem__ for index {idx}: {e}")
            # 안전한 기본값 반환
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.full((self.answer_max_length,), -100, dtype=torch.long),
                'target_text': "neutral",
                'premise': "Error loading premise",
                'hypothesis': "Error loading hypothesis",
                'genre': "unknown"
            }