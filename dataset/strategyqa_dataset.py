# dataset/strategyqa_dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class StrategyQADataset(Dataset):
    """StrategyQA Dataset - Yes/No 질문 (호환성 개선)"""
    
    def __init__(self, tokenizer, config, split="train"):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_seq_len
        self.answer_max_length = getattr(config, 'answer_max_length', 16)
        self.task_prefix = getattr(config, 'task_prefix', 'strategy')
        
        print(f"📦 Loading StrategyQA dataset ({split} split)...")
        
        # 검증된 데이터셋 사용
        try:
            # wics/strategy-qa가 가장 안정적이고 test split만 있음
            dataset = load_dataset("wics/strategy-qa", split="test")
            print(f"✅ Successfully loaded from wics/strategy-qa")
            print(f"   Raw dataset size: {len(dataset)}")
            
            # train/eval 분할 (dataset이 test만 있으므로)
            total_size = len(dataset)
            if split == "train":
                # 처음 80%를 train으로
                end_idx = int(total_size * 0.8)
                dataset = dataset.select(range(end_idx))
            else:
                # 나머지 20%를 eval로
                start_idx = int(total_size * 0.8)
                dataset = dataset.select(range(start_idx, total_size))
                
        except Exception as e:
            print(f"❌ StrategyQA loading failed: {e}")
            raise RuntimeError("Failed to load StrategyQA dataset")
        
        # 전처리 및 검증
        self.data = self._preprocess_and_validate(dataset)
        print(f"StrategyQA {split}: {len(self.data)} examples (after validation)")
    
    def _preprocess_and_validate(self, dataset):
        """전처리 및 데이터 검증"""
        processed = []
        skipped = 0
        
        for i, item in enumerate(dataset):
            try:
                # 필드 추출
                question = item['question'].strip()
                answer = item['answer']  # boolean 값
                
                # 검증: 빈 질문 건너뛰기
                if not question:
                    skipped += 1
                    continue
                
                # 너무 긴 질문 건너뛰기
                if len(question) > 400:
                    skipped += 1
                    continue
                
                # 입력 구성
                input_text = f"{self.task_prefix}: {question}"
                
                # 출력: "Yes" 또는 "No"
                target_text = "Yes" if answer else "No"
                
                processed.append({
                    'input_text': input_text,
                    'target_text': target_text,
                    'question': question,
                    'original_answer': answer,
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
        """Encoder-Decoder 호환 데이터 반환"""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")

        item = self.data[idx]

        try:
            # Source (encoder) 입력 토크나이징
            src_inputs = self.tokenizer(
                item['input_text'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Target (decoder) 토크나이징
            with self.tokenizer.as_target_tokenizer():
                tgt_inputs = self.tokenizer(
                    item['target_text'],
                    max_length=self.answer_max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
            
            # Target labels 처리
            labels = tgt_inputs.input_ids.squeeze().clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                'input_ids': src_inputs.input_ids.squeeze(),           # Source for encoder
                'attention_mask': src_inputs.attention_mask.squeeze(), # Source attention mask
                'decoder_input_ids': tgt_inputs.input_ids.squeeze(),   # Target input for decoder
                'decoder_attention_mask': tgt_inputs.attention_mask.squeeze(), # Target attention mask
                'labels': labels,                                      # Target labels for loss
                'target_text': item['target_text'],
                'question': item['question']
            }
            
        except Exception as e:
            print(f"⚠️ Error in __getitem__ for index {idx}: {e}")
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'decoder_input_ids': torch.zeros(self.answer_max_length, dtype=torch.long),
                'decoder_attention_mask': torch.zeros(self.answer_max_length, dtype=torch.long),
                'labels': torch.full((self.answer_max_length,), -100, dtype=torch.long),
                'target_text': "No",
                'question': "Error loading question"
            }