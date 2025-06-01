# dataset/logiqa_dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class LogiQADataset(Dataset):
    """LogiQA Dataset - 논리 추론 (호환성 개선)"""
    
    def __init__(self, tokenizer, config, split="train"):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_seq_len
        self.answer_max_length = getattr(config, 'answer_max_length', 16)
        self.task_prefix = getattr(config, 'task_prefix', 'reason')
        
        print(f"📦 Loading LogiQA dataset ({split} split)...")
        
        try:
            # 검증된 LogiQA 소스들 시도
            dataset = None
            successful_source = None
            
            sources_to_try = [
                "lucasmccabe/logiqa",
                "logiqa"
            ]
            
            for source in sources_to_try:
                try:
                    dataset = load_dataset(source, split=split)
                    successful_source = source
                    print(f"✅ Successfully loaded from {source}")
                    break
                except Exception as e:
                    print(f"⚠️ {source} failed: {str(e)[:50]}...")
                    continue
            
            if dataset is None:
                raise RuntimeError("All LogiQA sources failed")
                
            print(f"   Raw dataset size: {len(dataset)}")
            
        except Exception as e:
            print(f"❌ LogiQA loading failed: {e}")
            raise RuntimeError("Failed to load LogiQA dataset")
        
        # 전처리 및 검증
        self.data = self._preprocess_and_validate(dataset)
        print(f"LogiQA {split}: {len(self.data)} examples (after validation)")
    
    def _preprocess_and_validate(self, dataset):
        """전처리 및 데이터 검증"""
        processed = []
        skipped = 0
        
        for i, item in enumerate(dataset):
            try:
                # 필드 정규화
                context = item.get('context', item.get('passage', '')).strip()
                question = item.get('question', item.get('query', '')).strip()
                options = item.get('options', item.get('choices', []))
                answer = item.get('answer', item.get('label', 0))
                
                # 검증: 필수 필드 확인
                if not question:
                    skipped += 1
                    continue
                
                # 너무 긴 텍스트 건너뛰기
                total_length = len(context) + len(question)
                if total_length > 800:
                    skipped += 1
                    continue
                
                # 입력 구성
                if options and len(options) > 0:
                    options_text = " ".join([f"{chr(65+i)}) {opt.strip()}" for i, opt in enumerate(options)])
                    if context:
                        input_text = f"{self.task_prefix}: {context} Question: {question} Options: {options_text}"
                    else:
                        input_text = f"{self.task_prefix}: Question: {question} Options: {options_text}"
                else:
                    if context:
                        input_text = f"{self.task_prefix}: {context} Question: {question}"
                    else:
                        input_text = f"{self.task_prefix}: Question: {question}"
                
                # 출력: 선택지 문자 (A, B, C, D)
                if isinstance(answer, int) and options and 0 <= answer < len(options):
                    target_text = chr(65 + answer)  # 0->A, 1->B, etc.
                elif isinstance(answer, str) and answer.upper() in ['A', 'B', 'C', 'D']:
                    target_text = answer.upper()
                else:
                    target_text = "A"  # 기본값
                
                processed.append({
                    'input_text': input_text,
                    'target_text': target_text,
                    'context': context,
                    'question': question,
                    'options': options,
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
                'input_ids': src_inputs.input_ids.squeeze(),
                'attention_mask': src_inputs.attention_mask.squeeze(),
                'decoder_input_ids': tgt_inputs.input_ids.squeeze(),
                'decoder_attention_mask': tgt_inputs.attention_mask.squeeze(),
                'labels': labels,
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
                'target_text': "A",
                'question': "Error loading question"
            }
