# dataset/gsm8k_dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import re

class GSM8KDataset(Dataset):
    """GSM8K Dataset - 수학 문제 해결 (검증 강화)"""
    
    def __init__(self, tokenizer, config, split="train"):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_seq_len
        self.answer_max_length = getattr(config, 'answer_max_length', 64)
        self.task_prefix = getattr(config, 'task_prefix', 'solve')
        
        print(f"📦 Loading GSM8K dataset ({split} split)...")
        
        try:
            # 검증된 GSM8K 로드
            dataset = load_dataset("gsm8k", "main", split=split)
            print(f"✅ Successfully loaded GSM8K from gsm8k/main")
            print(f"   Raw dataset size: {len(dataset)}")
        except Exception as e:
            print(f"❌ GSM8K loading failed: {e}")
            raise RuntimeError("Failed to load GSM8K dataset")
        
        # 전처리 및 검증
        self.data = self._preprocess_and_validate(dataset)
        print(f"GSM8K {split}: {len(self.data)} examples (after validation)")
    
    def _preprocess_and_validate(self, dataset):
        """전처리 및 데이터 검증"""
        processed = []
        skipped = 0
        
        for i, item in enumerate(dataset):
            try:
                # 문제와 답 추출
                problem = item['question'].strip()
                answer_text = item['answer'].strip()
                
                # 입력 텍스트 구성
                input_text = f"{self.task_prefix}: {problem}"
                
                # 최종 답 추출
                final_answer = self._extract_clean_answer(answer_text)
                
                # 검증: 비어있거나 너무 긴 데이터 건너뛰기
                if not problem or not final_answer:
                    skipped += 1
                    continue
                
                if len(problem) > 800:  # 너무 긴 문제
                    skipped += 1
                    continue
                
                # 답이 숫자인지 검증
                try:
                    float(final_answer)  # 숫자 변환 가능한지 확인
                except:
                    # 숫자가 아닌 경우 재시도
                    numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_text)
                    if numbers:
                        final_answer = numbers[-1]
                    else:
                        skipped += 1
                        continue
                
                processed.append({
                    'input_text': input_text,
                    'target_text': final_answer,
                    'full_solution': answer_text,
                    'problem': problem,
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
    
    def _extract_clean_answer(self, answer_text):
        """GSM8K에서 정확한 최종 답 추출"""
        # 1. "#### 24" 패턴 찾기 (GSM8K 표준)
        if "####" in answer_text:
            parts = answer_text.split("####")
            if len(parts) > 1:
                final_part = parts[-1].strip()
                # 숫자만 추출
                numbers = re.findall(r'-?\d+(?:\.\d+)?', final_part)
                if numbers:
                    return numbers[0]
        
        # 2. 문장 끝의 숫자 찾기
        sentences = answer_text.split('.')
        for sentence in reversed(sentences):
            numbers = re.findall(r'-?\d+(?:\.\d+)?', sentence)
            if numbers:
                return numbers[-1]
        
        # 3. 전체에서 마지막 숫자
        all_numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_text)
        if all_numbers:
            return all_numbers[-1]
        
        return "0"  # 기본값
    
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
                
            except Exception as e:
                print(f"         Tokenization: ❌ {e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """개선된 데이터 반환"""
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
            
            # 타겟 토크나이징 (T5 방식)
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
                'problem': item['problem']
            }
            
        except Exception as e:
            print(f"⚠️ Error in __getitem__ for index {idx}: {e}")
            # 안전한 기본값 반환
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.full((self.answer_max_length,), -100, dtype=torch.long),
                'target_text': "0",
                'problem': "Error loading problem"
            }