# dataset/gsm8k_dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import re

class GSM8KDataset(Dataset):
    """GSM8K Dataset - 초등학교 수학 문제"""
    
    def __init__(self, tokenizer, config, split="train"):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_seq_len
        self.answer_max_length = getattr(config, 'answer_max_length', 128)
        self.task_prefix = getattr(config, 'task_prefix', 'solve')
        
        # 검증된 GSM8K 소스 사용
        print(f"📦 Loading GSM8K dataset ({split} split)...")
        
        try:
            # "main" config 명시적 사용 (HuggingFace 문서에 따르면 필요)
            dataset = load_dataset("gsm8k", "main", split=split)
            print(f"✅ Successfully loaded GSM8K from gsm8k/main")
        except Exception as e:
            # 대체 방법 시도
            try:
                dataset = load_dataset("openai/gsm8k", "main", split=split)
                print(f"✅ Successfully loaded GSM8K from openai/gsm8k/main")
            except Exception as e2:
                print(f"❌ GSM8K loading failed")
                print(f"   First attempt: {e}")
                print(f"   Second attempt: {e2}")
                raise RuntimeError("Failed to load GSM8K dataset")
        
        self.data = self._preprocess(dataset)
        print(f"GSM8K {split}: {len(self.data)} examples")
    
    def _preprocess(self, dataset):
        """데이터셋을 T5 형식으로 전처리"""
        processed = []
        
        for item in dataset:
            # T5 형식: "solve: <problem>"
            problem = item['question'].strip()
            input_text = f"{self.task_prefix}: {problem}"
            
            # 답변에서 최종 숫자 추출 (GSM8K 형식: "#### 24")
            answer = item['answer'].strip()
            target_text = self._extract_final_answer(answer)
            
            processed.append({
                'input_text': input_text,
                'target_text': target_text,
                'original_answer': answer,
                'problem': problem
            })
        
        return processed
    
    def _extract_final_answer(self, answer_text):
        """답변에서 최종 숫자를 추출 (GSM8K 특화)"""
        # GSM8K 표준 형식: "#### 24" 
        if "####" in answer_text:
            final_answer = answer_text.split("####")[-1].strip()
        else:
            # 대체 방법: 마지막 숫자 추출
            numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_text)
            final_answer = numbers[-1] if numbers else answer_text.strip()
        
        # 불필요한 공백 제거
        return final_answer.strip()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 입력 토크나이징
        inputs = self.tokenizer(
            item['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 타겟 토크나이징
        targets = self.tokenizer(
            item['target_text'],
            max_length=self.answer_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': targets.input_ids.squeeze(),
            'target_text': item['target_text'],
            'original_answer': item['original_answer'],
            'problem': item['problem']
        }