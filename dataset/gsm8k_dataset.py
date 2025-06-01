# dataset/gsm8k_dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import re

class GSM8KDataset(Dataset):
    """GSM8K Dataset - 수학 문제 해결 (T5 최적화)"""
    
    def __init__(self, tokenizer, config, split="train"):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_seq_len
        self.answer_max_length = getattr(config, 'answer_max_length', 64)
        self.task_prefix = getattr(config, 'task_prefix', 'solve')
        
        print(f"📦 Loading GSM8K dataset ({split} split)...")
        
        try:
            # HuggingFace에서 안정적인 GSM8K 로드
            dataset = load_dataset("gsm8k", "main", split=split)
            print(f"✅ Successfully loaded GSM8K from gsm8k/main")
        except Exception as e:
            print(f"❌ GSM8K loading failed: {e}")
            raise RuntimeError("Failed to load GSM8K dataset")
        
        self.data = self._preprocess(dataset)
        print(f"GSM8K {split}: {len(self.data)} examples")
    
    def _preprocess(self, dataset):
        """T5를 위한 적절한 데이터 전처리"""
        processed = []
        
        for item in dataset:
            # 입력: "solve: {문제}"
            problem = item['question'].strip()
            input_text = f"{self.task_prefix}: {problem}"
            
            # 출력: 최종 답만 추출 (GSM8K는 "#### 답" 형식)
            answer_text = item['answer'].strip()
            final_answer = self._extract_clean_answer(answer_text)
            
            processed.append({
                'input_text': input_text,
                'target_text': final_answer,
                'full_solution': answer_text,  # 전체 해답 보관
                'problem': problem
            })
        
        return processed
    
    def _extract_clean_answer(self, answer_text):
        """GSM8K에서 깔끔한 최종 답 추출"""
        # "#### 24" 패턴 찾기
        if "####" in answer_text:
            final_answer = answer_text.split("####")[-1].strip()
            # 숫자만 추출
            numbers = re.findall(r'-?\d+(?:\.\d+)?', final_answer)
            if numbers:
                return numbers[0]
        
        # 대체 방법: 마지막 숫자 찾기
        numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_text)
        if numbers:
            return numbers[-1]
        
        return "0"  # 기본값
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """T5용 정확한 입출력 생성"""
        item = self.data[idx]
        
        # 입력 토크나이징 (encoder)
        inputs = self.tokenizer(
            item['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # T5는 labels에서 pad_token_id를 -100으로 처리해야 함
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                item['target_text'],
                max_length=self.answer_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # -100으로 padding 토큰 마스킹 (CrossEntropyLoss가 무시)
        target_ids = targets.input_ids.squeeze()
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': target_ids,
            'target_text': item['target_text'],
            'problem': item['problem']
        }