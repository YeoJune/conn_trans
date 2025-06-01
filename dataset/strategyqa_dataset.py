# dataset/strategyqa_dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class StrategyQADataset(Dataset):
    """StrategyQA Dataset - Yes/No 질문 (T5 최적화)"""
    
    def __init__(self, tokenizer, config, split="train"):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_seq_len
        self.answer_max_length = getattr(config, 'answer_max_length', 16)
        self.task_prefix = getattr(config, 'task_prefix', 'strategy')
        
        print(f"📦 Loading StrategyQA dataset ({split} split)...")
        
        # 검증된 데이터셋 사용
        try:
            # wics/strategy-qa가 가장 안정적임
            dataset = load_dataset("wics/strategy-qa", split="test")  # test split만 있음
            print(f"✅ Successfully loaded from wics/strategy-qa")
            
            # train/eval 분할
            if split == "train":
                # 처음 80%를 train으로
                dataset = dataset.select(range(int(len(dataset) * 0.8)))
            else:
                # 나머지 20%를 eval로
                dataset = dataset.select(range(int(len(dataset) * 0.8), len(dataset)))
                
        except Exception as e:
            print(f"❌ StrategyQA loading failed: {e}")
            raise RuntimeError("Failed to load StrategyQA dataset")
        
        self.data = self._preprocess(dataset)
        print(f"StrategyQA {split}: {len(self.data)} examples")
    
    def _preprocess(self, dataset):
        """T5 적합한 전처리"""
        processed = []
        
        for item in dataset:
            # 입력: "strategy: {질문}"
            question = item['question'].strip()
            input_text = f"{self.task_prefix}: {question}"
            
            # 출력: "Yes" 또는 "No"
            answer = item['answer']
            target_text = "Yes" if answer else "No"
            
            processed.append({
                'input_text': input_text,
                'target_text': target_text,
                'question': question,
                'original_answer': answer
            })
        
        return processed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """T5용 정확한 데이터 생성"""
        item = self.data[idx]
        
        # 입력 토크나이징
        inputs = self.tokenizer(
            item['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # T5 target 토크나이징 (as_target_tokenizer 사용)
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                item['target_text'],
                max_length=self.answer_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Labels에서 padding을 -100으로 변경
        target_ids = targets.input_ids.squeeze()
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': target_ids,
            'target_text': item['target_text'],
            'question': item['question']
        }