# dataset/strategyqa_dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class StrategyQADataset(Dataset):
    """StrategyQA Dataset - 전략적 Yes/No 질문 답변"""
    
    def __init__(self, tokenizer, config, split="train"):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_seq_len
        self.answer_max_length = getattr(config, 'answer_max_length', 32)
        self.task_prefix = getattr(config, 'task_prefix', 'strategy')
        
        # 검증된 StrategyQA 소스들 시도
        print(f"📦 Loading StrategyQA dataset ({split} split)...")
        
        # 우선순위에 따른 소스 리스트
        sources_to_try = [
            "ChilleD/StrategyQA",
            "wics/strategy-qa", 
            "amydeng2000/strategy-qa"
        ]
        
        dataset = None
        successful_source = None
        
        for source in sources_to_try:
            try:
                dataset = load_dataset(source, split=split)
                successful_source = source
                print(f"✅ Successfully loaded from {source}")
                break
            except Exception as e:
                print(f"⚠️ {source} failed: {str(e)[:100]}...")
                continue
        
        if dataset is None:
            print(f"❌ All StrategyQA sources failed")
            raise RuntimeError("Failed to load StrategyQA dataset. Please check your internet connection and try again.")
        
        self.successful_source = successful_source
        self.data = self._preprocess(dataset)
        print(f"StrategyQA {split}: {len(self.data)} examples from {successful_source}")
    
    def _preprocess(self, dataset):
        """데이터셋을 T5 형식으로 전처리"""
        processed = []
        
        for item in dataset:
            # 필드명 정규화 (소스에 따라 다름)
            question = item.get('question', item.get('query', ''))
            answer = item.get('answer', item.get('label', False))
            
            # 빈 질문 처리
            if not question:
                question = "No question provided."
            
            # T5 형식: "strategy: <question>"
            input_text = f"{self.task_prefix}: {question}"
            
            # 답변 정규화 (Yes/No)
            target_text = self._normalize_answer(answer)
            
            processed.append({
                'input_text': input_text,
                'target_text': target_text,
                'original_answer': answer,
                'question': question
            })
        
        return processed
    
    def _normalize_answer(self, answer):
        """답변을 Yes/No로 정규화"""
        if isinstance(answer, bool):
            return "Yes" if answer else "No"
        elif isinstance(answer, int):
            return "Yes" if answer == 1 else "No"
        elif isinstance(answer, str):
            # 문자열인 경우 정규화
            answer_lower = answer.lower().strip()
            if answer_lower in ['true', '1', 'yes', 'y']:
                return "Yes"
            elif answer_lower in ['false', '0', 'no', 'n']:
                return "No"
            else:
                # 기타 문자열은 원본 유지하되 첫 글자만 대문자
                return answer.strip().capitalize()
        else:
            return str(answer)
    
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
            'question': item['question']
        }