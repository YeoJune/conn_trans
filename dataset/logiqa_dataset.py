# dataset/logiqa_dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class LogiQADataset(Dataset):
    """LogiQA Dataset - 논리 추론 (T5 최적화)"""
    
    def __init__(self, tokenizer, config, split="train"):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_seq_len
        self.answer_max_length = getattr(config, 'answer_max_length', 16)
        self.task_prefix = getattr(config, 'task_prefix', 'reason')
        
        print(f"📦 Loading LogiQA dataset ({split} split)...")
        
        try:
            # 안정적인 LogiQA 소스 시도
            dataset = load_dataset("lucasmccabe/logiqa", split=split)
            print(f"✅ Successfully loaded from lucasmccabe/logiqa")
        except Exception as e:
            try:
                # 대체 소스
                dataset = load_dataset("logiqa", split=split)
                print(f"✅ Successfully loaded from logiqa")
            except Exception as e2:
                print(f"❌ LogiQA loading failed: {e}, {e2}")
                raise RuntimeError("Failed to load LogiQA dataset")
        
        self.data = self._preprocess(dataset)
        print(f"LogiQA {split}: {len(self.data)} examples")
    
    def _preprocess(self, dataset):
        """T5 적합한 전처리"""
        processed = []
        
        for item in dataset:
            # 필드 정규화
            context = item.get('context', item.get('passage', '')).strip()
            question = item.get('question', item.get('query', '')).strip()
            options = item.get('options', item.get('choices', []))
            answer = item.get('answer', item.get('label', 0))
            
            # 빈 데이터 처리
            if not context:
                context = ""
            if not question:
                continue  # 질문이 없으면 스킵
            
            # 입력 구성: "reason: {context} Question: {question} Options: A) ... B) ..."
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
        
        # T5 target 토크나이징
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                item['target_text'],
                max_length=self.answer_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Labels 처리
        target_ids = targets.input_ids.squeeze()
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': target_ids,
            'target_text': item['target_text'],
            'question': item['question']
        }