# dataset/logiqa_dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class LogiQADataset(Dataset):
    """LogiQA Dataset - 논리적 추론을 위한 다중 선택 문제"""
    
    def __init__(self, tokenizer, config, split="train"):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_seq_len
        self.answer_max_length = getattr(config, 'answer_max_length', 64)
        self.task_prefix = getattr(config, 'task_prefix', 'reason')
        
        # 검증된 LogiQA 소스 사용
        print(f"📦 Loading LogiQA dataset ({split} split)...")
        
        dataset = None
        successful_source = None
        
        # 여러 소스 시도 (우선순위 순)
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
                print(f"⚠️ {source} failed: {str(e)[:100]}...")
                continue
        
        if dataset is None:
            print(f"❌ All LogiQA sources failed")
            raise RuntimeError("Failed to load LogiQA dataset. Please check your internet connection and try again.")
        
        self.successful_source = successful_source
        self.data = self._preprocess(dataset)
        print(f"LogiQA {split}: {len(self.data)} examples from {successful_source}")
    
    def _preprocess(self, dataset):
        """데이터셋을 T5 형식으로 전처리"""
        processed = []
        
        for item in dataset:
            # 필드명 정규화 (소스에 따라 다를 수 있음)
            context = item.get('context', item.get('passage', ''))
            question = item.get('question', item.get('query', ''))
            options = item.get('options', item.get('choices', []))
            answer = item.get('answer', item.get('label', 0))
            
            # 빈 필드 처리
            if not context:
                context = "No context provided."
            if not question:
                question = "No question provided."
            
            # T5 형식: "reason: <context> question: <question> options: <options>"
            if options and len(options) > 0:
                options_text = " ".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
                input_text = f"{self.task_prefix}: {context} question: {question} options: {options_text}"
            else:
                input_text = f"{self.task_prefix}: {context} question: {question}"
            
            # 답변 처리 (선택지 인덱스를 문자로 변환)
            if isinstance(answer, int) and options and 0 <= answer < len(options):
                target_text = chr(65 + answer)  # 0->A, 1->B, 2->C, 3->D
            elif isinstance(answer, str) and answer.upper() in ['A', 'B', 'C', 'D']:
                target_text = answer.upper()
            else:
                target_text = str(answer)
            
            processed.append({
                'input_text': input_text,
                'target_text': target_text,
                'original_answer': answer,
                'context': context,
                'question': question,
                'options': options
            })
        
        return processed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """T5 모델에 맞는 전처리 개선"""
        item = self.data[idx]
        
        # 입력 토크나이징 (T5 encoder 입력)
        inputs = self.tokenizer(
            item['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 타겟 토크나이징 (T5 decoder 출력)
        # T5에서는 decoder_input_ids가 labels보다 1 짧음 (<pad> 토큰으로 시작)
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
            'original_answer': item['original_answer']
        }
