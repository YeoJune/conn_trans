# dataset/logiqa_dataset.py
from datasets import load_dataset
from .base_dataset import BaseReasoningDataset

class LogiQADataset(BaseReasoningDataset):
    @property
    def dataset_name(self):
        return "LogiQA"
    
    def _load_raw_data(self):
        # Try multiple sources
        sources = ["lucasmccabe/logiqa", "logiqa"]
        
        for source in sources:
            try:
                return load_dataset(source, split=self.split)
            except Exception:
                continue
        
        raise RuntimeError("Failed to load LogiQA from any source")
    
    def _process_item(self, item, idx):
        # Normalize field names
        context = item.get('context', item.get('passage', '')).strip()
        question = item.get('question', item.get('query', '')).strip()
        options = item.get('options', item.get('choices', []))
        answer = item.get('answer', item.get('label', 0))
        
        # Build input text
        input_parts = [f"{self.task_prefix}:"]
        
        if context:
            input_parts.append(f"Context: {context}")
        
        input_parts.append(f"Question: {question}")
        
        if options:
            options_text = " ".join([f"{chr(65+i)}) {opt.strip()}" 
                                   for i, opt in enumerate(options)])
            input_parts.append(f"Options: {options_text}")
        
        # Convert answer to letter
        if isinstance(answer, int) and 0 <= answer < len(options):
            target_text = chr(65 + answer)  # 0->A, 1->B, etc.
        elif isinstance(answer, str) and answer.upper() in 'ABCD':
            target_text = answer.upper()
        else:
            target_text = "A"
        
        return {
            'input_text': " ".join(input_parts),
            'target_text': target_text,
            'metadata': {
                'question': question,
                'context': context,
                'options': options,
                'original_answer': answer,
                'index': idx
            }
        }
    
    def _get_default_answer(self):
        return "A"
