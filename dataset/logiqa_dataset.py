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
        # 🔧 FIX: 올바른 필드명 사용
        context = item.get('context', '').strip()
        question = item.get('query', item.get('question', '')).strip()  # query가 정확한 필드명
        options = item.get('options', item.get('choices', []))
        answer = item.get('correct_option', item.get('answer', item.get('label', 0)))  # correct_option이 정확한 필드명
        
        # Build input text
        input_parts = [f"{self.task_prefix}:"]
        
        if context:
            input_parts.append(f"Context: {context}")
        
        input_parts.append(f"Question: {question}")
        
        if options:
            options_text = " ".join([f"{chr(65+i)}) {opt.strip()}" 
                                   for i, opt in enumerate(options)])
            input_parts.append(f"Options: {options_text}")
        
        # 🔧 FIX: 정답 처리 개선
        if isinstance(answer, int) and 0 <= answer < len(options):
            target_text = chr(65 + answer)  # 0->A, 1->B, etc.
        elif isinstance(answer, str) and len(answer) == 1 and answer.upper() in 'ABCD':
            target_text = answer.upper()
        else:
            # 🚨 디버깅: 예상치 못한 답변 형식 로깅
            print(f"⚠️ LogiQA item {idx}: unexpected answer format: {answer} (type: {type(answer)})")
            target_text = "A"  # 기본값
        
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
    
    def _is_valid_item(self, item):
        """LogiQA 특화 검증"""
        base_valid = super()._is_valid_item(item)
        if not base_valid:
            return False
        
        # 추가 검증: 옵션과 정답이 유효한지 확인
        metadata = item.get('metadata', {})
        options = metadata.get('options', [])
        original_answer = metadata.get('original_answer')
        
        # 옵션이 2개 이상 있어야 함
        if len(options) < 2:
            return False
        
        # 정답이 유효한 범위에 있어야 함
        if isinstance(original_answer, int) and not (0 <= original_answer < len(options)):
            return False
            
        return True