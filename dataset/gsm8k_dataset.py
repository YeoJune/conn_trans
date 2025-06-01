# dataset/gsm8k_dataset.py
import re
from datasets import load_dataset
from .base_dataset import BaseReasoningDataset

class GSM8KDataset(BaseReasoningDataset):
    @property
    def dataset_name(self):
        return "GSM8K"
    
    def _load_raw_data(self):
        return load_dataset("gsm8k", "main", split=self.split)
    
    def _process_item(self, item, idx):
        problem = item['question'].strip()
        answer_text = item['answer'].strip()
        
        # Extract final answer
        final_answer = self._extract_final_answer(answer_text)
        
        return {
            'input_text': f"{self.task_prefix}: {problem}",
            'target_text': final_answer,
            'metadata': {
                'problem': problem,
                'full_solution': answer_text,
                'index': idx
            }
        }
    
    def _extract_final_answer(self, answer_text):
        """Extract clean final answer from GSM8K solution"""
        # Look for "#### number" pattern
        if "####" in answer_text:
            parts = answer_text.split("####")
            if len(parts) > 1:
                numbers = re.findall(r'-?\d+(?:\.\d+)?', parts[-1])
                if numbers:
                    return numbers[0]
        
        # Fallback: last number in text
        numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_text)
        return numbers[-1] if numbers else "0"
    
    def _get_default_answer(self):
        return "0"
