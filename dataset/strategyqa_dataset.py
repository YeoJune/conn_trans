# dataset/strategyqa_dataset.py
from datasets import load_dataset
from .base_dataset import BaseReasoningDataset

class StrategyQADataset(BaseReasoningDataset):
    @property
    def dataset_name(self):
        return "StrategyQA"
    
    def _load_raw_data(self):
        # Load full dataset and split
        full_dataset = load_dataset("wics/strategy-qa", split="test")
        
        total_size = len(full_dataset)
        if self.split == "train":
            return full_dataset.select(range(int(total_size * 0.8)))
        else:
            return full_dataset.select(range(int(total_size * 0.8), total_size))
    
    def _process_item(self, item, idx):
        question = item['question'].strip()
        answer = item['answer']  # boolean
        
        return {
            'input_text': f"{self.task_prefix}: {question}",
            'target_text': "Yes" if answer else "No",
            'metadata': {
                'question': question,
                'original_answer': answer,
                'index': idx
            }
        }
    
    def _get_default_answer(self):
        return "No"
