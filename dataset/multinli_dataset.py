# dataset/multinli_dataset.py
from datasets import load_dataset
from .base_dataset import BaseReasoningDataset

class MultiNLIDataset(BaseReasoningDataset):
    
    LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    @property
    def dataset_name(self):
        return "MultiNLI"
    
    def _load_raw_data(self):
        if self.split == "validation":
            return load_dataset("multi_nli", split="validation_matched")
        elif self.split == "test":
            return load_dataset("multi_nli", split="test_matched")
        else:
            return load_dataset("multi_nli", split=self.split)
    
    def _process_item(self, item, idx):
        premise = item['premise'].strip()
        hypothesis = item['hypothesis'].strip()
        label = item['label']
        
        input_text = (f"{self.task_prefix}: "
                     f"Premise: {premise} "
                     f"Hypothesis: {hypothesis}")
        
        target_text = self.LABEL_MAP.get(label, "neutral")
        
        return {
            'input_text': input_text,
            'target_text': target_text,
            'metadata': {
                'premise': premise,
                'hypothesis': hypothesis,
                'genre': item.get('genre', 'unknown'),
                'original_label': label,
                'index': idx
            }
        }
    
    def _is_valid_item(self, item):
        """Override with stricter validation for large dataset"""
        return (
            super()._is_valid_item(item) and
            len(item['input_text']) <= 800  # Stricter limit
        )
    
    def _get_default_answer(self):
        return "neutral"
