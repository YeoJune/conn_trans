# dataset/base_dataset.py
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class BaseReasoningDataset(Dataset, ABC):
    """Base class for all reasoning datasets"""
    
    def __init__(self, tokenizer, config, split="train"):
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.max_length = config.max_seq_len
        self.answer_max_length = getattr(config, 'answer_max_length', 32)
        self.task_prefix = getattr(config, 'task_prefix', 'answer')
        
        print(f"📦 Loading {self.dataset_name} ({split})...")
        
        # Load and preprocess data
        raw_data = self._load_raw_data()
        self.data = self._preprocess_data(raw_data)
        
        print(f"✅ {self.dataset_name} {split}: {len(self.data)} examples")
        
        # Quick validation
        if len(self.data) > 0:
            self._validate_samples(self.data[:2])
    
    @property
    @abstractmethod
    def dataset_name(self):
        """Dataset name for logging"""
        pass
    
    @abstractmethod
    def _load_raw_data(self):
        """Load raw dataset from source"""
        pass
    
    @abstractmethod
    def _process_item(self, item, idx):
        """Process single item into standard format
        Returns: dict with 'input_text', 'target_text', and metadata
        """
        pass
    
    def _preprocess_data(self, raw_data):
        """Preprocess raw data with validation"""
        processed = []
        skipped = 0
        
        for idx, item in enumerate(raw_data):
            try:
                processed_item = self._process_item(item, idx)
                if self._is_valid_item(processed_item):
                    processed.append(processed_item)
                else:
                    skipped += 1
            except Exception as e:
                print(f"⚠️ Error processing item {idx}: {str(e)[:50]}...")
                skipped += 1
        
        if skipped > 0:
            print(f"   Skipped {skipped} invalid examples")
        
        return processed
    
    def _is_valid_item(self, item):
        """Validate processed item"""
        return (
            item.get('input_text', '').strip() and 
            item.get('target_text', '').strip() and
            len(item['input_text']) <= 1000
        )
    
    def _validate_samples(self, samples):
        """Quick tokenization test"""
        print(f"   🔍 Testing tokenization on {len(samples)} samples...")
        
        for i, sample in enumerate(samples):
            try:
                self._tokenize_item(sample)
                print(f"      Sample {i+1}: ✅")
            except Exception as e:
                print(f"      Sample {i+1}: ❌ {e}")
                break
    
    def _tokenize_item(self, item):
        """Tokenize single item"""
        # Source tokenization
        src_inputs = self.tokenizer(
            item['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Target tokenization
        with self.tokenizer.as_target_tokenizer():
            tgt_inputs = self.tokenizer(
                item['target_text'],
                max_length=self.answer_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Create labels
        labels = tgt_inputs.input_ids.squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': src_inputs.input_ids.squeeze(),
            'attention_mask': src_inputs.attention_mask.squeeze(),
            'decoder_input_ids': tgt_inputs.input_ids.squeeze(),
            'decoder_attention_mask': tgt_inputs.attention_mask.squeeze(),
            'labels': labels,
            'target_text': item['target_text']
        }
    
    def _get_fallback_item(self):
        """Fallback item for errors"""
        return {
            'input_ids': torch.zeros(self.max_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
            'decoder_input_ids': torch.zeros(self.answer_max_length, dtype=torch.long),
            'decoder_attention_mask': torch.zeros(self.answer_max_length, dtype=torch.long),
            'labels': torch.full((self.answer_max_length,), -100, dtype=torch.long),
            'target_text': self._get_default_answer()
        }
    
    def _get_default_answer(self):
        """Default answer for this dataset type"""
        return "error"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range")
        
        try:
            result = self._tokenize_item(self.data[idx])
            # Add dataset-specific metadata
            result.update(self.data[idx].get('metadata', {}))
            return result
        except Exception as e:
            print(f"⚠️ Error in __getitem__[{idx}]: {e}")
            return self._get_fallback_item()