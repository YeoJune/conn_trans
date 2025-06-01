# training/data_collator.py
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class T5DataCollator:
    """Improved T5 data collator with better error handling"""
    
    tokenizer: Any
    max_length: int = 512
    padding: Union[bool, str] = True
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            return {}
        
        batch = {}
        
        # Process tensor fields
        tensor_fields = ['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels']
        
        for field in tensor_fields:
            if field in features[0]:
                tensors = [f[field] for f in features if torch.is_tensor(f[field])]
                if tensors:
                    pad_value = -100 if field == 'labels' else self.tokenizer.pad_token_id
                    batch[field] = self._pad_sequence(tensors, pad_value)
        
        # Process text fields
        text_fields = ['target_text', 'question', 'problem', 'premise', 'hypothesis', 'context']
        
        for field in text_fields:
            if field in features[0]:
                batch[field] = [f.get(field, '') for f in features]
        
        return batch
    
    def _pad_sequence(self, tensors: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
        """Pad tensor sequence to same length"""
        if not tensors:
            return torch.empty(0)
        
        # Find max length
        max_len = max(t.size(-1) for t in tensors)
        max_len = min(max_len, self.max_length)  # Respect max_length
        
        padded = []
        for tensor in tensors:
            current_len = tensor.size(-1)
            
            if current_len > max_len:
                # Truncate if too long
                padded_tensor = tensor[:max_len]
            elif current_len < max_len:
                # Pad if too short
                padding_size = max_len - current_len
                padding = torch.full((padding_size,), pad_value, dtype=tensor.dtype, device=tensor.device)
                padded_tensor = torch.cat([tensor, padding])
            else:
                padded_tensor = tensor
            
            padded.append(padded_tensor)
        
        return torch.stack(padded)
    