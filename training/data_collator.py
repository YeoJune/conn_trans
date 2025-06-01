# training/data_collator.py
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class T5DataCollator:
    """T5 Encoder-Decoder용 데이터 콜레이터"""
    tokenizer: Any
    padding: Union[bool, str] = True
    max_length: int = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            return {}
        
        batch = {}
        tensor_keys = ['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels']
        
        # 텐서 필드 처리
        for key in tensor_keys:
            if key in features[0]:
                tensors = [f[key] for f in features if torch.is_tensor(f[key])]
                if tensors:
                    pad_value = -100 if key == 'labels' else self.tokenizer.pad_token_id
                    batch[key] = self._pad_tensors(tensors, pad_value)
        
        # 텍스트 필드 처리
        text_keys = ['target_text', 'question', 'problem', 'premise', 'hypothesis']
        for key in text_keys:
            if key in features[0]:
                batch[key] = [f.get(key, '') for f in features]
        
        return batch
    
    def _pad_tensors(self, tensors: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
        max_length = max(t.size(-1) for t in tensors)
        padded = []
        
        for tensor in tensors:
            if tensor.size(-1) < max_length:
                padding = torch.full((max_length - tensor.size(-1),), pad_value, dtype=tensor.dtype)
                padded_tensor = torch.cat([tensor, padding])
            else:
                padded_tensor = tensor[:max_length]
            padded.append(padded_tensor)
        
        return torch.stack(padded)
