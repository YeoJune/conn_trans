# training/data_collator.py - T5용 데이터 콜레이터
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class T5DataCollator:
    """
    T5에 특화된 데이터 콜레이터
    배치 크기 불일치 문제 해결
    """
    tokenizer: Any
    padding: Union[bool, str] = True
    max_length: int = None
    pad_to_multiple_of: int = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        배치 생성 (안전한 크기 처리)
        """
        if not features:
            return {}
        
        # 배치 크기 확인
        batch_size = len(features)
        
        # 텐서 필드들을 따로 처리
        tensor_keys = ['input_ids', 'attention_mask', 'labels']
        text_keys = ['target_text', 'question', 'problem', 'premise', 'hypothesis', 'genre']
        
        batch = {}
        
        # 텐서 필드 처리
        for key in tensor_keys:
            if key in features[0]:
                values = [f[key] for f in features]
                
                # 모든 값이 텐서인지 확인
                if all(torch.is_tensor(v) for v in values):
                    try:
                        # 패딩하여 같은 크기로 만들기
                        if key == 'labels':
                            # Labels는 -100으로 패딩
                            batch[key] = self._pad_tensors(values, pad_value=-100)
                        else:
                            # 나머지는 pad_token_id로 패딩
                            pad_value = getattr(self.tokenizer, 'pad_token_id', 0)
                            batch[key] = self._pad_tensors(values, pad_value=pad_value)
                    except Exception as e:
                        print(f"⚠️ Error padding {key}: {e}")
                        # 실패시 첫 번째 요소 크기로 맞추기
                        target_shape = values[0].shape
                        padded_values = []
                        for v in values:
                            if v.shape == target_shape:
                                padded_values.append(v)
                            else:
                                # 크기가 다르면 잘라내거나 패딩
                                if len(v) < len(values[0]):
                                    pad_size = len(values[0]) - len(v)
                                    if key == 'labels':
                                        v = torch.cat([v, torch.full((pad_size,), -100, dtype=v.dtype)])
                                    else:
                                        v = torch.cat([v, torch.full((pad_size,), pad_value, dtype=v.dtype)])
                                else:
                                    v = v[:len(values[0])]
                                padded_values.append(v)
                        batch[key] = torch.stack(padded_values)
        
        # 텍스트 필드 처리
        for key in text_keys:
            if key in features[0]:
                values = [f.get(key, '') for f in features]
                # 문자열 리스트로 저장 (텐서가 아님)
                batch[key] = values
        
        # 배치 크기 일관성 확인
        tensor_batch_sizes = []
        for key, value in batch.items():
            if torch.is_tensor(value):
                tensor_batch_sizes.append(value.size(0))
            elif isinstance(value, (list, tuple)):
                tensor_batch_sizes.append(len(value))
        
        if tensor_batch_sizes and not all(size == batch_size for size in tensor_batch_sizes):
            print(f"⚠️ Batch size inconsistency: expected {batch_size}, got {tensor_batch_sizes}")
        
        return batch
    
    def _pad_tensors(self, tensors: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
        """
        텐서들을 같은 크기로 패딩
        """
        if not tensors:
            return torch.tensor([])
        
        # 최대 길이 찾기
        max_length = max(t.size(-1) for t in tensors)
        
        # 패딩된 텐서들
        padded = []
        for tensor in tensors:
            if tensor.dim() == 1:
                # 1D 텐서인 경우
                current_length = tensor.size(0)
                if current_length < max_length:
                    padding = torch.full((max_length - current_length,), pad_value, dtype=tensor.dtype)
                    padded_tensor = torch.cat([tensor, padding])
                else:
                    padded_tensor = tensor[:max_length]
                padded.append(padded_tensor)
            else:
                # 다차원 텐서인 경우 (잘못된 경우이지만 안전하게 처리)
                padded.append(tensor.flatten()[:max_length])
        
        try:
            return torch.stack(padded)
        except Exception as e:
            print(f"⚠️ Stack error: {e}")
            # 실패시 첫 번째 텐서 크기로 맞추기
            target_size = padded[0].size()
            fixed_tensors = []
            for t in padded:
                if t.size() != target_size:
                    # 크기 조정
                    if t.numel() < target_size.numel():
                        # 부족하면 패딩
                        pad_size = target_size.numel() - t.numel()
                        t = torch.cat([t.flatten(), torch.full((pad_size,), pad_value, dtype=t.dtype)])
                    else:
                        # 넘치면 자르기
                        t = t.flatten()[:target_size.numel()]
                    t = t.view(target_size)
                fixed_tensors.append(t)
            return torch.stack(fixed_tensors)
