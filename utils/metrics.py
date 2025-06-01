# utils/metrics.py
import re
import string
from typing import List, Optional

def normalize_answer(s: str) -> str:
    """답변 정규화"""
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())

def extract_final_answer(text: str, dataset_type: Optional[str] = None) -> str:
    """T5 답변에서 최종 답 추출"""
    if not text:
        return ""
    
    text = str(text).strip()
    first_word = text.split()[0] if text.split() else ""
    
    if dataset_type == "strategyqa":
        first_lower = first_word.lower()
        if first_lower.startswith('y') or first_lower in ['yes', 'true', '1']:
            return "Yes"
        elif first_lower.startswith('n') or first_lower in ['no', 'false', '0']:
            return "No"
        return first_word.capitalize()
    
    elif dataset_type == "gsm8k":
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        return numbers[0] if numbers else "0"
    
    elif dataset_type == "logiqa":
        first_upper = first_word.upper()
        if first_upper in ['A', 'B', 'C', 'D']:
            return first_upper
        clean = re.sub(r'[^\w]', '', first_upper)
        return clean if clean in ['A', 'B', 'C', 'D'] else "A"
    
    elif dataset_type == "multinli":
        first_lower = first_word.lower()
        if first_lower.startswith('ent'):
            return "entailment"
        elif first_lower.startswith('neu'):
            return "neutral"
        elif first_lower.startswith('con'):
            return "contradiction"
        return first_word.lower()
    
    return first_word

def exact_match_score(prediction: str, ground_truth: str, dataset_type: Optional[str] = None) -> bool:
    """정확한 일치 여부"""
    pred = extract_final_answer(prediction, dataset_type)
    true = extract_final_answer(ground_truth, dataset_type)
    
    pred_norm = normalize_answer(pred)
    true_norm = normalize_answer(true)
    
    # 숫자 비교 (GSM8K)
    if dataset_type == "gsm8k":
        try:
            return abs(float(pred_norm) - float(true_norm)) < 1e-6
        except ValueError:
            pass
    
    return pred_norm == true_norm

def calculate_accuracy(predictions: List[str], targets: List[str], dataset_type: Optional[str] = None) -> float:
    """정확도 계산"""
    if len(predictions) != len(targets) or len(predictions) == 0:
        return 0.0
    
    # 데이터셋 타입 자동 감지
    if dataset_type is None:
        dataset_type = detect_dataset_type(targets)
    
    correct = sum(exact_match_score(pred, target, dataset_type) 
                  for pred, target in zip(predictions, targets))
    
    accuracy = correct / len(predictions)
    print(f"🎯 Accuracy: {correct}/{len(predictions)} = {accuracy:.4f} (type: {dataset_type})")
    
    return accuracy

def detect_dataset_type(targets: List[str]) -> Optional[str]:
    """타겟으로부터 데이터셋 타입 자동 감지"""
    if not targets:
        return None
    
    sample = targets[:min(10, len(targets))]
    
    yes_no = sum(1 for t in sample if str(t).lower().strip() in ['yes', 'no', 'true', 'false'])
    numbers = sum(1 for t in sample if re.match(r'^\d+(\.\d+)?$', str(t).strip()))
    choices = sum(1 for t in sample if str(t).upper().strip() in ['A', 'B', 'C', 'D'])
    entailment = sum(1 for t in sample if str(t).lower().strip() in ['entailment', 'neutral', 'contradiction'])
    
    threshold = len(sample) * 0.5
    
    if yes_no >= threshold:
        return 'strategyqa'
    elif numbers >= threshold:
        return 'gsm8k'
    elif choices >= threshold:
        return 'logiqa'
    elif entailment >= threshold:
        return 'multinli'
    
    return None
