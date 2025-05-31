# utils/metrics.py
import re
import string
from collections import Counter

def normalize_answer(s):
    """답변을 정규화 (공백, 구두점, 대소문자 처리)"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_final_answer(text):
    """텍스트에서 최종 답변 추출"""
    # 숫자 추출 (GSM8K용)
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]
    
    # 선택지 추출 (LogiQA용) - A, B, C, D
    choices = re.findall(r'\b[ABCD]\b', text.upper())
    if choices:
        return choices[0]
    
    # Yes/No 추출 (StrategyQA용)
    yes_no = re.findall(r'\b(yes|no)\b', text.lower())
    if yes_no:
        return yes_no[-1].capitalize()
    
    # 그 외의 경우 첫 번째 단어 반환
    words = text.strip().split()
    return words[0] if words else text

def exact_match_score(prediction, ground_truth):
    """정확한 일치 점수"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    """F1 점수 (토큰 레벨)"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def calculate_accuracy(predictions, targets):
    """전체 정확도 계산"""
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions and targets must have same length: {len(predictions)} vs {len(targets)}")
    
    correct = 0
    total = len(predictions)
    
    for pred, target in zip(predictions, targets):
        # 최종 답변 추출
        extracted_pred = extract_final_answer(str(pred))
        extracted_target = extract_final_answer(str(target))
        
        # 정확한 일치 확인
        if exact_match_score(extracted_pred, extracted_target):
            correct += 1
    
    return correct / total if total > 0 else 0.0

def calculate_detailed_metrics(predictions, targets):
    """상세 메트릭 계산"""
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions and targets must have same length")
    
    exact_matches = []
    f1_scores = []
    
    for pred, target in zip(predictions, targets):
        extracted_pred = extract_final_answer(str(pred))
        extracted_target = extract_final_answer(str(target))
        
        em = exact_match_score(extracted_pred, extracted_target)
        f1 = f1_score(extracted_pred, extracted_target)
        
        exact_matches.append(em)
        f1_scores.append(f1)
    
    return {
        'exact_match': sum(exact_matches) / len(exact_matches),
        'f1': sum(f1_scores) / len(f1_scores),
        'total_samples': len(predictions)
    }

def analyze_error_cases(predictions, targets, max_examples=10):
    """오류 사례 분석"""
    errors = []
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        extracted_pred = extract_final_answer(str(pred))
        extracted_target = extract_final_answer(str(target))
        
        if not exact_match_score(extracted_pred, extracted_target):
            errors.append({
                'index': i,
                'prediction': pred,
                'target': target,
                'extracted_prediction': extracted_pred,
                'extracted_target': extracted_target,
                'f1_score': f1_score(extracted_pred, extracted_target)
            })
    
    # F1 점수가 낮은 순으로 정렬
    errors.sort(key=lambda x: x['f1_score'])
    
    return errors[:max_examples]