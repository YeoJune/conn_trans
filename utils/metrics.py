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

def extract_final_answer(text, dataset_type=None):
    """
    텍스트에서 최종 답변 추출 - 데이터셋별 특화
    """
    if not text or text.strip() == "":
        return ""
    
    text = str(text).strip()
    
    # 🔍 디버깅: 원본 텍스트 출력
    print(f"DEBUG: Extracting from: '{text}'")
    
    # StrategyQA: Yes/No 답변
    if dataset_type == "strategyqa" or any(word in text.lower() for word in ["yes", "no", "true", "false"]):
        # 더 강력한 Yes/No 추출
        yes_patterns = r'\b(yes|true|1|correct|positive)\b'
        no_patterns = r'\b(no|false|0|incorrect|negative)\b'
        
        text_lower = text.lower()
        if re.search(yes_patterns, text_lower):
            return "Yes"
        elif re.search(no_patterns, text_lower):
            return "No"
    
    # GSM8K: 숫자 답변 (더 정확한 추출)
    if dataset_type == "gsm8k" or re.search(r'-?\d+(?:\.\d+)?', text):
        # #### 형식 우선 확인
        if "####" in text:
            after_hash = text.split("####")[-1].strip()
            numbers = re.findall(r'-?\d+(?:\.\d+)?', after_hash)
            if numbers:
                return numbers[0]
        
        # 일반적인 숫자 추출
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]  # 마지막 숫자 사용
    
    # LogiQA: 선택지 추출 (A, B, C, D)
    if dataset_type == "logiqa" or any(choice in text.upper() for choice in ["A", "B", "C", "D"]):
        choices = re.findall(r'\b[ABCD]\b', text.upper())
        if choices:
            return choices[0]
    
    # MultiNLI: entailment, neutral, contradiction
    if dataset_type == "multinli" or any(word in text.lower() for word in ["entailment", "neutral", "contradiction"]):
        entailment_patterns = r'\b(entailment|entails|follows|implies)\b'
        neutral_patterns = r'\b(neutral|unknown|uncertain|unclear)\b'
        contradiction_patterns = r'\b(contradiction|contradicts|opposes|conflicts)\b'
        
        text_lower = text.lower()
        if re.search(entailment_patterns, text_lower):
            return "entailment"
        elif re.search(contradiction_patterns, text_lower):
            return "contradiction"
        elif re.search(neutral_patterns, text_lower):
            return "neutral"
    
    # 기본: 첫 번째 단어 또는 전체 텍스트
    words = text.strip().split()
    result = words[0] if words else text
    
    print(f"DEBUG: Extracted: '{result}'")
    return result

def exact_match_score(prediction, ground_truth):
    """정확한 일치 점수 - 디버깅 추가"""
    norm_pred = normalize_answer(str(prediction))
    norm_gt = normalize_answer(str(ground_truth))
    
    match = norm_pred == norm_gt
    
    # 🔍 디버깅 출력
    if not match:
        print(f"DEBUG: MISMATCH - Pred: '{norm_pred}' vs GT: '{norm_gt}'")
    
    return match

def calculate_accuracy(predictions, targets, dataset_type=None):
    """
    전체 정확도 계산 - 강화된 디버깅
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions and targets must have same length: {len(predictions)} vs {len(targets)}")
    
    correct = 0
    total = len(predictions)
    
    print(f"\n🔍 DEBUG: Calculating accuracy for {total} samples")
    print(f"Dataset type: {dataset_type}")
    
    # 처음 5개 샘플 상세 분석
    for i, (pred, target) in enumerate(zip(predictions[:5], targets[:5])):
        print(f"\n--- Sample {i+1} ---")
        print(f"Raw prediction: '{pred}'")
        print(f"Raw target: '{target}'")
        
        # 최종 답변 추출
        extracted_pred = extract_final_answer(str(pred), dataset_type)
        extracted_target = extract_final_answer(str(target), dataset_type)
        
        print(f"Extracted pred: '{extracted_pred}'")
        print(f"Extracted target: '{extracted_target}'")
        
        # 정확한 일치 확인
        is_match = exact_match_score(extracted_pred, extracted_target)
        print(f"Match: {is_match}")
    
    # 전체 정확도 계산
    for pred, target in zip(predictions, targets):
        extracted_pred = extract_final_answer(str(pred), dataset_type)
        extracted_target = extract_final_answer(str(target), dataset_type)
        
        if exact_match_score(extracted_pred, extracted_target):
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n✅ Final accuracy: {correct}/{total} = {accuracy:.4f}")
    
    return accuracy

def calculate_detailed_metrics(predictions, targets, dataset_type=None):
    """상세 메트릭 계산"""
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions and targets must have same length")
    
    exact_matches = []
    f1_scores = []
    
    for pred, target in zip(predictions, targets):
        extracted_pred = extract_final_answer(str(pred), dataset_type)
        extracted_target = extract_final_answer(str(target), dataset_type)
        
        em = exact_match_score(extracted_pred, extracted_target)
        f1 = f1_score(extracted_pred, extracted_target)
        
        exact_matches.append(em)
        f1_scores.append(f1)
    
    return {
        'exact_match': sum(exact_matches) / len(exact_matches),
        'f1': sum(f1_scores) / len(f1_scores),
        'total_samples': len(predictions),
        'correct_samples': sum(exact_matches)
    }

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