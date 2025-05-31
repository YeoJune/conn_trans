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
    ✅ 데이터셋별로 최적화된 답변 추출
    """
    if not text or text.strip() == "":
        return ""
    
    text = str(text).strip()
    
    # StrategyQA: Yes/No 답변 추출
    if dataset_type == "strategyqa" or any(word in text.lower() for word in ["yes", "no"]):
        # Yes/No 패턴 찾기 (대소문자 무관)
        yes_pattern = re.search(r'\b(yes|true|y)\b', text.lower())
        no_pattern = re.search(r'\b(no|false|n)\b', text.lower())
        
        if yes_pattern and no_pattern:
            # 둘 다 있으면 마지막에 나온 것 사용
            yes_pos = yes_pattern.start()
            no_pos = no_pattern.start()
            return "Yes" if yes_pos > no_pos else "No"
        elif yes_pattern:
            return "Yes"
        elif no_pattern:
            return "No"
        
        # 패턴이 없으면 첫 글자로 판단
        first_char = text.lower().strip()[0] if text.strip() else ""
        if first_char == 'y':
            return "Yes"
        elif first_char == 'n':
            return "No"
    
    # GSM8K: 숫자 답변 추출
    if dataset_type == "gsm8k" or re.search(r'\d', text):
        # "#### 24" 형식 먼저 확인
        gsm_pattern = re.search(r'####\s*([+-]?\d+(?:\.\d+)?)', text)
        if gsm_pattern:
            return gsm_pattern.group(1).strip()
        
        # 일반 숫자 패턴 (단위 제거)
        # $24, 24 dollars, 24.5, -5 등 처리
        number_patterns = [
            r'([+-]?\d+(?:\.\d+)?)\s*(?:dollars?|usd|\$)?(?:\s|$)',  # 24 dollars, 24$
            r'\$?\s*([+-]?\d+(?:\.\d+)?)',  # $24, 24
            r'([+-]?\d+(?:\.\d+)?)'  # 단순 숫자
        ]
        
        for pattern in number_patterns:
            numbers = re.findall(pattern, text, re.IGNORECASE)
            if numbers:
                # 마지막 숫자 사용 (보통 최종 답)
                return numbers[-1].strip()
    
    # LogiQA: A, B, C, D 선택지 추출
    if dataset_type == "logiqa" or re.search(r'\b[ABCD]\b', text.upper()):
        choices = re.findall(r'\b([ABCD])\b', text.upper())
        if choices:
            return choices[-1]  # 마지막 선택지 사용
        
        # 괄호 안의 선택지도 확인: (A), (B) 등
        paren_choices = re.findall(r'\(([ABCD])\)', text.upper())
        if paren_choices:
            return paren_choices[-1]
    
    # MultiNLI: entailment, neutral, contradiction
    if dataset_type == "multinli" or any(word in text.lower() for word in ["entailment", "neutral", "contradiction"]):
        text_lower = text.lower()
        
        # 완전한 단어로 매칭
        if re.search(r'\bentailment\b', text_lower):
            return "entailment"
        elif re.search(r'\bneutral\b', text_lower):
            return "neutral"
        elif re.search(r'\bcontradiction\b', text_lower):
            return "contradiction"
        
        # 축약형도 확인
        if re.search(r'\bentail\b', text_lower):
            return "entailment"
        elif re.search(r'\bcontradict\b', text_lower):
            return "contradiction"
    
    # 기본값: 첫 번째 의미있는 단어 반환
    words = text.strip().split()
    return words[0] if words else text

def exact_match_score(prediction, ground_truth, dataset_type=None):
    """✅ 데이터셋별 정확한 일치 점수"""
    # 답변 추출
    pred_answer = extract_final_answer(prediction, dataset_type)
    true_answer = extract_final_answer(ground_truth, dataset_type)
    
    # 정규화 후 비교
    pred_norm = normalize_answer(pred_answer)
    true_norm = normalize_answer(true_answer)
    
    # 특별 케이스 처리
    if dataset_type == "gsm8k":
        # 숫자는 float 변환 후 비교
        try:
            pred_num = float(pred_norm) if pred_norm else None
            true_num = float(true_norm) if true_norm else None
            if pred_num is not None and true_num is not None:
                return abs(pred_num - true_num) < 1e-6  # 부동소수점 오차 고려
        except:
            pass
    
    return pred_norm == true_norm

def calculate_accuracy(predictions, targets, dataset_type=None):
    """✅ 데이터셋별 최적화된 정확도 계산"""
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions and targets must have same length: {len(predictions)} vs {len(targets)}")
    
    if len(predictions) == 0:
        return 0.0
    
    # 데이터셋 타입 자동 감지 (지정되지 않은 경우)
    if dataset_type is None:
        dataset_type = detect_dataset_type(targets)
    
    correct = 0
    total = len(predictions)
    
    # 디버깅을 위한 샘플 출력
    debug_samples = min(5, total)
    print(f"\n🔍 Accuracy Debug (dataset_type: {dataset_type}):")
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        # 답변 추출
        extracted_pred = extract_final_answer(str(pred), dataset_type)
        extracted_target = extract_final_answer(str(target), dataset_type)
        
        # 정확한 일치 확인
        is_correct = exact_match_score(str(pred), str(target), dataset_type)
        if is_correct:
            correct += 1
        
        # 디버깅 출력 (처음 몇 개만)
        if i < debug_samples:
            status = "✅" if is_correct else "❌"
            print(f"  {status} Pred: '{pred}' -> '{extracted_pred}' | Target: '{target}' -> '{extracted_target}'")
    
    accuracy = correct / total
    print(f"🎯 Final accuracy: {correct}/{total} = {accuracy:.4f}")
    
    return accuracy

def detect_dataset_type(targets):
    """타겟 데이터로부터 데이터셋 타입 자동 감지"""
    if not targets:
        return None
    
    # 샘플링해서 확인
    sample_targets = targets[:min(10, len(targets))]
    
    yes_no_count = 0
    number_count = 0
    choice_count = 0
    entail_count = 0
    
    for target in sample_targets:
        target_str = str(target).lower().strip()
        
        # Yes/No 패턴
        if target_str in ['yes', 'no', 'true', 'false']:
            yes_no_count += 1
        
        # 숫자 패턴
        if re.search(r'^\d+(\.\d+)?$', target_str):
            number_count += 1
        
        # 선택지 패턴
        if re.search(r'^[abcd]$', target_str):
            choice_count += 1
        
        # Entailment 패턴
        if target_str in ['entailment', 'neutral', 'contradiction']:
            entail_count += 1
    
    # 가장 많은 패턴으로 결정
    counts = {
        'strategyqa': yes_no_count,
        'gsm8k': number_count,
        'logiqa': choice_count,
        'multinli': entail_count
    }
    
    detected_type = max(counts, key=counts.get)
    max_count = counts[detected_type]
    
    # 확신도가 낮으면 None 반환
    if max_count < len(sample_targets) * 0.3:
        return None
    
    return detected_type

def f1_score(prediction, ground_truth, dataset_type=None):
    """F1 점수 (토큰 레벨)"""
    pred_answer = extract_final_answer(prediction, dataset_type)
    true_answer = extract_final_answer(ground_truth, dataset_type)
    
    prediction_tokens = normalize_answer(pred_answer).split()
    ground_truth_tokens = normalize_answer(true_answer).split()
    
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

def calculate_detailed_metrics(predictions, targets, dataset_type=None):
    """상세 메트릭 계산"""
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions and targets must have same length")
    
    if dataset_type is None:
        dataset_type = detect_dataset_type(targets)
    
    exact_matches = []
    f1_scores = []
    
    for pred, target in zip(predictions, targets):
        em = exact_match_score(pred, target, dataset_type)
        f1 = f1_score(pred, target, dataset_type)
        
        exact_matches.append(em)
        f1_scores.append(f1)
    
    return {
        'exact_match': sum(exact_matches) / len(exact_matches),
        'f1': sum(f1_scores) / len(f1_scores),
        'total_samples': len(predictions),
        'detected_dataset_type': dataset_type
    }

def analyze_error_cases(predictions, targets, dataset_type=None, max_examples=10):
    """오류 사례 분석"""
    if dataset_type is None:
        dataset_type = detect_dataset_type(targets)
    
    errors = []
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        extracted_pred = extract_final_answer(str(pred), dataset_type)
        extracted_target = extract_final_answer(str(target), dataset_type)
        
        if not exact_match_score(pred, target, dataset_type):
            errors.append({
                'index': i,
                'prediction': pred,
                'target': target,
                'extracted_prediction': extracted_pred,
                'extracted_target': extracted_target,
                'f1_score': f1_score(pred, target, dataset_type),
                'dataset_type': dataset_type
            })
    
    # F1 점수가 낮은 순으로 정렬
    errors.sort(key=lambda x: x['f1_score'])
    
    return errors[:max_examples]