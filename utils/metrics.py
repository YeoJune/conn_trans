# utils/metrics.py
import re
import string
from typing import List, Optional, Tuple
import logging

def normalize_answer(s: str) -> str:
    """답변 정규화 - 더 견고하게"""
    if not s:
        return ""
    
    s = str(s).lower().strip()
    # 관사 제거
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # 특수문자 제거 (알파벳, 숫자, 공백만 유지)
    s = re.sub(r'[^\w\s]', '', s)
    # 연속 공백 정리
    return ' '.join(s.split())

def extract_final_answer(text: str, dataset_type: Optional[str] = None) -> str:
    """T5 답변에서 최종 답 추출 - 새 데이터셋 지원"""
    if not text:
        return ""
    
    text = str(text).strip()
    if not text:
        return ""
    
    # 첫 번째 단어 추출
    words = text.split()
    first_word = words[0] if words else ""
    
    if dataset_type == "strategyqa":
        # Yes/No 질문 처리
        first_lower = first_word.lower()
        
        # 명확한 Yes 패턴
        if any(pattern in first_lower for pattern in ['yes', 'true', 'correct', 'right']):
            return "Yes"
        # 명확한 No 패턴  
        elif any(pattern in first_lower for pattern in ['no', 'false', 'incorrect', 'wrong']):
            return "No"
        # 숫자로 표현된 경우
        elif first_lower in ['1', 'one']:
            return "Yes"
        elif first_lower in ['0', 'zero']:
            return "No"
        
        # 기본값: 첫 글자 기준
        return "Yes" if first_lower.startswith('y') else "No"
    
    elif dataset_type == "gsm8k":
        # 수학 문제 - 숫자 추출
        # 첫 번째 단어가 숫자인지 확인
        first_numbers = re.findall(r'^-?\d+(?:\.\d+)?$', first_word)
        if first_numbers:
            try:
                # 정수로 변환 가능하면 정수로, 아니면 실수로
                num = float(first_numbers[0])
                if num.is_integer():
                    return str(int(num))
                else:
                    return str(num)
            except ValueError:
                pass
        
        # 첫 번째 단어가 숫자가 아니면 "error" 반환 (틀린 것으로 간주)
        return "error"
    
    elif dataset_type == "logiqa":
        # 다중 선택 문제 - A, B, C, D
        first_upper = first_word.upper()
        
        # 직접적인 선택지
        if first_upper in ['A', 'B', 'C', 'D']:
            return first_upper
        
        # 괄호나 기타 문자 제거 후 확인
        clean = re.sub(r'[^\w]', '', first_upper)
        if clean in ['A', 'B', 'C', 'D']:
            return clean
        
        # 단어로 된 선택지 처리
        choice_map = {
            'FIRST': 'A', 'ONE': 'A', '1': 'A',
            'SECOND': 'B', 'TWO': 'B', '2': 'B', 
            'THIRD': 'C', 'THREE': 'C', '3': 'C',
            'FOURTH': 'D', 'FOUR': 'D', '4': 'D'
        }
        return choice_map.get(first_upper, "A")  # 기본값 A
    
    elif dataset_type == "multinli":
        # 자연어 추론 - entailment, neutral, contradiction
        first_lower = first_word.lower()
        
        # 명확한 매칭
        if first_lower.startswith('ent'):
            return "entailment"
        elif first_lower.startswith('neu'):
            return "neutral"  
        elif first_lower.startswith('con'):
            return "contradiction"
        
        # 동의어 처리
        entailment_words = ['yes', 'true', 'follows', 'implies', 'supports']
        neutral_words = ['maybe', 'unclear', 'unknown', 'possible']
        contradiction_words = ['no', 'false', 'contradicts', 'opposes']
        
        if any(word in first_lower for word in entailment_words):
            return "entailment"
        elif any(word in first_lower for word in neutral_words):
            return "neutral"
        elif any(word in first_lower for word in contradiction_words):
            return "contradiction"
        
        return "neutral"  # 기본값
    
    elif dataset_type == "eli5":
        # ELI5 - 긴 설명 생성 (첫 문장 추출)
        # 문장 단위로 분할하여 첫 문장 반환
        sentences = re.split(r'[.!?]+', text)
        first_sentence = sentences[0].strip() if sentences else text.strip()
        
        # 너무 짧으면 전체 텍스트의 처음 일부 반환
        if len(first_sentence.split()) < 3:
            words = text.split()[:20]  # 처음 20 단어
            return ' '.join(words)
        
        return first_sentence
    
    elif dataset_type == "commongen":
        # CommonGen - 개념 연결 문장 (전체 텍스트 반환, 단 적절히 정리)
        # 불필요한 앞뒤 공백 제거 및 연속 공백 정리
        cleaned = ' '.join(text.split())
        
        # 너무 길면 첫 문장만
        sentences = re.split(r'[.!?]+', cleaned)
        if sentences and len(sentences[0].split()) <= 30:
            return sentences[0].strip()
        
        # 첫 문장이 너무 길거나 없으면 처음 30 단어만
        words = cleaned.split()[:30]
        result = ' '.join(words)
        
        # 문장이 완성되지 않았으면 마침표 추가
        if result and not result.endswith(('.', '!', '?')):
            result += '.'
        
        return result
    
    # 알 수 없는 데이터셋 타입인 경우 첫 단어 반환
    return first_word

def exact_match_score(prediction: str, ground_truth: str, dataset_type: Optional[str] = None) -> bool:
    """정확한 일치 여부 - 새 데이터셋 지원"""
    try:
        pred = extract_final_answer(prediction, dataset_type)
        true = extract_final_answer(ground_truth, dataset_type)
        
        pred_norm = normalize_answer(pred)
        true_norm = normalize_answer(true)
        
        # 수학 문제는 숫자 비교
        if dataset_type == "gsm8k":
            try:
                pred_num = float(pred_norm) if pred_norm else 0.0
                true_num = float(true_norm) if true_norm else 0.0
                return abs(pred_num - true_num) < 1e-6  # 부동소수점 오차 고려
            except ValueError:
                # 숫자 변환 실패 시 문자열 비교
                pass
        
        # ELI5와 CommonGen은 의미적 유사성을 고려한 평가
        elif dataset_type in ["eli5", "commongen"]:
            return calculate_semantic_similarity(pred_norm, true_norm)
        
        # 기본적으로 정규화된 문자열 비교
        return pred_norm == true_norm
        
    except Exception as e:
        # 예외 발생 시 False 반환 (로깅은 선택적)
        logging.debug(f"Exact match error: {e}")
        return False

def calculate_semantic_similarity(pred: str, target: str) -> bool:
    """ELI5, CommonGen용 의미적 유사성 계산"""
    if not pred or not target:
        return False
    
    # 토큰 기반 유사성 (Jaccard similarity)
    pred_tokens = set(pred.split())
    target_tokens = set(target.split())
    
    if not pred_tokens or not target_tokens:
        return False
    
    intersection = len(pred_tokens & target_tokens)
    union = len(pred_tokens | target_tokens)
    
    jaccard_similarity = intersection / union if union > 0 else 0.0
    
    # 30% 이상 유사하면 정답으로 간주 (생성 태스크는 관대하게)
    return jaccard_similarity >= 0.3

def calculate_accuracy(predictions: List[str], targets: List[str], 
                      dataset_type: Optional[str] = None, 
                      verbose: bool = True) -> float:
    """정확도 계산 - 새 데이터셋 지원"""
    if not predictions or not targets:
        return 0.0
    
    if len(predictions) != len(targets):
        if verbose:
            print(f"⚠️ Length mismatch: predictions={len(predictions)}, targets={len(targets)}")
        # 길이가 다르면 짧은 쪽에 맞춤
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]
    
    # 데이터셋 타입 자동 감지
    if dataset_type is None:
        dataset_type = detect_dataset_type(targets)
        if verbose and dataset_type:
            print(f"🔍 Dataset type auto-detected: {dataset_type}")
    
    # 정확도 계산
    correct_count = 0
    total_count = len(predictions)
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        if exact_match_score(pred, target, dataset_type):
            correct_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    if verbose:
        print(f"🎯 Accuracy: {correct_count}/{total_count} = {accuracy:.4f} (type: {dataset_type or 'unknown'})")
        
        # 새 데이터셋의 경우 평가 방식 설명
        if dataset_type in ["eli5", "commongen"]:
            print(f"   📝 Note: Using semantic similarity (30%+ token overlap) for {dataset_type}")
    
    return accuracy

def detect_dataset_type(targets: List[str]) -> Optional[str]:
    """타겟으로부터 데이터셋 타입 자동 감지 - 새 데이터셋 지원"""
    if not targets:
        return None
    
    # 샘플 크기 조정 (최대 20개, 최소 5개)
    sample_size = min(max(len(targets) // 10, 5), 20)
    sample = targets[:sample_size]
    
    # 각 타입별 패턴 카운트
    counters = {
        'strategyqa': 0,
        'gsm8k': 0, 
        'logiqa': 0,
        'multinli': 0,
        'eli5': 0,
        'commongen': 0
    }
    
    for target in sample:
        target_str = str(target).lower().strip()
        word_count = len(target_str.split())
        
        # StrategyQA: Yes/No 답변
        if target_str in ['yes', 'no', 'true', 'false', '1', '0']:
            counters['strategyqa'] += 1
        
        # GSM8K: 숫자 답변
        elif re.match(r'^-?\d+(\.\d+)?$', target_str):
            counters['gsm8k'] += 1
            
        # LogiQA: A, B, C, D 선택지
        elif target_str.upper() in ['A', 'B', 'C', 'D']:
            counters['logiqa'] += 1
            
        # MultiNLI: entailment, neutral, contradiction
        elif target_str in ['entailment', 'neutral', 'contradiction']:
            counters['multinli'] += 1
        
        # ELI5: 긴 설명 (보통 50+ 단어)
        elif word_count >= 20 and any(word in target_str for word in ['because', 'when', 'how', 'why', 'what', 'explanation']):
            counters['eli5'] += 1
        
        # CommonGen: 중간 길이 문장 (5-30 단어, 완전한 문장)
        elif 5 <= word_count <= 30 and target_str.endswith('.'):
            counters['commongen'] += 1
    
    # 가장 많이 매칭된 타입 반환 (40% 이상 매칭 시)
    threshold = sample_size * 0.4  # 새 데이터셋은 패턴이 덜 명확할 수 있어서 임계값 낮춤
    
    for dataset_type, count in counters.items():
        if count >= threshold:
            return dataset_type
    
    # 특별 케이스: 평균 길이로 추가 판단
    avg_length = sum(len(str(target).split()) for target in sample) / len(sample)
    
    if avg_length >= 30:
        return 'eli5'
    elif 10 <= avg_length < 30:
        return 'commongen'
    
    return None

def get_accuracy_breakdown(predictions: List[str], targets: List[str], 
                          dataset_type: Optional[str] = None) -> dict:
    """정확도 세부 분석 정보 반환 - 새 데이터셋 지원"""
    if not predictions or not targets:
        return {'accuracy': 0.0, 'correct': 0, 'total': 0, 'details': []}
    
    # 길이 맞춤
    min_len = min(len(predictions), len(targets))
    predictions = predictions[:min_len]
    targets = targets[:min_len]
    
    # 데이터셋 타입 감지
    if dataset_type is None:
        dataset_type = detect_dataset_type(targets)
    
    # 세부 분석
    details = []
    correct_count = 0
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        is_correct = exact_match_score(pred, target, dataset_type)
        if is_correct:
            correct_count += 1
            
        # 새 데이터셋의 경우 추가 정보
        extra_info = {}
        if dataset_type in ["eli5", "commongen"]:
            pred_norm = normalize_answer(extract_final_answer(pred, dataset_type))
            target_norm = normalize_answer(extract_final_answer(target, dataset_type))
            
            if pred_norm and target_norm:
                pred_tokens = set(pred_norm.split())
                target_tokens = set(target_norm.split())
                intersection = len(pred_tokens & target_tokens)
                union = len(pred_tokens | target_tokens)
                similarity = intersection / union if union > 0 else 0.0
                extra_info['token_similarity'] = similarity
            
        details.append({
            'index': i,
            'prediction': pred,
            'target': target,
            'correct': is_correct,
            'extracted_pred': extract_final_answer(pred, dataset_type),
            'extracted_target': extract_final_answer(target, dataset_type),
            **extra_info
        })
    
    return {
        'accuracy': correct_count / len(predictions),
        'correct': correct_count,
        'total': len(predictions),
        'dataset_type': dataset_type,
        'details': details
    }