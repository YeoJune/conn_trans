# utils/metrics.py
import re
import string
import torch
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

def normalize_answer(s: str) -> str:
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

def extract_final_answer(text: str, dataset_type: Optional[str] = None) -> str:
    """
    ✅ T5 Text2Text Generation에 최적화된 답변 추출
    
    T5는 짧고 직접적인 답변을 생성하므로 복잡한 파싱보다는
    첫 번째 의미있는 토큰을 추출하는 것이 효과적
    """
    if not text or text.strip() == "":
        return ""
    
    text = str(text).strip()
    
    # T5는 보통 첫 번째 단어가 답이므로 단순하게 처리
    words = text.split()
    if not words:
        return ""
    
    first_word = words[0].strip()
    
    # 데이터셋별 특화 처리
    if dataset_type == "strategyqa":
        # Yes/No 패턴
        first_lower = first_word.lower()
        if first_lower in ['yes', 'y', 'true', '1']:
            return "Yes"
        elif first_lower in ['no', 'n', 'false', '0']:
            return "No"
        # 첫 글자로 판단
        elif first_lower.startswith('y'):
            return "Yes"
        elif first_lower.startswith('n'):
            return "No"
        return first_word.capitalize()
    
    elif dataset_type == "gsm8k":
        # 숫자 답변 - T5는 보통 숫자만 출력
        # 첫 번째 숫자 찾기
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[0]
        return "0"
    
    elif dataset_type == "logiqa":
        # A, B, C, D 선택지
        first_upper = first_word.upper()
        if first_upper in ['A', 'B', 'C', 'D']:
            return first_upper
        # 괄호 제거
        clean_first = re.sub(r'[^\w]', '', first_upper)
        if clean_first in ['A', 'B', 'C', 'D']:
            return clean_first
        return "A"  # 기본값
    
    elif dataset_type == "multinli":
        # entailment, neutral, contradiction
        first_lower = first_word.lower()
        if first_lower.startswith('ent'):
            return "entailment"
        elif first_lower.startswith('neu'):
            return "neutral"
        elif first_lower.startswith('con'):
            return "contradiction"
        return first_word.lower()
    
    # 기본: 첫 번째 단어 반환
    return first_word

def exact_match_score(prediction: str, ground_truth: str, dataset_type: Optional[str] = None) -> bool:
    """✅ T5에 최적화된 정확한 일치 점수"""
    pred_answer = extract_final_answer(prediction, dataset_type)
    true_answer = extract_final_answer(ground_truth, dataset_type)
    
    # 정규화 후 비교
    pred_norm = normalize_answer(pred_answer)
    true_norm = normalize_answer(true_answer)
    
    # GSM8K 특별 처리 (숫자 비교)
    if dataset_type == "gsm8k":
        try:
            pred_num = float(pred_norm) if pred_norm else None
            true_num = float(true_norm) if true_norm else None
            if pred_num is not None and true_num is not None:
                return abs(pred_num - true_num) < 1e-6
        except ValueError:
            pass
    
    return pred_norm == true_norm

def calculate_accuracy(predictions: List[str], targets: List[str], 
                      dataset_type: Optional[str] = None) -> float:
    """✅ T5 Text2Text Generation에 최적화된 정확도 계산"""
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions and targets must have same length: {len(predictions)} vs {len(targets)}")
    
    if len(predictions) == 0:
        return 0.0
    
    # 데이터셋 타입 자동 감지
    if dataset_type is None:
        dataset_type = detect_dataset_type(targets)
    
    correct = 0
    total = len(predictions)
    
    print(f"\n🔍 T5 Accuracy Debug (dataset_type: {dataset_type}):")
    
    # 각 예측 분석
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        extracted_pred = extract_final_answer(str(pred), dataset_type)
        extracted_target = extract_final_answer(str(target), dataset_type)
        
        is_correct = exact_match_score(str(pred), str(target), dataset_type)
        if is_correct:
            correct += 1
        
        # 처음 5개 샘플만 상세 출력
        if i < 5:
            status = "✅" if is_correct else "❌"
            print(f"  {status} Pred: '{pred}' -> '{extracted_pred}' | Target: '{target}' -> '{extracted_target}'")
    
    accuracy = correct / total
    print(f"🎯 Final T5 accuracy: {correct}/{total} = {accuracy:.4f}")
    
    return accuracy

def calculate_token_level_accuracy(predictions: List[str], targets: List[str],
                                 tokenizer, dataset_type: Optional[str] = None) -> Dict[str, float]:
    """✅ T5 토큰 레벨 정확도 (더 세밀한 분석)"""
    if len(predictions) != len(targets):
        raise ValueError(f"Length mismatch: {len(predictions)} vs {len(targets)}")
    
    total_tokens = 0
    correct_tokens = 0
    
    for pred, target in zip(predictions, targets):
        # 토크나이징
        pred_tokens = tokenizer.encode(str(pred), add_special_tokens=False)
        target_tokens = tokenizer.encode(str(target), add_special_tokens=False)
        
        # 길이 맞추기 (더 짧은 것에 맞춤)
        min_len = min(len(pred_tokens), len(target_tokens))
        
        for i in range(min_len):
            total_tokens += 1
            if pred_tokens[i] == target_tokens[i]:
                correct_tokens += 1
    
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    return {
        'token_accuracy': token_accuracy,
        'total_tokens': total_tokens,
        'correct_tokens': correct_tokens
    }

def calculate_sequence_level_metrics(predictions: List[str], targets: List[str],
                                   dataset_type: Optional[str] = None) -> Dict[str, float]:
    """✅ T5에 적합한 시퀀스 레벨 메트릭"""
    exact_matches = []
    f1_scores = []
    
    for pred, target in zip(predictions, targets):
        # Exact Match
        em = exact_match_score(pred, target, dataset_type)
        exact_matches.append(em)
        
        # F1 Score (토큰 레벨)
        pred_tokens = normalize_answer(extract_final_answer(pred, dataset_type)).split()
        target_tokens = normalize_answer(extract_final_answer(target, dataset_type)).split()
        
        if len(pred_tokens) == 0 and len(target_tokens) == 0:
            f1 = 1.0
        elif len(pred_tokens) == 0 or len(target_tokens) == 0:
            f1 = 0.0
        else:
            common = Counter(pred_tokens) & Counter(target_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                f1 = 0.0
            else:
                precision = num_same / len(pred_tokens)
                recall = num_same / len(target_tokens)
                f1 = 2 * precision * recall / (precision + recall)
        
        f1_scores.append(f1)
    
    return {
        'exact_match': np.mean(exact_matches),
        'f1_score': np.mean(f1_scores),
        'total_samples': len(predictions)
    }

def calculate_generation_quality_metrics(predictions: List[str]) -> Dict[str, float]:
    """✅ T5 생성 품질 메트릭"""
    if not predictions:
        return {}
    
    # 길이 통계
    lengths = [len(pred.split()) for pred in predictions]
    
    # 다양성 측정
    unique_predictions = len(set(predictions))
    diversity = unique_predictions / len(predictions)
    
    # 빈 예측 비율
    empty_predictions = sum(1 for pred in predictions if not pred.strip())
    empty_ratio = empty_predictions / len(predictions)
    
    # 반복 패턴 감지
    repetitive_predictions = 0
    for pred in predictions:
        words = pred.split()
        if len(words) > 1:
            # 같은 단어가 연속으로 반복되는지 확인
            for i in range(len(words) - 1):
                if words[i] == words[i + 1]:
                    repetitive_predictions += 1
                    break
    
    repetition_ratio = repetitive_predictions / len(predictions)
    
    return {
        'avg_length': np.mean(lengths),
        'length_std': np.std(lengths),
        'diversity': diversity,
        'empty_ratio': empty_ratio,
        'repetition_ratio': repetition_ratio,
        'min_length': min(lengths),
        'max_length': max(lengths)
    }

def detect_dataset_type(targets: List[str]) -> Optional[str]:
    """타겟 데이터로부터 데이터셋 타입 자동 감지 (개선됨)"""
    if not targets:
        return None
    
    # 더 많은 샘플로 정확도 향상
    sample_targets = targets[:min(20, len(targets))]
    
    yes_no_count = 0
    number_count = 0
    choice_count = 0
    entail_count = 0
    
    for target in sample_targets:
        target_str = str(target).lower().strip()
        
        # Yes/No 패턴
        if target_str in ['yes', 'no', 'true', 'false']:
            yes_no_count += 1
        
        # 숫자 패턴 (더 유연하게)
        if re.search(r'^\d+(\.\d+)?$', target_str) or target_str.isdigit():
            number_count += 1
        
        # 선택지 패턴
        if re.search(r'^[abcd]$', target_str):
            choice_count += 1
        
        # Entailment 패턴
        if target_str in ['entailment', 'neutral', 'contradiction']:
            entail_count += 1
    
    # 가장 많은 패턴으로 결정 (더 엄격한 기준)
    total_samples = len(sample_targets)
    threshold = total_samples * 0.5  # 50% 이상이어야 확신
    
    if yes_no_count >= threshold:
        return 'strategyqa'
    elif number_count >= threshold:
        return 'gsm8k'
    elif choice_count >= threshold:
        return 'logiqa'
    elif entail_count >= threshold:
        return 'multinli'
    
    return None  # 확신할 수 없음

def analyze_error_cases(predictions: List[str], targets: List[str],
                       dataset_type: Optional[str] = None, 
                       max_examples: int = 10) -> Dict[str, Any]:
    """✅ T5 오류 사례 분석 (개선됨)"""
    if dataset_type is None:
        dataset_type = detect_dataset_type(targets)
    
    errors = []
    error_types = {
        'format_errors': 0,      # 형식 오류 (예: Yes/No 대신 다른 답)
        'partial_matches': 0,    # 부분 일치 (올바른 내용이지만 형식이 다름)
        'content_errors': 0,     # 내용 오류 (완전히 틀림)
        'empty_predictions': 0   # 빈 예측
    }
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        extracted_pred = extract_final_answer(str(pred), dataset_type)
        extracted_target = extract_final_answer(str(target), dataset_type)
        
        if not exact_match_score(pred, target, dataset_type):
            # 오류 타입 분류
            error_type = 'content_errors'  # 기본값
            
            if not pred.strip():
                error_type = 'empty_predictions'
            elif dataset_type == "strategyqa":
                if extracted_pred not in ['Yes', 'No']:
                    error_type = 'format_errors'
                else:
                    error_type = 'content_errors'
            elif dataset_type == "logiqa":
                if extracted_pred not in ['A', 'B', 'C', 'D']:
                    error_type = 'format_errors'
                else:
                    error_type = 'content_errors'
            elif dataset_type == "multinli":
                if extracted_pred not in ['entailment', 'neutral', 'contradiction']:
                    error_type = 'format_errors'
                else:
                    error_type = 'content_errors'
            
            error_types[error_type] += 1
            
            if len(errors) < max_examples:
                errors.append({
                    'index': i,
                    'prediction': pred,
                    'target': target,
                    'extracted_prediction': extracted_pred,
                    'extracted_target': extracted_target,
                    'error_type': error_type,
                    'dataset_type': dataset_type
                })
    
    return {
        'errors': errors,
        'error_types': error_types,
        'total_errors': len([p for p, t in zip(predictions, targets) 
                           if not exact_match_score(p, t, dataset_type)]),
        'total_samples': len(predictions)
    }

def calculate_comprehensive_metrics(predictions: List[str], targets: List[str],
                                  tokenizer=None, dataset_type: Optional[str] = None) -> Dict[str, Any]:
    """✅ T5에 최적화된 종합 메트릭"""
    if not predictions or not targets:
        return {}
    
    metrics = {}
    
    # 기본 정확도
    metrics['accuracy'] = calculate_accuracy(predictions, targets, dataset_type)
    
    # 시퀀스 레벨 메트릭
    seq_metrics = calculate_sequence_level_metrics(predictions, targets, dataset_type)
    metrics.update(seq_metrics)
    
    # 생성 품질 메트릭
    quality_metrics = calculate_generation_quality_metrics(predictions)
    metrics.update(quality_metrics)
    
    # 토큰 레벨 정확도 (토크나이저가 있는 경우)
    if tokenizer is not None:
        token_metrics = calculate_token_level_accuracy(predictions, targets, tokenizer, dataset_type)
        metrics.update(token_metrics)
    
    # 오류 분석
    error_analysis = analyze_error_cases(predictions, targets, dataset_type)
    metrics['error_analysis'] = error_analysis
    
    # 데이터셋 특화 메트릭
    if dataset_type:
        metrics['dataset_type'] = dataset_type
        metrics['detected_format_compliance'] = _check_format_compliance(predictions, dataset_type)
    
    return metrics

def _check_format_compliance(predictions: List[str], dataset_type: str) -> float:
    """예측이 예상 형식에 맞는지 확인"""
    compliant = 0
    
    for pred in predictions:
        extracted = extract_final_answer(pred, dataset_type)
        
        if dataset_type == "strategyqa":
            if extracted in ['Yes', 'No']:
                compliant += 1
        elif dataset_type == "logiqa":
            if extracted in ['A', 'B', 'C', 'D']:
                compliant += 1
        elif dataset_type == "multinli":
            if extracted in ['entailment', 'neutral', 'contradiction']:
                compliant += 1
        elif dataset_type == "gsm8k":
            try:
                float(extracted)
                compliant += 1
            except ValueError:
                pass
    
    return compliant / len(predictions) if predictions else 0.0