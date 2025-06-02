# utils/metrics.py
import re
import string
from typing import List, Optional, Tuple, Dict
import logging
from collections import Counter
import math

def normalize_answer(s: str) -> str:
    """답변 정규화"""
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
    """답변 추출 (기존 로직 유지)"""
    if not text:
        return ""
    
    text = str(text).strip()
    if not text:
        return ""
    
    words = text.split()
    first_word = words[0] if words else ""
    
    if dataset_type == "strategyqa":
        first_lower = first_word.lower()
        if any(pattern in first_lower for pattern in ['yes', 'true', 'correct', 'right']):
            return "Yes"
        elif any(pattern in first_lower for pattern in ['no', 'false', 'incorrect', 'wrong']):
            return "No"
        elif first_lower in ['1', 'one']:
            return "Yes"
        elif first_lower in ['0', 'zero']:
            return "No"
        return "Yes" if first_lower.startswith('y') else "No"
    
    elif dataset_type == "gsm8k":
        first_numbers = re.findall(r'^-?\d+(?:\.\d+)?$', first_word)
        if first_numbers:
            try:
                num = float(first_numbers[0])
                if num.is_integer():
                    return str(int(num))
                else:
                    return str(num)
            except ValueError:
                pass
        return "error"
    
    elif dataset_type == "logiqa":
        first_upper = first_word.upper()
        if first_upper in ['A', 'B', 'C', 'D']:
            return first_upper
        clean = re.sub(r'[^\w]', '', first_upper)
        if clean in ['A', 'B', 'C', 'D']:
            return clean
        choice_map = {
            'FIRST': 'A', 'ONE': 'A', '1': 'A',
            'SECOND': 'B', 'TWO': 'B', '2': 'B', 
            'THIRD': 'C', 'THREE': 'C', '3': 'C',
            'FOURTH': 'D', 'FOUR': 'D', '4': 'D'
        }
        return choice_map.get(first_upper, "A")
    
    elif dataset_type == "multinli":
        first_lower = first_word.lower()
        if first_lower.startswith('ent'):
            return "entailment"
        elif first_lower.startswith('neu'):
            return "neutral"  
        elif first_lower.startswith('con'):
            return "contradiction"
        
        entailment_words = ['yes', 'true', 'follows', 'implies', 'supports']
        neutral_words = ['maybe', 'unclear', 'unknown', 'possible']
        contradiction_words = ['no', 'false', 'contradicts', 'opposes']
        
        if any(word in first_lower for word in entailment_words):
            return "entailment"
        elif any(word in first_lower for word in neutral_words):
            return "neutral"
        elif any(word in first_lower for word in contradiction_words):
            return "contradiction"
        return "neutral"
    
    elif dataset_type in ["eli5", "commongen"]:
        # 생성 태스크는 전체 텍스트 반환
        return text.strip()
    
    return first_word

def calculate_bleu_score(prediction: str, reference: str, max_n: int = 4) -> float:
    """BLEU score 계산 (표준 방식)"""
    try:
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # n-gram precision 계산
        scores = []
        for n in range(1, max_n + 1):
            pred_ngrams = Counter([tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)])
            ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
            
            if not pred_ngrams:
                scores.append(0.0)
                continue
                
            overlap = sum((pred_ngrams & ref_ngrams).values())
            precision = overlap / sum(pred_ngrams.values())
            scores.append(precision)
        
        if not any(scores):
            return 0.0
        
        # Geometric mean
        geometric_mean = math.exp(sum(math.log(s) if s > 0 else -float('inf') for s in scores) / len(scores))
        
        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens))) if pred_tokens else 0.0
        
        return bp * geometric_mean
        
    except:
        return 0.0

def calculate_rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L score 계산 (Longest Common Subsequence)"""
    try:
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # LCS 계산
        def lcs_length(a, b):
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs_len = lcs_length(pred_tokens, ref_tokens)
        
        # ROUGE-L = 2 * precision * recall / (precision + recall)
        precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
        
    except:
        return 0.0

def exact_match_score(prediction: str, ground_truth: str, dataset_type: Optional[str] = None) -> bool:
    """정확한 일치 여부 - 데이터셋별 적절한 평가"""
    try:
        pred = extract_final_answer(prediction, dataset_type)
        true = extract_final_answer(ground_truth, dataset_type)
        
        # 분류 태스크는 정확한 매칭
        if dataset_type in ["strategyqa", "logiqa", "multinli"]:
            pred_norm = normalize_answer(pred)
            true_norm = normalize_answer(true)
            return pred_norm == true_norm
        
        # 수학 문제는 숫자 비교
        elif dataset_type == "gsm8k":
            try:
                pred_num = float(normalize_answer(pred)) if normalize_answer(pred) else 0.0
                true_num = float(normalize_answer(true)) if normalize_answer(true) else 0.0
                return abs(pred_num - true_num) < 1e-6
            except ValueError:
                return False
        
        # 생성 태스크는 ROUGE-L 기반 (더 엄격하게)
        elif dataset_type in ["eli5", "commongen"]:
            rouge_l = calculate_rouge_l(pred, true)
            # ROUGE-L 0.3 이상이면 정답 (표준적인 임계값)
            return rouge_l >= 0.3
        
        # 기본값
        return normalize_answer(pred) == normalize_answer(true)
        
    except Exception as e:
        logging.debug(f"Exact match error: {e}")
        return False

def calculate_accuracy(predictions: List[str], targets: List[str], 
                      dataset_type: Optional[str] = None, 
                      verbose: bool = True) -> float:
    """정확도 계산 - 적절한 메트릭 사용"""
    if not predictions or not targets:
        return 0.0
    
    if len(predictions) != len(targets):
        if verbose:
            print(f"⚠️ Length mismatch: predictions={len(predictions)}, targets={len(targets)}")
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
        metric_name = get_metric_name(dataset_type)
        print(f"🎯 {metric_name}: {correct_count}/{total_count} = {accuracy:.4f} (type: {dataset_type or 'unknown'})")
    
    return accuracy

def get_detailed_metrics(predictions: List[str], targets: List[str], 
                        dataset_type: Optional[str] = None) -> Dict[str, float]:
    """상세한 메트릭 계산 (연구용)"""
    if not predictions or not targets:
        return {}
    
    min_len = min(len(predictions), len(targets))
    predictions = predictions[:min_len]
    targets = targets[:min_len]
    
    if dataset_type is None:
        dataset_type = detect_dataset_type(targets)
    
    metrics = {}
    
    # 기본 정확도
    metrics['accuracy'] = calculate_accuracy(predictions, targets, dataset_type, verbose=False)
    
    # 생성 태스크의 경우 추가 메트릭
    if dataset_type in ["eli5", "commongen"]:
        bleu_scores = []
        rouge_scores = []
        
        for pred, target in zip(predictions, targets):
            pred_clean = extract_final_answer(pred, dataset_type)
            target_clean = extract_final_answer(target, dataset_type)
            
            bleu_scores.append(calculate_bleu_score(pred_clean, target_clean))
            rouge_scores.append(calculate_rouge_l(pred_clean, target_clean))
        
        metrics['bleu_4'] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        metrics['rouge_l'] = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
        
        # CommonGen의 경우 개념 커버리지도 계산 가능
        if dataset_type == "commongen":
            metrics['concept_coverage'] = calculate_concept_coverage(predictions, targets)
    
    return metrics

def calculate_concept_coverage(predictions: List[str], targets: List[str]) -> float:
    """CommonGen용 개념 커버리지 계산"""
    # 간단한 구현 - 실제로는 더 복잡할 수 있음
    coverages = []
    
    for pred, target in zip(predictions, targets):
        pred_words = set(pred.lower().split())
        target_words = set(target.lower().split())
        
        if target_words:
            coverage = len(pred_words & target_words) / len(target_words)
            coverages.append(coverage)
    
    return sum(coverages) / len(coverages) if coverages else 0.0

def get_metric_name(dataset_type: Optional[str]) -> str:
    """데이터셋별 주요 메트릭 이름"""
    metric_map = {
        'eli5': 'ROUGE-L based Accuracy',
        'commongen': 'ROUGE-L based Accuracy', 
        'strategyqa': 'Exact Match',
        'gsm8k': 'Exact Match',
        'logiqa': 'Exact Match',
        'multinli': 'Exact Match'
    }
    return metric_map.get(dataset_type, 'Accuracy')

def detect_dataset_type(targets: List[str]) -> Optional[str]:
    """데이터셋 타입 자동 감지 (기존 로직 유지)"""
    if not targets:
        return None
    
    sample_size = min(max(len(targets) // 10, 5), 20)
    sample = targets[:sample_size]
    
    counters = {
        'strategyqa': 0, 'gsm8k': 0, 'logiqa': 0, 
        'multinli': 0, 'eli5': 0, 'commongen': 0
    }
    
    for target in sample:
        target_str = str(target).lower().strip()
        word_count = len(target_str.split())
        
        if target_str in ['yes', 'no', 'true', 'false', '1', '0']:
            counters['strategyqa'] += 1
        elif re.match(r'^-?\d+(\.\d+)?$', target_str):
            counters['gsm8k'] += 1
        elif target_str.upper() in ['A', 'B', 'C', 'D']:
            counters['logiqa'] += 1
        elif target_str in ['entailment', 'neutral', 'contradiction']:
            counters['multinli'] += 1
        elif word_count >= 20:
            counters['eli5'] += 1
        elif 5 <= word_count < 20 and '.' in target_str:
            counters['commongen'] += 1
    
    threshold = sample_size * 0.4
    for dataset_type, count in counters.items():
        if count >= threshold:
            return dataset_type
    
    # 길이 기반 추가 판단
    avg_length = sum(len(str(target).split()) for target in sample) / len(sample)
    if avg_length >= 25:
        return 'eli5'
    elif 8 <= avg_length < 25:
        return 'commongen'
    
    return None