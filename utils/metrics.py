# utils/metrics.py
# 이 파일은 SQuAD의 공식 EM/F1 스코어를 계산하기 위한 로직을 포함할 수 있습니다.
# Hugging Face `datasets.load_metric("squad")`를 사용하거나,
# 공식 evaluate-v1.1.py 또는 evaluate-v2.0.py 스크립트 로직을 참고하여 구현합니다.
# 현재는 간단한 스팬 정확도만 사용하므로, 이 파일은 비워두거나 나중에 구현합니다.

def compute_squad_em_f1(predictions, references):
    # predictions: 모델이 예측한 답변 텍스트 리스트 또는 (example_id, predicted_text) 딕셔너리
    # references: 실제 답변 정보 리스트 (SQuAD 형식)
    # 예시: from datasets import load_metric
    # squad_metric = load_metric("squad") # 또는 "squad_v2"
    # results = squad_metric.compute(predictions=formatted_preds, references=formatted_refs)
    # return results # {'exact_match': ..., 'f1': ...}
    print("⚠️ SQuAD EM/F1 metric computation not fully implemented yet.")
    return {"exact_match": 0.0, "f1": 0.0}