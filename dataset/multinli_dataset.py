# dataset/multinli_dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class MultiNLIDataset(Dataset):
    """MultiNLI Dataset - Multi-Genre Natural Language Inference (433K examples)"""
    
    def __init__(self, tokenizer, config, split="train"):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_seq_len
        self.answer_max_length = getattr(config, 'answer_max_length', 32)
        self.task_prefix = getattr(config, 'task_prefix', 'infer')
        
        # MultiNLI 데이터셋 로드
        print(f"📦 Loading MultiNLI dataset ({split} split)...")
        
        try:
            # HuggingFace에서 공식 MultiNLI 로드
            if split == "validation":
                # MultiNLI는 validation_matched와 validation_mismatched가 있음
                # matched를 기본으로 사용 (같은 도메인)
                dataset = load_dataset("multi_nli", split="validation_matched")
                print(f"✅ Successfully loaded MultiNLI validation_matched")
            elif split == "test":
                # 테스트 셋도 matched 사용
                dataset = load_dataset("multi_nli", split="test_matched")
                print(f"✅ Successfully loaded MultiNLI test_matched")
            else:
                dataset = load_dataset("multi_nli", split=split)
                print(f"✅ Successfully loaded MultiNLI {split}")
                
        except Exception as e:
            print(f"❌ MultiNLI loading failed: {e}")
            raise RuntimeError("Failed to load MultiNLI dataset")
        
        self.data = self._preprocess(dataset)
        print(f"MultiNLI {split}: {len(self.data):,} examples")
    
    def _preprocess(self, dataset):
        """데이터셋을 T5 형식으로 전처리"""
        processed = []
        
        # 라벨 매핑
        label_map = {
            0: "entailment",
            1: "neutral", 
            2: "contradiction"
        }
        
        for item in dataset:
            # 필드 추출
            premise = item.get('premise', '').strip()
            hypothesis = item.get('hypothesis', '').strip()
            label = item.get('label', -1)
            
            # 빈 텍스트 처리
            if not premise:
                premise = "No premise provided."
            if not hypothesis:
                hypothesis = "No hypothesis provided."
            
            # T5 형식: "infer: premise: <premise> hypothesis: <hypothesis>"
            input_text = f"{self.task_prefix}: premise: {premise} hypothesis: {hypothesis}"
            
            # 라벨을 텍스트로 변환
            if label in label_map:
                target_text = label_map[label]
            else:
                # 라벨이 없는 경우 (테스트 셋 등)
                target_text = "unknown"
            
            processed.append({
                'input_text': input_text,
                'target_text': target_text,
                'original_label': label,
                'premise': premise,
                'hypothesis': hypothesis,
                'genre': item.get('genre', 'unknown')  # MultiNLI의 장르 정보
            })
        
        return processed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """MultiNLI T5 전처리"""
        item = self.data[idx]
        
        # 입력 토크나이징
        inputs = self.tokenizer(
            item['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 타겟 토크나이징
        targets = self.tokenizer(
            item['target_text'],
            max_length=self.answer_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': targets.input_ids.squeeze(),
            'target_text': item['target_text'],
            'original_label': item['original_label'],
            'premise': item['premise'],
            'hypothesis': item['hypothesis'],
            'genre': item['genre']
        }