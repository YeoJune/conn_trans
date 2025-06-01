# dataset/multinli_dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class MultiNLIDataset(Dataset):
    """MultiNLI Dataset - 자연어 추론 (T5 최적화)"""
    
    def __init__(self, tokenizer, config, split="train"):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_seq_len
        self.answer_max_length = getattr(config, 'answer_max_length', 32)
        self.task_prefix = getattr(config, 'task_prefix', 'infer')
        
        print(f"📦 Loading MultiNLI dataset ({split} split)...")
        
        try:
            # 표준 MultiNLI 데이터셋
            if split == "validation":
                dataset = load_dataset("multi_nli", split="validation_matched")
                print(f"✅ Successfully loaded MultiNLI validation_matched")
            elif split == "test":
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
        """T5 적합한 전처리"""
        processed = []
        
        # 라벨 매핑
        label_map = {
            0: "entailment",
            1: "neutral", 
            2: "contradiction"
        }
        
        for item in dataset:
            premise = item.get('premise', '').strip()
            hypothesis = item.get('hypothesis', '').strip()
            label = item.get('label', -1)
            
            # 빈 텍스트 건너뛰기
            if not premise or not hypothesis:
                continue
            
            # 입력: "infer: Premise: {premise} Hypothesis: {hypothesis}"
            input_text = f"{self.task_prefix}: Premise: {premise} Hypothesis: {hypothesis}"
            
            # 출력: "entailment" / "neutral" / "contradiction"
            if label in label_map:
                target_text = label_map[label]
            else:
                continue  # 라벨이 없으면 스킵
            
            processed.append({
                'input_text': input_text,
                'target_text': target_text,
                'premise': premise,
                'hypothesis': hypothesis,
                'original_label': label,
                'genre': item.get('genre', 'unknown')
            })
        
        return processed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """T5용 정확한 데이터 생성"""
        item = self.data[idx]
        
        # 입력 토크나이징
        inputs = self.tokenizer(
            item['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # T5 target 토크나이징
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                item['target_text'],
                max_length=self.answer_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Labels 처리
        target_ids = targets.input_ids.squeeze()
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': target_ids,
            'target_text': item['target_text'],
            'premise': item['premise'],
            'hypothesis': item['hypothesis'],
            'genre': item['genre']
        }