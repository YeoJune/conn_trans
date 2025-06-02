# dataset/multinli_dataset.py
from datasets import load_dataset
from .base_dataset import BaseReasoningDataset

class MultiNLIDataset(BaseReasoningDataset):
    
    # 🔧 FIX: 올바른 라벨 매핑 (MultiNLI 공식 문서 기준)
    LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
    REVERSE_LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}
    
    @property
    def dataset_name(self):
        return "MultiNLI"
    
    def _load_raw_data(self):
        # 🔧 FIX: 다양한 소스와 올바른 split 이름 시도
        sources = [
            ("nyu-mll/multi_nli", None),
            ("multi_nli", None),
            ("multinli", None)
        ]
        
        for source_name, config in sources:
            try:
                # 🔧 FIX: 올바른 split 매핑
                split_mapping = {
                    "train": "train",
                    "validation": "validation_matched", 
                    "test": "validation_mismatched",  # test set이 없으므로 mismatched를 test로 사용
                    "dev": "validation_matched",
                    "eval": "validation_matched"
                }
                
                actual_split = split_mapping.get(self.split, self.split)
                
                if config:
                    dataset = load_dataset(source_name, config, split=actual_split)
                else:
                    dataset = load_dataset(source_name, split=actual_split)
                
                print(f"✅ Loaded {source_name} {actual_split}: {len(dataset)} examples")
                return dataset
                
            except Exception as e:
                print(f"❌ Failed to load {source_name}: {str(e)[:100]}...")
                continue
        
        raise RuntimeError("Failed to load MultiNLI from any source")
    
    def _process_item(self, item, idx):
        # 🔧 FIX: 필드명 정규화 및 라벨 검증
        premise = item.get('premise', '').strip()
        hypothesis = item.get('hypothesis', '').strip()
        label = item.get('label', -1)
        
        # 🚨 중요: 잘못된 라벨 (-1) 필터링
        if label == -1:
            print(f"⚠️ MultiNLI item {idx}: invalid label (-1), skipping")
            return None
        
        # 🔧 FIX: 라벨 검증 및 변환
        if label not in self.LABEL_MAP:
            print(f"⚠️ MultiNLI item {idx}: unknown label {label}, using neutral")
            label = 1  # neutral as default
        
        target_text = self.LABEL_MAP[label]
        
        # 🔧 FIX: 입력 텍스트 형식 개선
        input_text = (f"{self.task_prefix}: "
                     f"Premise: {premise} "
                     f"Hypothesis: {hypothesis}")
        
        return {
            'input_text': input_text,
            'target_text': target_text,
            'metadata': {
                'premise': premise,
                'hypothesis': hypothesis,
                'genre': item.get('genre', 'unknown'),
                'original_label': label,
                'pair_id': item.get('pairID', f'pair_{idx}'),
                'index': idx
            }
        }
    
    def _is_valid_item(self, item):
        """MultiNLI 특화 검증 - 더 엄격한 필터링"""
        base_valid = super()._is_valid_item(item)
        if not base_valid:
            return False
        
        # 길이 제한 (대용량 데이터셋이므로)
        if len(item['input_text']) > 800:
            return False
        
        # 메타데이터 검증
        metadata = item.get('metadata', {})
        premise = metadata.get('premise', '')
        hypothesis = metadata.get('hypothesis', '')
        
        # premise와 hypothesis가 모두 있어야 함
        if not premise or not hypothesis:
            return False
        
        # 너무 짧은 텍스트 제외
        if len(premise.split()) < 3 or len(hypothesis.split()) < 3:
            return False
            
        return True
    
    def _get_default_answer(self):
        return "neutral"
    
    def verify_split_integrity(self):
        """데이터 분할 무결성 검증"""
        print(f"\n🔍 MultiNLI {self.split} Split Verification")
        print(f"📊 Total samples: {len(self.data)}")
        
        # 라벨 분포 및 장르 분포 확인
        label_dist = {}
        genre_dist = {}
        invalid_labels = 0
        
        sample_size = min(1000, len(self.data))  # 큰 데이터셋이므로 샘플링
        
        for i in range(sample_size):
            try:
                item = self.__getitem__(i)
                target = item['target_text']
                metadata = item.get('metadata', {})
                
                # 라벨 분포
                label_dist[target] = label_dist.get(target, 0) + 1
                
                # 장르 분포
                genre = metadata.get('genre', 'unknown')
                genre_dist[genre] = genre_dist.get(genre, 0) + 1
                
                # 원본 라벨 검증
                original_label = metadata.get('original_label', -1)
                if original_label == -1:
                    invalid_labels += 1
                
            except Exception as e:
                print(f"❌ Error in item {i}: {e}")
        
        print(f"🎯 Label distribution (sample of {sample_size}):")
        for label, count in sorted(label_dist.items()):
            percentage = (count / sample_size) * 100
            print(f"   {label}: {count} ({percentage:.1f}%)")
        
        print(f"📚 Genre distribution (top 5):")
        sorted_genres = sorted(genre_dist.items(), key=lambda x: x[1], reverse=True)
        for genre, count in sorted_genres[:5]:
            percentage = (count / sample_size) * 100
            print(f"   {genre}: {count} ({percentage:.1f}%)")
        
        if invalid_labels > 0:
            print(f"⚠️ WARNING: Found {invalid_labels} items with invalid labels (-1)")
        
        # 검증
        if len(label_dist) != 3:
            print(f"🚨 WARNING: Expected 3 labels, found {len(label_dist)}")
        elif all(label in ["entailment", "neutral", "contradiction"] for label in label_dist.keys()):
            print("✅ All labels are valid NLI categories.")
        else:
            print("🚨 WARNING: Found unexpected label categories!")
        
        # 균형성 검사 (NLI는 보통 비교적 균형잡힌 분포)
        if label_dist:
            max_ratio = max(label_dist.values()) / sum(label_dist.values())
            if max_ratio > 0.6:  # 60% 이상이면 불균형
                print(f"⚠️ WARNING: Label distribution is imbalanced (max: {max_ratio:.1%})")
            else:
                print("✅ Label distribution is reasonably balanced.")
        
        # 샘플 출력
        print(f"\n📋 Sample examples:")
        for i in range(min(3, len(self.data))):
            item = self.__getitem__(i)
            metadata = item.get('metadata', {})
            premise = metadata.get('premise', '')[:50]
            hypothesis = metadata.get('hypothesis', '')[:50]
            target = item['target_text']
            
            print(f"  {i+1}. {target} | P: '{premise}...' H: '{hypothesis}...'")
        
        return {
            'label_distribution': label_dist,
            'genre_distribution': genre_dist,
            'invalid_labels': invalid_labels,
            'sample_size': sample_size
        }