# dataset/logiqa_dataset.py
from datasets import load_dataset
from .base_dataset import BaseReasoningDataset

class LogiQADataset(BaseReasoningDataset):
    @property
    def dataset_name(self):
        return "LogiQA"
    
    def _load_raw_data(self):
        # Try multiple sources with proper split handling
        sources = ["lucasmccabe/logiqa", "logiqa"]
        
        for source in sources:
            try:
                # 먼저 전체 데이터셋 정보 확인
                dataset_info = load_dataset(source)
                print(f"🔍 Available splits in {source}: {list(dataset_info.keys())}")
                
                # 요청된 split이 존재하는지 확인
                if self.split in dataset_info:
                    dataset = load_dataset(source, split=self.split)
                    print(f"✅ Loaded {source} {self.split}: {len(dataset)} examples")
                    return dataset
                else:
                    # split이 없으면 train을 사용하고 수동으로 분할
                    print(f"⚠️ {self.split} not found in {source}, using train and manual split")
                    full_dataset = load_dataset(source, split="train")
                    
                    # 수동 분할 (80% train, 20% test)
                    if self.split == "train":
                        split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
                        return split_dataset["train"]
                    elif self.split in ["test", "validation"]:
                        split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
                        return split_dataset["test"]
                    else:
                        return full_dataset
                        
            except Exception as e:
                print(f"❌ Failed to load {source}: {e}")
                continue
        
        raise RuntimeError("Failed to load LogiQA from any source")
    
    def _process_item(self, item, idx):
        # 🔧 FIX: 올바른 필드명 사용
        context = item.get('context', '').strip()
        question = item.get('query', item.get('question', '')).strip()  # query가 정확한 필드명
        options = item.get('options', item.get('choices', []))
        answer = item.get('correct_option', item.get('answer', item.get('label', 0)))  # correct_option이 정확한 필드명
        
        # Build input text
        input_parts = [f"{self.task_prefix}:"]
        
        if context:
            input_parts.append(f"Context: {context}")
        
        input_parts.append(f"Question: {question}")
        
        if options:
            options_text = " ".join([f"{chr(65+i)}) {opt.strip()}" 
                                   for i, opt in enumerate(options)])
            input_parts.append(f"Options: {options_text}")
        
        # 🔧 FIX: 정답 처리 개선
        if isinstance(answer, int) and 0 <= answer < len(options):
            target_text = chr(65 + answer)  # 0->A, 1->B, etc.
        elif isinstance(answer, str) and len(answer) == 1 and answer.upper() in 'ABCD':
            target_text = answer.upper()
        else:
            # 🚨 디버깅: 예상치 못한 답변 형식 로깅
            print(f"⚠️ LogiQA item {idx}: unexpected answer format: {answer} (type: {type(answer)})")
            target_text = "A"  # 기본값
        
        return {
            'input_text': " ".join(input_parts),
            'target_text': target_text,
            'metadata': {
                'question': question,
                'context': context,
                'options': options,
                'original_answer': answer,
                'index': idx
            }
        }
    
    def _get_default_answer(self):
        return "A"
    
    def _is_valid_item(self, item):
        """LogiQA 특화 검증"""
        base_valid = super()._is_valid_item(item)
        if not base_valid:
            return False
        
        # 추가 검증: 옵션과 정답이 유효한지 확인
        metadata = item.get('metadata', {})
        options = metadata.get('options', [])
        original_answer = metadata.get('original_answer')
        
        # 옵션이 2개 이상 있어야 함
        if len(options) < 2:
            return False
        
        # 정답이 유효한 범위에 있어야 함
        if isinstance(original_answer, int) and not (0 <= original_answer < len(options)):
            return False
            
        return True
    
    def verify_split_integrity(self):
        """데이터 분할 무결성 검증"""
        print(f"\n🔍 LogiQA {self.split} Split Verification")
        print(f"📊 Total samples: {len(self.data)}")
        
        # 정답 분포 확인
        answers = []
        for i in range(min(100, len(self.data))):
            try:
                item = self.__getitem__(i)
                answers.append(item['target_text'])
            except Exception as e:
                print(f"❌ Error in item {i}: {e}")
        
        from collections import Counter
        answer_dist = Counter(answers)
        print(f"🎯 Answer distribution (first 100): {answer_dist}")
        
        # 분포가 너무 편향되어 있으면 경고
        if len(answer_dist) == 1:
            print("🚨 WARNING: All answers are the same! This suggests a data loading error.")
        elif max(answer_dist.values()) > 80:  # 80% 이상이 같은 답
            print("⚠️ WARNING: Answer distribution is highly skewed.")
        else:
            print("✅ Answer distribution looks reasonable.")
        
        return answer_dist