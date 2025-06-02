# dataset/strategyqa_dataset.py
from datasets import load_dataset
from .base_dataset import BaseReasoningDataset

class StrategyQADataset(BaseReasoningDataset):
    @property
    def dataset_name(self):
        return "StrategyQA"
    
    def _load_raw_data(self):
        # 🔧 FIX: 다양한 소스 시도하고 proper split 처리
        sources = [
            "wics/strategy-qa", 
            "voidful/StrategyQA", 
            "ChilleD/StrategyQA"
        ]
        
        for source in sources:
            try:
                # 먼저 사용 가능한 split 확인
                try:
                    dataset_info = load_dataset(source)
                    available_splits = list(dataset_info.keys())
                    print(f"🔍 Available splits in {source}: {available_splits}")
                    
                    # 요청된 split이 있는지 확인
                    if self.split in available_splits:
                        dataset = load_dataset(source, split=self.split)
                        print(f"✅ Loaded {source} {self.split}: {len(dataset)} examples")
                        return dataset
                    elif "test" in available_splits:
                        # test만 있으면 수동 분할
                        print(f"⚠️ {self.split} not found, using test and manual split")
                        full_dataset = load_dataset(source, split="test")
                        return self._manual_split(full_dataset)
                    else:
                        print(f"❌ No suitable splits found in {source}")
                        continue
                        
                except Exception as e:
                    # DatasetDict가 아닌 경우 직접 split 시도
                    dataset = load_dataset(source, split=self.split)
                    if dataset is not None:
                        print(f"✅ Loaded {source} {self.split}: {len(dataset)} examples")
                        return dataset
                    
            except Exception as e:
                print(f"❌ Failed to load {source}: {str(e)[:100]}...")
                continue
        
        raise RuntimeError("Failed to load StrategyQA from any source")
    
    def _manual_split(self, full_dataset):
        """수동으로 train/test 분할"""
        total_size = len(full_dataset)
        train_size = int(total_size * 0.8)
        
        if self.split == "train":
            return full_dataset.select(range(train_size))
        else:  # test, validation
            return full_dataset.select(range(train_size, total_size))
    
    def _process_item(self, item, idx):
        # 🔧 FIX: 필드명 정규화 및 타입 처리
        question = item.get('question', '').strip()
        
        # 🚨 중요: answer 필드 타입 확인 및 처리
        answer = item.get('answer')
        
        # Boolean 타입 처리
        if isinstance(answer, bool):
            target_text = "Yes" if answer else "No"
        elif isinstance(answer, str):
            # 문자열인 경우 정규화
            answer_lower = answer.lower().strip()
            if answer_lower in ['true', 'yes', '1']:
                target_text = "Yes"
            elif answer_lower in ['false', 'no', '0']:
                target_text = "No"
            else:
                print(f"⚠️ StrategyQA item {idx}: unexpected answer string: '{answer}'")
                target_text = "No"  # 기본값
        elif isinstance(answer, (int, float)):
            # 숫자인 경우
            target_text = "Yes" if answer > 0 else "No"
        else:
            print(f"⚠️ StrategyQA item {idx}: unexpected answer type: {type(answer)} = {answer}")
            target_text = "No"  # 기본값
        
        # 🔧 FIX: 입력 텍스트 형식 개선
        input_text = f"{self.task_prefix}: {question}"
        
        return {
            'input_text': input_text,
            'target_text': target_text,
            'metadata': {
                'question': question,
                'original_answer': answer,
                'answer_type': type(answer).__name__,
                'index': idx
            }
        }
    
    def _get_default_answer(self):
        return "No"
    
    def _is_valid_item(self, item):
        """StrategyQA 특화 검증"""
        base_valid = super()._is_valid_item(item)
        if not base_valid:
            return False
        
        # 질문이 최소 길이를 만족하는지 확인
        metadata = item.get('metadata', {})
        question = metadata.get('question', '')
        
        if len(question.split()) < 3:  # 최소 3단어
            return False
            
        return True
    
    def verify_split_integrity(self):
        """데이터 분할 무결성 검증"""
        print(f"\n🔍 StrategyQA {self.split} Split Verification")
        print(f"📊 Total samples: {len(self.data)}")
        
        # 정답 분포 및 타입 확인
        answers = []
        answer_types = []
        original_answers = []
        
        for i in range(min(100, len(self.data))):
            try:
                item = self.__getitem__(i)
                answers.append(item['target_text'])
                
                # 메타데이터에서 원본 답변 정보 수집
                metadata = item.get('metadata', {})
                answer_types.append(metadata.get('answer_type', 'unknown'))
                original_answers.append(metadata.get('original_answer'))
                
            except Exception as e:
                print(f"❌ Error in item {i}: {e}")
        
        from collections import Counter
        answer_dist = Counter(answers)
        type_dist = Counter(answer_types)
        original_dist = Counter(original_answers)
        
        print(f"🎯 Answer distribution (first 100): {answer_dist}")
        print(f"🔢 Answer types: {type_dist}")
        print(f"📋 Original answers sample: {dict(list(original_dist.items())[:5])}")
        
        # 검증
        if len(answer_dist) == 1:
            print("🚨 WARNING: All answers are the same! This suggests a data processing error.")
        elif max(answer_dist.values()) > 90:  # 90% 이상이 같은 답
            print("⚠️ WARNING: Answer distribution is highly skewed.")
        else:
            print("✅ Answer distribution looks reasonable.")
        
        # Yes/No 비율이 합리적인지 확인
        yes_ratio = answer_dist.get('Yes', 0) / max(sum(answer_dist.values()), 1)
        print(f"📈 Yes ratio: {yes_ratio:.2f}")
        
        if 0.2 <= yes_ratio <= 0.8:
            print("✅ Yes/No ratio is balanced.")
        else:
            print(f"⚠️ WARNING: Yes/No ratio seems unbalanced (Yes: {yes_ratio:.2f})")
        
        return answer_dist