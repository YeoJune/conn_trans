# dataset/strategyqa_dataset.py
from datasets import load_dataset
from .base_dataset import BaseReasoningDataset

class StrategyQADataset(BaseReasoningDataset):
    @property
    def dataset_name(self):
        return "StrategyQA"
    
    def _load_raw_data(self):
        """StrategyQA 데이터 로딩 - 간소화된 접근"""
        # 주요 소스들
        sources = [
            "ChilleD/StrategyQA",
            "wics/strategy-qa", 
            "voidful/StrategyQA"
        ]
        
        for source in sources:
            try:
                print(f"🔍 Trying {source}...")
                
                # 직접 split 시도
                dataset = load_dataset(source, split=self.split)
                print(f"✅ Loaded {source} {self.split}: {len(dataset)} examples")
                return dataset
                
            except Exception as e:
                print(f"❌ {source} failed: {str(e)[:80]}...")
                
                # split이 없는 경우 전체 로딩 후 수동 분할 시도
                try:
                    full_dataset = load_dataset(source)
                    print(f"🔍 Available splits: {list(full_dataset.keys())}")
                    
                    # train이 있으면 수동 분할
                    if "train" in full_dataset:
                        train_data = full_dataset["train"]
                        return self._create_split(train_data)
                    elif "test" in full_dataset:
                        test_data = full_dataset["test"]  
                        return self._create_split(test_data)
                        
                except Exception as e2:
                    print(f"   Manual split also failed: {str(e2)[:50]}...")
                    continue
        
        raise RuntimeError("Failed to load StrategyQA from any source")
    
    def _create_split(self, dataset):
        """데이터를 train/test로 분할"""
        total_size = len(dataset)
        
        if self.split == "train":
            # 처음 80%를 train으로
            end_idx = int(total_size * 0.8)
            return dataset.select(range(end_idx))
        else:
            # 나머지 20%를 test로
            start_idx = int(total_size * 0.8)
            return dataset.select(range(start_idx, total_size))
    
    def _process_item(self, item, idx):
        """StrategyQA 아이템 처리 - 단순화"""
        
        # 질문 추출
        question = item.get('question', '').strip()
        if not question:
            print(f"⚠️ StrategyQA item {idx}: missing question")
            return None
        
        # 답변 처리 - StrategyQA는 보통 이미 "Yes"/"No" 문자열
        answer = item.get('answer', '')
        
        # 답변 정규화
        if isinstance(answer, str):
            answer_clean = answer.strip()
            if answer_clean.lower() in ['yes', 'true', '1']:
                target_text = "Yes"
            elif answer_clean.lower() in ['no', 'false', '0']:
                target_text = "No"
            else:
                # 예상치 못한 답변
                print(f"⚠️ StrategyQA item {idx}: unexpected answer '{answer}'")
                target_text = "No"  # 기본값
        elif isinstance(answer, bool):
            # Boolean인 경우
            target_text = "Yes" if answer else "No"
        else:
            print(f"⚠️ StrategyQA item {idx}: unexpected answer type {type(answer)}: {answer}")
            target_text = "No"
        
        # 입력 텍스트 구성
        input_text = f"{self.task_prefix}: {question}"
        
        # 추가 정보 (있으면 포함)
        decomposition = item.get('decomposition', [])
        evidence = item.get('evidence', [])
        
        return {
            'input_text': input_text,
            'target_text': target_text,
            'metadata': {
                'question': question,
                'original_answer': answer,
                'decomposition': decomposition,
                'evidence': evidence,
                'index': idx
            }
        }
    
    def _get_default_answer(self):
        return "No"
    
    def _is_valid_item(self, item):
        """StrategyQA 검증 - 단순화"""
        if not super()._is_valid_item(item):
            return False
        
        # 질문이 합리적인 길이인지만 확인
        question = item.get('metadata', {}).get('question', '')
        target = item.get('target_text', '')
        
        # 최소 요구사항
        return (
            len(question.split()) >= 3 and  # 최소 3단어
            target in ['Yes', 'No']         # 유효한 답변
        )
    
    def verify_split_integrity(self):
        """데이터 무결성 검증 - 간소화"""
        print(f"\n🔍 StrategyQA {self.split} Split Verification")
        print(f"📊 Total samples: {len(self.data)}")
        
        if len(self.data) == 0:
            print("🚨 ERROR: No data loaded!")
            return {}
        
        # 샘플 확인
        answers = []
        question_lengths = []
        
        sample_size = min(100, len(self.data))
        
        for i in range(sample_size):
            try:
                item = self.__getitem__(i)
                answers.append(item['target_text'])
                
                # 질문 길이 확인
                question = item.get('metadata', {}).get('question', '')
                question_lengths.append(len(question.split()))
                
            except Exception as e:
                print(f"❌ Error in item {i}: {e}")
        
        from collections import Counter
        answer_dist = Counter(answers)
        
        print(f"🎯 Answer distribution (sample {sample_size}): {answer_dist}")
        print(f"📏 Avg question length: {sum(question_lengths)/len(question_lengths):.1f} words")
        
        # Yes/No 비율 확인
        yes_count = answer_dist.get('Yes', 0)
        no_count = answer_dist.get('No', 0)
        total_count = yes_count + no_count
        
        if total_count > 0:
            yes_ratio = yes_count / total_count
            print(f"📈 Yes/No ratio: {yes_ratio:.2f} / {1-yes_ratio:.2f}")
            
            if 0.2 <= yes_ratio <= 0.8:
                print("✅ Balanced Yes/No distribution")
            else:
                print(f"⚠️ Imbalanced distribution (Yes: {yes_ratio:.2f})")
        
        # 샘플 출력
        print(f"\n📋 Sample questions:")
        for i in range(min(3, len(self.data))):
            item = self.__getitem__(i)
            question = item.get('metadata', {}).get('question', '')
            answer = item['target_text']
            print(f"  {i+1}. Q: {question[:50]}...")
            print(f"     A: {answer}")
        
        return answer_dist