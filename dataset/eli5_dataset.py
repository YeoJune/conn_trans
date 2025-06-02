# dataset/eli5_dataset.py
from datasets import load_dataset
from .base_dataset import BaseReasoningDataset

class ELI5Dataset(BaseReasoningDataset):
    """ELI5 - Explain Like I'm 5 dataset for multi-step reasoning and explanation"""
    
    @property
    def dataset_name(self):
        return "ELI5"
    
    def _load_raw_data(self):
        """Load ELI5 dataset from HuggingFace"""
        try:
            # 🔧 FIX: 더 안정적인 데이터셋 로딩
            if self.split == "train":
                dataset = load_dataset("eli5_category", split="train[:20000]")  # 적당한 크기
            elif self.split in ["test", "validation"]:
                dataset = load_dataset("eli5_category", split="validation1[:2000]")  
            else:
                dataset = load_dataset("eli5_category", split="train[:5000]")  
            
            print(f"✅ Loaded ELI5 {self.split}: {len(dataset)} examples")
            return dataset
            
        except Exception as e:
            print(f"❌ Failed to load ELI5: {e}")
            # 🔧 더 간단한 fallback 
            try:
                dataset = load_dataset("eli5", split="train_asks[:1000]")
                print(f"⚠️ Using ELI5 fallback: {len(dataset)} examples")
                return dataset
            except:
                raise RuntimeError("Failed to load ELI5 from any source")
    
    def _process_item(self, item, idx):
        """Process ELI5 item - 파이프라인 호환성 개선"""
        # 🔧 FIX: 필드명 안전 처리
        title = item.get('title', item.get('question', '')).strip()
        selftext = item.get('selftext', item.get('body', '')).strip()
        
        # 🔧 FIX: answers 구조 안전 처리
        answers_data = item.get('answers', {})
        if isinstance(answers_data, dict):
            answers = answers_data.get('text', [])
        elif isinstance(answers_data, list):
            answers = answers_data
        else:
            answers = []
        
        # 질문 구성 - 더 간결하게
        if selftext and len(selftext) > 20:
            # selftext가 너무 길면 자르기
            if len(selftext) > 200:
                selftext = selftext[:200] + "..."
            question = f"{title} {selftext}"
        else:
            question = title
        
        # 🔧 FIX: 답변 선택 로직 개선
        target_answer = self._select_best_answer(answers)
        
        # 🔧 FIX: 입력 프롬프트 개선 (T5 스타일)
        input_text = f"{self.task_prefix}: {question.strip()}"
        
        return {
            'input_text': input_text,
            'target_text': target_answer,
            'metadata': {
                'question': title,
                'context': selftext,
                'original_length': len(target_answer),
                'index': idx,
                'num_answers': len(answers)  # 디버깅용
            }
        }
    
    def _select_best_answer(self, answers):
        """최적의 답변 선택 로직"""
        if not answers:
            return "I need more information to explain this properly."
        
        # 적절한 길이의 답변 찾기 (50-400 토큰 정도)
        good_answers = []
        for ans in answers:
            ans_clean = str(ans).strip()
            word_count = len(ans_clean.split())
            if 20 <= word_count <= 150:  # 🔧 토큰이 아닌 단어 기준으로 더 보수적
                good_answers.append(ans_clean)
        
        if good_answers:
            return good_answers[0]
        
        # 없으면 첫 번째 답변을 적절히 자르기
        first_answer = str(answers[0]).strip()
        if len(first_answer.split()) > 150:
            words = first_answer.split()[:150]
            return ' '.join(words) + "..."
        
        return first_answer if first_answer else "No good answer available."
    
    def _get_default_answer(self):
        return "I need more information to explain this properly."
    
    def _is_valid_item(self, item):
        """ELI5 특화 검증 - 더 엄격하게"""
        base_valid = super()._is_valid_item(item)
        if not base_valid:
            return False
        
        metadata = item.get('metadata', {})
        question = metadata.get('question', '')
        target = item.get('target_text', '')
        
        # 🔧 FIX: 더 실용적인 검증
        if len(question.strip()) < 5:   # 너무 짧은 질문
            return False
        if len(target.strip()) < 20:    # 너무 짧은 답변  
            return False
        if len(target.split()) > 200:   # 너무 긴 답변 (단어 기준)
            return False
        if "deleted" in target.lower() or "removed" in target.lower():  # 삭제된 답변
            return False
            
        return True
    
    def verify_dataset_compatibility(self):
        """파이프라인 호환성 검증"""
        print(f"\n🔍 ELI5 Pipeline Compatibility Check")
        
        if len(self.data) == 0:
            print("❌ No data loaded!")
            return False
        
        # 첫 번째 샘플로 토크나이저 테스트
        try:
            sample = self.__getitem__(0)
            required_fields = ['input_ids', 'attention_mask', 'decoder_input_ids', 'labels', 'target_text']
            
            for field in required_fields:
                if field not in sample:
                    print(f"❌ Missing field: {field}")
                    return False
                    
            print(f"✅ All required fields present")
            print(f"✅ Input length: {len(sample['input_ids'])} tokens")
            print(f"✅ Target length: {len(sample['labels'])} tokens")
            print(f"✅ Sample target: '{sample['target_text'][:50]}...'")
            
            return True
            
        except Exception as e:
            print(f"❌ Tokenization failed: {e}")
            return False