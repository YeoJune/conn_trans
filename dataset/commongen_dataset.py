# dataset/commongen_dataset.py
from datasets import load_dataset
from .base_dataset import BaseReasoningDataset

class CommonGenDataset(BaseReasoningDataset):
    """CommonGen dataset for concept-to-text reasoning and generation"""
    
    @property
    def dataset_name(self):
        return "CommonGen"
    
    def _load_raw_data(self):
        """Load CommonGen dataset - 파이프라인 호환성 개선"""
        try:
            # 🔧 FIX: 더 안정적인 소스부터 시도
            dataset = load_dataset("common_gen", split=self.split)
            print(f"✅ Loaded CommonGen {self.split}: {len(dataset)} examples")
            return dataset
            
        except Exception as e1:
            print(f"❌ Failed to load 'common_gen': {e1}")
            
            try:
                # GEM 버전 시도
                if self.split == "validation":
                    dataset = load_dataset("gem", "common_gen", split="validation")
                else:
                    dataset = load_dataset("gem", "common_gen", split=self.split)
                print(f"✅ Loaded CommonGen from GEM: {len(dataset)} examples")
                return dataset
                
            except Exception as e2:
                print(f"❌ Failed to load from GEM: {e2}")
                
                # 🔧 최후 수단: 작은 샘플 생성
                print("⚠️ Using synthetic CommonGen data for testing")
                return self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """테스트용 합성 데이터 생성"""
        synthetic_samples = [
            {"concepts": ["dog", "park", "run"], "target": "The dog runs in the park."},
            {"concepts": ["cat", "sleep", "bed"], "target": "The cat sleeps on the bed."},
            {"concepts": ["car", "road", "drive"], "target": "The car drives on the road."},
            {"concepts": ["book", "read", "library"], "target": "I read a book in the library."},
            {"concepts": ["phone", "call", "friend"], "target": "I call my friend on the phone."},
        ] * 200  # 1000개 샘플 생성
        
        return synthetic_samples
    
    def _process_item(self, item, idx):
        """Process CommonGen item - T5 파이프라인 호환"""
        # 🔧 FIX: 다양한 필드명 지원
        concepts = self._extract_concepts(item)
        target = self._extract_target(item)
        
        # 🔧 FIX: 개념 텍스트 구성 개선
        concept_text = self._format_concepts(concepts)
        
        # 🔧 FIX: T5 스타일 프롬프트 구성
        input_text = f"{self.task_prefix}: {concept_text}"
        
        # 🔧 FIX: 타겟 검증 및 정리
        target_text = self._clean_target(target)
        
        return {
            'input_text': input_text,
            'target_text': target_text,
            'metadata': {
                'concepts': concepts,
                'original_target': target,
                'num_concepts': len(concepts),
                'index': idx
            }
        }
    
    def _extract_concepts(self, item):
        """다양한 형식의 concepts 추출"""
        # 여러 가능한 필드명 시도
        for field in ['concepts', 'concept_set', 'inputs']:
            if field in item:
                concepts = item[field]
                break
        else:
            concepts = []
        
        # 문자열인 경우 리스트로 변환
        if isinstance(concepts, str):
            # 쉼표, 공백, 기타 구분자로 분할
            import re
            concepts = re.split(r'[,\s]+', concepts.strip())
            concepts = [c.strip() for c in concepts if c.strip()]
        elif not isinstance(concepts, list):
            concepts = [str(concepts)] if concepts else []
        
        # 빈 요소 제거
        concepts = [c.strip() for c in concepts if c and c.strip()]
        
        return concepts[:6]  # 최대 6개 개념까지만
    
    def _extract_target(self, item):
        """다양한 형식의 target 추출"""
        for field in ['target', 'scene', 'targets', 'text']:
            if field in item:
                target = item[field]
                break
        else:
            target = ""
        
        # 리스트인 경우 첫 번째 요소 사용
        if isinstance(target, list):
            target = target[0] if target else ""
        
        return str(target).strip()
    
    def _format_concepts(self, concepts):
        """개념들을 자연스럽게 포맷팅"""
        if not concepts:
            return "unknown concepts"
        
        if len(concepts) == 1:
            return f"the concept '{concepts[0]}'"
        elif len(concepts) == 2:
            return f"the concepts '{concepts[0]}' and '{concepts[1]}'"
        else:
            formatted = "', '".join(concepts[:-1])
            return f"the concepts '{formatted}', and '{concepts[-1]}'"
    
    def _clean_target(self, target):
        """타겟 텍스트 정리"""
        if not target:
            return "No meaningful connection found."
        
        # 기본 정리
        cleaned = target.strip()
        
        # 너무 짧으면 기본 응답
        if len(cleaned.split()) < 3:
            return "No meaningful connection found."
        
        # 너무 길면 자르기 (문장 단위)
        if len(cleaned.split()) > 50:
            sentences = cleaned.split('. ')
            if len(sentences) > 1:
                cleaned = sentences[0] + '.'
            else:
                words = cleaned.split()[:50]
                cleaned = ' '.join(words) + '...'
        
        return cleaned
    
    def _get_default_answer(self):
        return "No meaningful connection found."
    
    def _is_valid_item(self, item):
        """CommonGen 특화 검증 - 더 실용적"""
        base_valid = super()._is_valid_item(item)
        if not base_valid:
            return False
        
        metadata = item.get('metadata', {})
        concepts = metadata.get('concepts', [])
        target = item.get('target_text', '')
        
        # 🔧 FIX: 더 관대한 검증
        if len(concepts) < 1:  # 최소 1개 개념
            return False
        if len(concepts) > 8:   # 너무 많으면 복잡
            return False
        if len(target.strip()) < 5:  # 너무 짧은 타겟
            return False
        if len(target.split()) > 60:  # 너무 긴 타겟
            return False
            
        return True
    
    def verify_dataset_compatibility(self):
        """파이프라인 호환성 검증"""
        print(f"\n🔍 CommonGen Pipeline Compatibility Check")
        
        if len(self.data) == 0:
            print("❌ No data loaded!")
            return False
        
        try:
            sample = self.__getitem__(0)
            
            # 필수 필드 확인
            required_fields = ['input_ids', 'attention_mask', 'decoder_input_ids', 'labels', 'target_text']
            missing_fields = [f for f in required_fields if f not in sample]
            
            if missing_fields:
                print(f"❌ Missing fields: {missing_fields}")
                return False
            
            print(f"✅ All required fields present")
            print(f"✅ Input length: {len(sample['input_ids'])} tokens") 
            print(f"✅ Target length: {len(sample['labels'])} tokens")
            print(f"✅ Sample input: '{sample.get('target_text', '')[:50]}...'")
            
            # 개념 추출 테스트
            metadata = sample.get('metadata', {})
            concepts = metadata.get('concepts', [])
            print(f"✅ Concepts extracted: {concepts}")
            
            return True
            
        except Exception as e:
            print(f"❌ Compatibility test failed: {e}")
            return False