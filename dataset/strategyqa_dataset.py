# dataset/strategyqa_dataset.py
from datasets import load_dataset
from .base_dataset import BaseReasoningDataset

class StrategyQADataset(BaseReasoningDataset):
    @property
    def dataset_name(self):
        return "StrategyQA"
    
    def _load_raw_data(self):
        """StrategyQA 데이터 로딩"""
        sources = [
            "ChilleD/StrategyQA",
            "wics/strategy-qa"
        ]
        
        for source in sources:
            try:
                dataset = load_dataset(source, split=self.split)
                print(f"✅ Loaded {source} {self.split}: {len(dataset)} examples")
                return dataset
            except Exception as e:
                print(f"❌ {source} failed: {str(e)[:50]}...")
                continue
        
        raise RuntimeError("Failed to load StrategyQA")
    
    def _process_item(self, item, idx):
        """StrategyQA를 추론 가능한 input_text로 변환"""
        
        # 1. 기본 정보 추출
        question = item.get('question', '').strip()
        answer = item.get('answer', '')
        decomposition = item.get('decomposition', [])
        evidence = item.get('evidence', [])
        
        if not question:
            return None
        
        # 2. 답변 정규화
        if isinstance(answer, str):
            if answer.lower() in ['yes', 'true', '1']:
                target_text = "Yes"
            elif answer.lower() in ['no', 'false', '0']:
                target_text = "No"
            else:
                target_text = "No"
        elif isinstance(answer, bool):
            target_text = "Yes" if answer else "No"
        else:
            target_text = "No"
        
        # 3. 🔥 핵심: 구조화된 input_text 생성
        input_parts = [f"{self.task_prefix}: {question}"]
        
        # Decomposition이 있으면 추가 (추론 단계)
        if decomposition and len(decomposition) > 0:
            input_parts.append("Reasoning steps:")
            for i, step in enumerate(decomposition, 1):
                input_parts.append(f"{i}. {step}")
        
        # Evidence가 있으면 추가 (증거/맥락)
        if evidence and len(evidence) > 0:
            input_parts.append("Evidence:")
            for i, fact in enumerate(evidence, 1):
                input_parts.append(f"- {fact}")
        
        # 최종 질문 반복 (명확성을 위해)
        input_parts.append(f"Answer (Yes or No): ")
        
        input_text = " ".join(input_parts)
        
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
        """StrategyQA 검증 - 구조화된 데이터 기준"""
        if not super()._is_valid_item(item):
            return False
        
        # 메타데이터 검증
        metadata = item.get('metadata', {})
        question = metadata.get('question', '')
        
        # 기본 요구사항
        return (
            len(question.split()) >= 3 and  # 최소 3단어
            item.get('target_text') in ['Yes', 'No']  # 유효한 답변
        )
    
    def verify_split_integrity(self):
        """데이터 검증 - 구조 품질 확인"""
        print(f"\n🔍 StrategyQA {self.split} Split Verification")
        print(f"📊 Total samples: {len(self.data)}")
        
        # 구조 분석
        has_decomposition = 0
        has_evidence = 0
        answers = []
        input_lengths = []
        
        sample_size = min(50, len(self.data))
        
        for i in range(sample_size):
            try:
                raw_item = self.data[i]  # 원본 데이터
                processed_item = self.__getitem__(i)  # 처리된 데이터
                
                # 구조 정보 수집
                decomposition = raw_item.get('metadata', {}).get('decomposition', [])
                evidence = raw_item.get('metadata', {}).get('evidence', [])
                
                if decomposition and len(decomposition) > 0:
                    has_decomposition += 1
                if evidence and len(evidence) > 0:
                    has_evidence += 1
                
                answers.append(processed_item['target_text'])
                input_lengths.append(len(processed_item['input_text']))
                
            except Exception as e:
                print(f"❌ Error in item {i}: {e}")
        
        from collections import Counter
        answer_dist = Counter(answers)
        
        print(f"🎯 Answer distribution: {answer_dist}")
        print(f"🧩 Decomposition coverage: {has_decomposition}/{sample_size} ({has_decomposition/sample_size*100:.1f}%)")
        print(f"📚 Evidence coverage: {has_evidence}/{sample_size} ({has_evidence/sample_size*100:.1f}%)")
        print(f"📏 Avg input length: {sum(input_lengths)/len(input_lengths):.0f} chars")
        
        # 샘플 출력 (구조 확인)
        print(f"\n📋 Sample inputs:")
        for i in range(min(2, len(self.data))):
            processed_item = self.__getitem__(i)
            print(f"\nSample {i+1}:")
            print(f"Input: {processed_item['input_text'][:200]}...")
            print(f"Target: {processed_item['target_text']}")
        
        # 품질 평가
        if has_decomposition < sample_size * 0.5:
            print("⚠️ WARNING: Low decomposition coverage - model may not learn multi-step reasoning")
        
        if has_evidence < sample_size * 0.5:
            print("⚠️ WARNING: Low evidence coverage - model may not learn fact-based reasoning")
        
        yes_ratio = answer_dist.get('Yes', 0) / max(sum(answer_dist.values()), 1)
        if 0.2 <= yes_ratio <= 0.8:
            print("✅ Balanced Yes/No distribution")
        else:
            print(f"⚠️ Imbalanced distribution (Yes: {yes_ratio:.2f})")
        
        return {
            'answer_distribution': answer_dist,
            'decomposition_coverage': has_decomposition / sample_size,
            'evidence_coverage': has_evidence / sample_size,
            'avg_input_length': sum(input_lengths) / len(input_lengths)
        }