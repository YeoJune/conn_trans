# dataset/strategyqa_dataset.py
from datasets import load_dataset
from .base_dataset import BaseReasoningDataset
import torch

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
        """원본 데이터 항목을 그대로 반환 (전처리 없음)"""
        # 기본 검증만 수행
        question = item.get('question', '').strip()
        answer = item.get('answer', '')
        
        if not question:
            return None
            
        return item  # ✅ 원본 데이터 그대로 반환
    
    def _tokenize_item(self, item):
        """StrategyQA 전용 토크나이징 - BaseDataset 오버라이드"""
        
        # 1. 질문과 답변 추출
        question = item.get('question', '').strip()
        answer = item.get('answer', '')
        
        # 2. 답변 정규화
        if isinstance(answer, str):
            if answer.lower() in ['yes', 'true']:
                target_text = "Yes"
            elif answer.lower() in ['no', 'false']:
                target_text = "No"
            else:
                target_text = "No"  # 기본값
        elif isinstance(answer, bool):
            target_text = "Yes" if answer else "No"
        else:
            target_text = "No"
        
        # 3. 입력 텍스트 구성
        input_text = f"{self.task_prefix}: {question}"
        
        # 4. 토크나이징
        src_inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )
        
        tgt_inputs = self.tokenizer(
            target_text,
            max_length=self.answer_max_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )
        
        # 5. T5 방식 decoder_input_ids
        decoder_input_ids = self._create_decoder_input_ids(tgt_inputs.input_ids.squeeze())
        
        # 6. Labels
        labels = tgt_inputs.input_ids.squeeze().clone()
        
        return {
            'input_ids': src_inputs.input_ids.squeeze(),
            'attention_mask': src_inputs.attention_mask.squeeze(),
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': tgt_inputs.attention_mask.squeeze(),
            'labels': labels,
            'target_text': target_text,
            # 메타데이터 추가
            'question': question,
            'original_answer': answer,
            'decomposition': item.get('decomposition', []),
            'evidence': item.get('evidence', [])
        }
    
    def _create_decoder_input_ids(self, target_ids):
        """T5 방식 decoder input 생성"""
        if len(target_ids.shape) == 0:
            target_ids = target_ids.unsqueeze(0)
            
        start_token = getattr(self.tokenizer, 'decoder_start_token_id', self.tokenizer.pad_token_id)
        
        decoder_input_ids = torch.cat([
            torch.tensor([start_token]), 
            target_ids[:-1]
        ])
        return decoder_input_ids
    
    def _is_valid_item(self, item):
        """StrategyQA 검증"""
        question = item.get('question', '').strip()
        answer = item.get('answer', '')
        
        return (
            len(question) > 0 and
            len(question.split()) >= 3 and  # 최소 3단어
            answer is not None
        )
    
    def _get_default_answer(self):
        return "No"
    
    def verify_split_integrity(self):
        """데이터 검증"""
        print(f"\n🔍 StrategyQA {self.split} Split Verification")
        print(f"📊 Total samples: {len(self.data)}")
        
        # 샘플 확인
        answers = []
        for i in range(min(10, len(self.data))):
            try:
                item = self.__getitem__(i)
                answers.append(item['target_text'])
                
                if i < 3:  # 첫 3개 샘플 출력
                    print(f"\nSample {i+1}:")
                    print(f"  Question: {item.get('question', 'N/A')}")
                    print(f"  Target: {item['target_text']}")
                    print(f"  Input shape: {item['input_ids'].shape}")
                    print(f"  Decoder shape: {item['decoder_input_ids'].shape}")
                    
            except Exception as e:
                print(f"❌ Error in item {i}: {e}")
                import traceback
                traceback.print_exc()
        
        from collections import Counter
        answer_dist = Counter(answers)
        print(f"\n🎯 Answer distribution: {answer_dist}")
        
        return answer_dist