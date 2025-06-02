# dataset/gsm8k_dataset.py
import re
from datasets import load_dataset
from .base_dataset import BaseReasoningDataset

class GSM8KDataset(BaseReasoningDataset):
    @property
    def dataset_name(self):
        return "GSM8K"
    
    def _load_raw_data(self):
        # 🔧 FIX: 여러 소스 시도 및 오류 처리
        sources = [
            ("gsm8k", "main"),
            ("openai/gsm8k", "main"), 
            ("gsm8k", None)  # config 없이 시도
        ]
        
        for source_name, config in sources:
            try:
                if config:
                    dataset = load_dataset(source_name, config, split=self.split)
                else:
                    dataset = load_dataset(source_name, split=self.split)
                
                print(f"✅ Loaded {source_name} {config or ''} {self.split}: {len(dataset)} examples")
                return dataset
                
            except Exception as e:
                print(f"❌ Failed to load {source_name} {config}: {str(e)[:100]}...")
                continue
        
        raise RuntimeError("Failed to load GSM8K from any source")
    
    def _process_item(self, item, idx):
        # 🔧 FIX: 필드명 정규화
        problem = item.get('question', '').strip()
        answer_text = item.get('answer', '').strip()
        
        if not problem or not answer_text:
            print(f"⚠️ GSM8K item {idx}: missing question or answer")
            return None
        
        # 🔧 FIX: 향상된 최종 답 추출
        final_answer = self._extract_final_answer(answer_text, idx)
        
        return {
            'input_text': f"{self.task_prefix}: {problem}",
            'target_text': final_answer,
            'metadata': {
                'problem': problem,
                'full_solution': answer_text,
                'index': idx
            }
        }
    
    def _extract_final_answer(self, answer_text, idx=None):
        """향상된 GSM8K 솔루션에서 최종 답 추출"""
        # 🔧 FIX: 더 robust한 패턴 매칭
        
        # 1. "#### 숫자" 패턴 찾기 (가장 일반적)
        pattern1 = r'####\s*(-?\d+(?:[\.,]\d+)*)'
        match = re.search(pattern1, answer_text)
        if match:
            answer = match.group(1).replace(',', '')  # 콤마 제거
            try:
                # 정수로 변환 가능한지 확인
                if '.' in answer:
                    float_val = float(answer)
                    if float_val.is_integer():
                        return str(int(float_val))
                    else:
                        return answer
                else:
                    return str(int(answer))
            except ValueError:
                pass
        
        # 2. "The answer is 숫자" 패턴
        pattern2 = r'(?:the answer is|answer:\s*)(-?\d+(?:[\.,]\d+)*)'
        match = re.search(pattern2, answer_text, re.IGNORECASE)
        if match:
            answer = match.group(1).replace(',', '')
            try:
                if '.' in answer:
                    float_val = float(answer)
                    if float_val.is_integer():
                        return str(int(float_val))
                    else:
                        return answer
                else:
                    return str(int(answer))
            except ValueError:
                pass
        
        # 3. <<계산=결과>> 패턴에서 마지막 결과 추출
        pattern3 = r'<<[^>]*=(-?\d+(?:[\.,]\d+)*)>>'
        matches = re.findall(pattern3, answer_text)
        if matches:
            last_calc = matches[-1].replace(',', '')
            try:
                if '.' in last_calc:
                    float_val = float(last_calc)
                    if float_val.is_integer():
                        return str(int(float_val))
                    else:
                        return last_calc
                else:
                    return str(int(last_calc))
            except ValueError:
                pass
        
        # 4. 마지막 수단: 텍스트의 마지막 숫자
        all_numbers = re.findall(r'-?\d+(?:[\.,]\d+)*', answer_text)
        if all_numbers:
            last_number = all_numbers[-1].replace(',', '')
            try:
                if '.' in last_number:
                    float_val = float(last_number)
                    if float_val.is_integer():
                        return str(int(float_val))
                    else:
                        return last_number
                else:
                    return str(int(last_number))
            except ValueError:
                pass
        
        # 5. 기본값
        if idx is not None:
            print(f"⚠️ GSM8K item {idx}: Could not extract final answer from: '{answer_text[:100]}...'")
        
        return "0"
    
    def _get_default_answer(self):
        return "0"
    
    def _is_valid_item(self, item):
        """GSM8K 특화 검증"""
        base_valid = super()._is_valid_item(item)
        if not base_valid:
            return False
        
        # 추가 검증: 수학 문제 특성 확인
        metadata = item.get('metadata', {})
        problem = metadata.get('problem', '')
        target_text = item.get('target_text', '')
        
        # 문제가 수학 문제인지 확인 (숫자가 포함되어야 함)
        if not re.search(r'\d+', problem):
            return False
        
        # 답이 유효한 숫자인지 확인
        try:
            float(target_text.replace(',', ''))
        except ValueError:
            return False
            
        return True
    
    def verify_split_integrity(self):
        """데이터 분할 무결성 검증"""
        print(f"\n🔍 GSM8K {self.split} Split Verification")
        print(f"📊 Total samples: {len(self.data)}")
        
        # 답변 분포 및 추출 성공률 확인
        successful_extractions = 0
        answer_types = []
        answer_lengths = []
        
        for i in range(min(100, len(self.data))):
            try:
                item = self.__getitem__(i)
                target = item['target_text']
                
                # 추출 성공 여부 확인
                if target != "0":
                    successful_extractions += 1
                
                # 답변 타입 분석
                try:
                    val = float(target.replace(',', ''))
                    if val.is_integer():
                        answer_types.append('integer')
                    else:
                        answer_types.append('float')
                    answer_lengths.append(len(target))
                except ValueError:
                    answer_types.append('invalid')
                
            except Exception as e:
                print(f"❌ Error in item {i}: {e}")
        
        from collections import Counter
        type_dist = Counter(answer_types)
        
        success_rate = successful_extractions / min(100, len(self.data))
        avg_length = sum(answer_lengths) / max(len(answer_lengths), 1)
        
        print(f"🎯 Answer extraction success rate: {success_rate:.2%}")
        print(f"🔢 Answer types: {type_dist}")
        print(f"📏 Average answer length: {avg_length:.1f} characters")
        
        # 검증
        if success_rate < 0.8:
            print("🚨 WARNING: Low answer extraction success rate! Check _extract_final_answer method.")
        elif success_rate > 0.95:
            print("✅ Answer extraction working well.")
        else:
            print("⚠️ Moderate answer extraction success rate.")
        
        # 샘플 답변 표시
        print(f"\n📋 Sample answers:")
        for i in range(min(5, len(self.data))):
            item = self.__getitem__(i)
            metadata = item.get('metadata', {})
            full_solution = metadata.get('full_solution', '')
            target = item['target_text']
            
            print(f"  {i+1}. Target: '{target}' | Solution excerpt: '{full_solution[-50:]}'")
        
        return {
            'success_rate': success_rate,
            'type_distribution': type_dist,
            'average_length': avg_length
        }