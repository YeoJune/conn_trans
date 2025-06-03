# final_verification.py
"""
시스템 검증 전용 스크립트 - 훈련 전 환경과 코드 검증
"""
import torch
import warnings
import sys
from pathlib import Path

warnings.filterwarnings("ignore")

class SystemVerifier:
    """시스템 환경 및 코드 검증"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_total = 0
        self.errors = []
    
    def run_test(self, test_name, test_func):
        """단일 테스트 실행"""
        self.tests_total += 1
        print(f"\n🔍 Testing {test_name}...")
        
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name} PASSED")
                self.tests_passed += 1
                return True
            else:
                print(f"❌ {test_name} FAILED")
                self.errors.append(f"{test_name}: Test returned False")
                return False
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            self.errors.append(f"{test_name}: {str(e)}")
            return False
    
    def test_basic_imports(self):
        """핵심 라이브러리 및 모듈 import 테스트"""
        try:
            # 외부 라이브러리
            import torch
            import transformers
            
            # 모델 모듈
            from models.connection_transformer import ConnectionTransformer
            from models.baseline_transformer import BaselineTransformer
            
            # 훈련 모듈
            from training.trainer import Trainer
            from training.data_collator import T5DataCollator
            
            # 데이터셋 모듈
            from dataset.tokenizer_utils import get_tokenizer_and_dataset
            
            # 유틸리티 모듈
            from utils.metrics import calculate_accuracy
            from utils.result_manager import ResultManager
            
            # 설정 모듈
            import configs.strategyqa_config
            import configs.logiqa_config
            import configs.gsm8k_config
            import configs.multinli_config
            import configs.eli5_config
            import configs.commongen_config
            
            print("   All critical imports successful")
            return True
            
        except ImportError as e:
            print(f"   Import error: {e}")
            return False
    
    def test_config_system(self):
        """설정 시스템 테스트"""
        try:
            from configs.strategyqa_config import get_config
            
            # 다양한 모델 크기 테스트
            for size in ["micro", "small", "base"]:
                config = get_config(size)
                
                # 필수 속성 확인
                required_attrs = [
                    'd_model', 'num_slots', 'bilinear_rank', 'max_reasoning_steps',
                    'learning_rate', 'batch_size', 'num_epochs'
                ]
                
                for attr in required_attrs:
                    if not hasattr(config, attr):
                        print(f"   Missing attribute: {attr}")
                        return False
                
                # 값 검증
                if config.d_model <= 0 or config.batch_size <= 0:
                    print(f"   Invalid config values for {size}")
                    return False
            
            print("   Config system working correctly")
            return True
            
        except Exception as e:
            print(f"   Config error: {e}")
            return False
    
    def test_model_creation(self):
        """모델 생성 및 기본 순전파 테스트"""
        try:
            from models.connection_transformer import ConnectionTransformer
            from models.baseline_transformer import BaselineTransformer
            from transformers import T5Tokenizer
            
            # 토크나이저 로드
            tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
            vocab_size = tokenizer.vocab_size
            
            # Connection Transformer 테스트
            conn_model = ConnectionTransformer(
                src_vocab_size=vocab_size,
                tgt_vocab_size=vocab_size,
                d_model=64,
                num_slots=16,
                bilinear_rank=4,
                max_reasoning_steps=2,
                max_seq_len=32,
                src_pad_token_id=tokenizer.pad_token_id,
                tgt_pad_token_id=tokenizer.pad_token_id,
                num_decoder_layers=3,
                num_heads=4
            )
            
            # Baseline Transformer 테스트
            base_model = BaselineTransformer(
                src_vocab_size=vocab_size,
                tgt_vocab_size=vocab_size,
                d_model=64,
                num_encoder_layers=3,
                num_decoder_layers=3,
                num_heads=4,
                max_seq_len=32,
                src_pad_token_id=tokenizer.pad_token_id,
                tgt_pad_token_id=tokenizer.pad_token_id
            )
            
            # 순전파 테스트
            batch_size, src_len, tgt_len = 2, 8, 6
            src_ids = torch.randint(1, 1000, (batch_size, src_len))
            tgt_ids = torch.randint(1, 1000, (batch_size, tgt_len))
            src_mask = torch.ones(batch_size, src_len, dtype=torch.bool)
            tgt_mask = torch.ones(batch_size, tgt_len, dtype=torch.bool)
            
            with torch.no_grad():
                # Connection Transformer
                conn_output = conn_model(src_ids, tgt_ids, src_mask, tgt_mask)
                expected_shape = (batch_size, tgt_len, vocab_size)
                if conn_output.shape != expected_shape:
                    print(f"   Wrong Connection output shape: {conn_output.shape} vs {expected_shape}")
                    return False
                
                # Reasoning trace 테스트
                conn_output, reasoning_info = conn_model(src_ids, tgt_ids, src_mask, tgt_mask, return_reasoning_trace=True)
                if 'actual_steps' not in reasoning_info:
                    print("   Missing reasoning info")
                    return False
                
                # Baseline Transformer
                base_output = base_model(src_ids, tgt_ids, src_mask, tgt_mask)
                if base_output.shape != expected_shape:
                    print(f"   Wrong Baseline output shape: {base_output.shape} vs {expected_shape}")
                    return False
            
            print("   Model creation and forward pass successful")
            return True
            
        except Exception as e:
            print(f"   Model error: {e}")
            return False
    
    def test_dataset_loading(self):
        """데이터셋 로딩 테스트"""
        try:
            from dataset.tokenizer_utils import get_tokenizer_and_dataset
            from configs.strategyqa_config import get_config
            
            # 최소 설정으로 빠른 테스트
            config = get_config("micro")
            
            # 데이터셋 로딩 테스트
            tokenizer, train_dataset, eval_dataset = get_tokenizer_and_dataset("strategyqa", config)
            
            # 데이터셋 검증
            if len(train_dataset) == 0:
                print("   Empty train dataset")
                return False
            
            if len(eval_dataset) == 0:
                print("   Empty eval dataset")
                return False
            
            # 샘플 데이터 검증
            sample = train_dataset[0]
            required_keys = ['input_ids', 'attention_mask', 'decoder_input_ids', 'labels']
            for key in required_keys:
                if key not in sample:
                    print(f"   Missing key in sample: {key}")
                    return False
                if not torch.is_tensor(sample[key]):
                    print(f"   {key} is not a tensor")
                    return False
            
            print(f"   Dataset loading successful: {len(train_dataset)} train, {len(eval_dataset)} eval")
            return True
            
        except Exception as e:
            print(f"   Dataset error: {e}")
            return False
    
    def test_training_setup(self):
        """훈련 환경 설정 테스트"""
        try:
            from training.trainer import Trainer
            from models.connection_transformer import ConnectionTransformer
            from configs.strategyqa_config import get_config
            from transformers import T5Tokenizer
            
            # 설정
            config = get_config("micro")
            tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
            config.vocab_size = tokenizer.vocab_size
            config.pad_token_id = tokenizer.pad_token_id
            
            model = ConnectionTransformer(
                src_vocab_size=config.vocab_size,
                tgt_vocab_size=config.vocab_size,
                d_model=config.d_model,
                num_slots=config.num_slots,
                bilinear_rank=config.bilinear_rank,
                max_reasoning_steps=config.max_reasoning_steps,
                max_seq_len=config.max_seq_len,
                src_pad_token_id=config.pad_token_id,
                tgt_pad_token_id=config.pad_token_id,
                num_decoder_layers=config.num_decoder_layers,
                num_heads=config.num_heads
            )
            
            # 트레이너 생성
            trainer = Trainer(model, config, model_type="connection")
            trainer.set_tokenizer(tokenizer)
            
            # 트레이너 검증
            if trainer.device.type not in ['cuda', 'cpu']:
                print(f"   Invalid device: {trainer.device}")
                return False
            
            if trainer.model_type != "connection":
                print(f"   Wrong model type: {trainer.model_type}")
                return False
            
            print("   Training setup successful")
            return True
            
        except Exception as e:
            print(f"   Training setup error: {e}")
            return False
    
    def test_metrics_system(self):
        """메트릭 시스템 테스트"""
        try:
            from utils.metrics import calculate_accuracy, extract_final_answer, exact_match_score
            
            # 메트릭 함수 테스트
            predictions = ["Yes", "A", "42", "entailment"]
            targets = ["Yes", "A", "42", "entailment"]
            
            # 정확도 계산 테스트
            acc = calculate_accuracy(predictions, targets, "strategyqa")
            if acc != 1.0:
                print(f"   Wrong accuracy: {acc} (expected 1.0)")
                return False
            
            # 답변 추출 테스트
            test_cases = [
                ("Yes, this is correct", "strategyqa", "Yes"),
                ("The answer is A", "logiqa", "A"),
                ("42 is the answer", "gsm8k", "42"),
                ("entailment is correct", "multinli", "entailment")
            ]
            
            for text, dataset_type, expected in test_cases:
                result = extract_final_answer(text, dataset_type)
                if result != expected:
                    print(f"   Wrong extraction: '{result}' vs '{expected}' for {dataset_type}")
                    return False
            
            print("   Metrics system working correctly")
            return True
            
        except Exception as e:
            print(f"   Metrics error: {e}")
            return False
    
    def run_all_tests(self):
        """모든 검증 테스트 실행"""
        print("🚀 Connection Transformer System Verification")
        print("=" * 60)
        
        tests = [
            ("Basic Imports", self.test_basic_imports),
            ("Config System", self.test_config_system),
            ("Model Creation", self.test_model_creation),
            ("Dataset Loading", self.test_dataset_loading),
            ("Training Setup", self.test_training_setup),
            ("Metrics System", self.test_metrics_system)
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # 최종 리포트
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_total} passed")
        
        if self.tests_passed == self.tests_total:
            print("🎉 All tests PASSED! System is ready.")
            print("\n🚀 You can now run:")
            print("   python main.py --dataset strategyqa --model connection --model_size micro")
            return True
        else:
            print(f"❌ {self.tests_total - self.tests_passed} tests FAILED:")
            for error in self.errors:
                print(f"   • {error}")
            print("\n🔧 Please fix the issues above before training.")
            return False
    
    def quick_test(self):
        """필수 테스트만 빠르게 실행"""
        print("⚡ Quick System Check")
        print("-" * 30)
        
        essential_tests = [
            ("Imports", self.test_basic_imports),
            ("Config", self.test_config_system),
            ("Models", self.test_model_creation)
        ]
        
        for test_name, test_func in essential_tests:
            if not self.run_test(test_name, test_func):
                print(f"\n❌ Quick test failed at {test_name}")
                return False
        
        print(f"\n✅ Quick test passed! ({len(essential_tests)}/{len(essential_tests)})")
        return True

def main():
    """검증 스크립트 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Connection Transformer system")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick tests only")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed output")
    
    args = parser.parse_args()
    
    if not args.verbose:
        warnings.filterwarnings("ignore")
    
    verifier = SystemVerifier()
    
    if args.quick:
        success = verifier.quick_test()
    else:
        success = verifier.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())