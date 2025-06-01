# final_verification.py
import torch
import warnings
warnings.filterwarnings("ignore")

def test_basic_imports():
    """기본 임포트 테스트"""
    print("🔍 Testing basic imports...")
    
    try:
        from transformers import T5Tokenizer
        from models.connection_transformer import ConnectionTransformer
        from models.baseline_transformer import BaselineTransformer
        from training.trainer import Trainer
        from configs.strategyqa_config import get_config
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_creation():
    """모델 생성 테스트"""
    print("\n🔍 Testing model creation...")
    
    try:
        from models.connection_transformer import ConnectionTransformer
        from models.baseline_transformer import BaselineTransformer
        
        # 작은 모델로 테스트
        vocab_size = 1000
        
        # Connection Transformer
        conn_model = ConnectionTransformer(
            vocab_size=vocab_size,
            d_model=32,
            num_slots=8,
            bilinear_rank=2,
            max_reasoning_steps=1,
            max_seq_len=16,
            pad_token_id=0
        )
        
        # Baseline Transformer
        baseline_model = BaselineTransformer(
            vocab_size=vocab_size,
            d_model=32,
            num_layers=2,
            ffn_multiplier=2,
            max_seq_len=16,
            pad_token_id=0
        )
        
        # Forward pass 테스트
        input_ids = torch.randint(0, vocab_size, (2, 8))
        attention_mask = torch.ones(2, 8, dtype=torch.bool)
        
        with torch.no_grad():
            conn_output = conn_model(input_ids, attention_mask)
            baseline_output = baseline_model(input_ids, attention_mask)
            
            print(f"   Connection output: {conn_output.shape}")
            print(f"   Baseline output: {baseline_output.shape}")
        
        print("✅ Model creation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_config_system():
    """설정 시스템 테스트"""
    print("\n🔍 Testing config system...")
    
    try:
        from configs.strategyqa_config import get_config
        from configs.logiqa_config import get_config as logiqa_get_config
        
        # 다양한 크기 테스트
        config_nano = get_config("nano")
        config_micro = logiqa_get_config("micro")
        
        # 필수 속성 확인
        required = ['d_model', 'num_slots', 'bilinear_rank', 'learning_rate']
        for attr in required:
            assert hasattr(config_nano, attr), f"Missing {attr}"
            assert hasattr(config_micro, attr), f"Missing {attr}"
        
        print(f"   Nano config: d_model={config_nano.d_model}")
        print(f"   Micro config: d_model={config_micro.d_model}")
        print("✅ Config system tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_training_setup():
    """훈련 설정 테스트"""
    print("\n🔍 Testing training setup...")
    
    try:
        from training.trainer import Trainer
        from models.connection_transformer import ConnectionTransformer
        from configs.strategyqa_config import get_config
        from transformers import T5Tokenizer
        
        config = get_config("nano")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        
        model = ConnectionTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            num_slots=config.num_slots,
            bilinear_rank=config.bilinear_rank,
            max_reasoning_steps=config.max_reasoning_steps,
            max_seq_len=config.max_seq_len,
            pad_token_id=tokenizer.pad_token_id
        )
        
        trainer = Trainer(model, config, model_type="connection")
        trainer.set_tokenizer(tokenizer)
        
        print(f"   Trainer device: {trainer.device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("✅ Training setup tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Training setup test failed: {e}")
        return False

def main():
    """전체 검증 실행"""
    print("🚀 Connection Transformer Quick Verification")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Model Creation", test_model_creation),
        ("Config System", test_config_system),
        ("Training Setup", test_training_setup)
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    print("\n" + "=" * 50)
    print("📋 Verification Results")
    print("=" * 50)
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:20}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Ready to run experiments.")
        print("\n📋 Quick start:")
        print("   python main.py --dataset strategyqa --model connection --model_size nano")
    else:
        print("\n⚠️ Some tests failed. Check your environment.")
    
    return all_passed

if __name__ == "__main__":
    main()