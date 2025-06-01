# final_verification.py - 간단한 시스템 검증
import torch
import warnings
warnings.filterwarnings("ignore")

def test_imports():
    """필수 모듈 import 테스트"""
    print("🔍 Testing imports...")
    
    try:
        from transformers import T5Tokenizer
        from models.connection_transformer import ConnectionTransformer
        from models.baseline_transformer import BaselineTransformer
        from training.trainer import Trainer
        from configs.strategyqa_config import get_config
        print("✅ Imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_models():
    """모델 생성 테스트"""
    print("\n🔍 Testing models...")
    
    try:
        from transformers import T5Tokenizer
        from models.connection_transformer import ConnectionTransformer
        from models.baseline_transformer import BaselineTransformer
        
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        vocab_size = tokenizer.vocab_size
        
        # Connection model
        conn_model = ConnectionTransformer(
            src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
            d_model=32, num_slots=8, bilinear_rank=2,
            max_reasoning_steps=1, max_seq_len=16,
            src_pad_token_id=0, tgt_pad_token_id=0,
            num_decoder_layers=2, num_heads=2
        )
        
        # Baseline model
        baseline_model = BaselineTransformer(
            src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
            d_model=32, num_encoder_layers=2, num_decoder_layers=2,
            ffn_multiplier=2, max_seq_len=16,
            src_pad_token_id=0, tgt_pad_token_id=0
        )
        
        # Quick forward test
        src_ids = torch.randint(1, 1000, (1, 4))
        tgt_ids = torch.randint(1, 1000, (1, 3))
        src_mask = torch.ones(1, 4, dtype=torch.bool)
        tgt_mask = torch.ones(1, 3, dtype=torch.bool)
        
        with torch.no_grad():
            conn_out = conn_model(src_ids, tgt_ids, src_mask, tgt_mask)
            base_out = baseline_model(src_ids, tgt_ids, src_mask, tgt_mask)
        
        print(f"   Connection output: {conn_out.shape}")
        print(f"   Baseline output: {base_out.shape}")
        print("✅ Models working")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_configs():
    """설정 시스템 테스트"""
    print("\n🔍 Testing configs...")
    
    try:
        from configs.strategyqa_config import get_config
        
        config = get_config("nano")
        
        # Basic attributes
        assert hasattr(config, 'd_model')
        assert hasattr(config, 'num_slots')
        assert hasattr(config, 'learning_rate')
        
        print(f"   Config d_model: {config.d_model}")
        print("✅ Configs working")
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_training():
    """훈련 시스템 테스트"""
    print("\n🔍 Testing training...")
    
    try:
        from training.trainer import Trainer
        from models.connection_transformer import ConnectionTransformer
        from configs.strategyqa_config import get_config
        from transformers import T5Tokenizer
        
        config = get_config("nano")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        
        model = ConnectionTransformer(
            src_vocab_size=tokenizer.vocab_size,
            tgt_vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            num_slots=config.num_slots,
            bilinear_rank=config.bilinear_rank,
            max_reasoning_steps=config.max_reasoning_steps,
            max_seq_len=config.max_seq_len,
            src_pad_token_id=tokenizer.pad_token_id,
            tgt_pad_token_id=tokenizer.pad_token_id,
            num_decoder_layers=2,
            num_heads=2
        )
        
        trainer = Trainer(model, config, model_type="connection")
        trainer.set_tokenizer(tokenizer)
        
        print(f"   Trainer device: {trainer.device}")
        print("✅ Training setup working")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False

def main():
    """간단한 검증 실행"""
    print("🚀 Connection Transformer Quick Check")
    print("="*40)
    
    tests = [
        ("Imports", test_imports),
        ("Models", test_models),
        ("Configs", test_configs),
        ("Training", test_training)
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    print("\n" + "="*40)
    print("📋 Results")
    print("="*40)
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{name:10}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\nTry: python main.py --dataset strategyqa --model connection --model_size nano")
    else:
        print("\n⚠️ Some tests failed. Check your setup.")
    
    return all_passed

if __name__ == "__main__":
    main()