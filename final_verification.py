# final_verification.py
import torch
import warnings
warnings.filterwarnings("ignore")

def test_imports():
    """기본 import 테스트"""
    print("🔍 Testing imports...")
    try:
        from transformers import T5Tokenizer
        from models.connection_transformer import ConnectionTransformer
        from models.baseline_transformer import BaselineTransformer
        from training.trainer import Trainer
        from configs.strategyqa_config import get_config
        print("✅ Imports OK")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Config 시스템 테스트"""
    print("\n🔍 Testing config...")
    try:
        from configs.strategyqa_config import get_config
        config = get_config("nano")
        assert config.d_model == 32
        assert config.dataset_name == "strategyqa"
        print("✅ Config OK")
        return True
    except Exception as e:
        print(f"❌ Config failed: {e}")
        return False

def test_model():
    """모델 생성 테스트"""
    print("\n🔍 Testing model...")
    try:
        from models.connection_transformer import ConnectionTransformer
        from transformers import T5Tokenizer
        
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = ConnectionTransformer(
            src_vocab_size=tokenizer.vocab_size,
            tgt_vocab_size=tokenizer.vocab_size,
            d_model=32, num_slots=8, bilinear_rank=2,
            max_reasoning_steps=1, max_seq_len=16,
            src_pad_token_id=0, tgt_pad_token_id=0,
            num_decoder_layers=2, num_heads=2
        )
        
        # Quick forward
        src = torch.randint(1, 1000, (1, 4))
        tgt = torch.randint(1, 1000, (1, 3))
        
        with torch.no_grad():
            out = model(src, tgt, torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool))
        
        print(f"✅ Model OK: output shape {out.shape}")
        return True
    except Exception as e:
        print(f"❌ Model failed: {e}")
        return False

def main():
    """간단한 검증"""
    print("🚀 Quick Verification")
    print("="*30)
    
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Model", test_model)
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    print("\n" + "="*30)
    all_passed = all(results.values())
    
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed.")
    
    return all_passed

if __name__ == "__main__":
    main()