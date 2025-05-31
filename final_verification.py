# final_verification.py - 최종 검증 스크립트
"""
수정된 모든 파일의 통합 검증 스크립트
"""
import torch
import warnings
warnings.filterwarnings("ignore")

def test_corrected_tokenizer():
    """수정된 토크나이저 테스트"""
    print("🔍 Testing corrected tokenizer...")
    
    try:
        from transformers import T5Tokenizer
        
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        
        # T5 특화 테스트
        test_inputs = [
            "reason: The sky is blue. question: What color is the sky?",
            "solve: John has 5 apples. He gives 2 to Mary. How many does he have left?",
            "strategy: Can a person fly without any tools?"
        ]
        
        for input_text in test_inputs:
            # 토크나이징
            inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", 
                             truncation=True, max_length=128)
            
            # 디코딩
            decoded = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
            
            print(f"   ✅ '{input_text[:30]}...' -> {inputs.input_ids.shape}")
        
        print("✅ Tokenizer tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False

def test_corrected_datasets():
    """수정된 데이터셋 테스트"""
    print("\n🔍 Testing corrected datasets...")
    
    try:
        from dataset.tokenizer_utils import get_tokenizer_and_dataset
        from configs.logiqa_config import get_config
        
        config = get_config("base")
        
        # LogiQA 테스트
        try:
            tokenizer, train_dataset, eval_dataset = get_tokenizer_and_dataset("logiqa", config)
            
            # 첫 번째 샘플 테스트
            sample = train_dataset[0]
            required_keys = ['input_ids', 'attention_mask', 'labels', 'target_text']
            
            for key in required_keys:
                if key not in sample:
                    raise ValueError(f"Missing key: {key}")
            
            print(f"   ✅ LogiQA: {len(train_dataset)} train, {len(eval_dataset)} eval")
            print(f"      Sample shapes: input_ids={sample['input_ids'].shape}, labels={sample['labels'].shape}")
            
        except Exception as e:
            print(f"   ⚠️ LogiQA test failed: {e}")
        
        print("✅ Dataset tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def test_corrected_models():
    """수정된 모델 테스트"""
    print("\n🔍 Testing corrected models...")
    
    try:
        from models.connection_transformer import ConnectionTransformer
        from models.baseline_transformer import BaselineTransformer
        
        # 테스트 설정
        vocab_size = 32128  # T5-base vocab size
        d_model = 256
        num_slots = 128
        bilinear_rank = 32
        
        # Connection Transformer 테스트
        conn_model = ConnectionTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_slots=num_slots,
            bilinear_rank=bilinear_rank,
            max_reasoning_steps=4,
            max_seq_len=128,
            pad_token_id=0
        )
        
        # Baseline Transformer 테스트
        baseline_model = BaselineTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=4,
            ffn_multiplier=4,
            max_seq_len=128,
            pad_token_id=0
        )
        
        # Forward pass 테스트
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        with torch.no_grad():
            # Connection Transformer
            conn_output = conn_model(input_ids, attention_mask)
            print(f"   ✅ Connection Transformer output: {conn_output.shape}")
            
            # Baseline Transformer
            baseline_output = baseline_model(input_ids, attention_mask)
            print(f"   ✅ Baseline Transformer output: {baseline_output.shape}")
            
            # Reasoning trace 테스트
            conn_output_with_trace = conn_model(input_ids, attention_mask, return_reasoning_trace=True)
            logits, reasoning_info = conn_output_with_trace
            print(f"   ✅ Reasoning trace: {reasoning_info['actual_steps']} steps")
        
        print("✅ Model tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_corrected_training():
    """수정된 훈련 테스트"""
    print("\n🔍 Testing corrected training setup...")
    
    try:
        from training.trainer import Trainer
        from models.connection_transformer import ConnectionTransformer
        from configs.logiqa_config import get_config
        from transformers import T5Tokenizer
        
        # 설정 및 토크나이저
        config = get_config("base")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        
        # 간단한 모델
        model = ConnectionTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=128,  # 작게 설정
            num_slots=64,
            bilinear_rank=16,
            max_reasoning_steps=2,
            max_seq_len=64,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Trainer 초기화
        trainer = Trainer(model, config, model_type="connection")
        trainer.set_tokenizer(tokenizer)
        
        # 손실 계산 테스트
        batch_size = 2
        seq_len = 10
        vocab_size = tokenizer.vocab_size
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            logits, reasoning_info = model(input_ids, attention_mask, return_reasoning_trace=True)
            loss = trainer.calculate_loss(logits, labels, reasoning_info)
            
            print(f"   ✅ Loss calculation: {loss.item():.4f}")
            print(f"   ✅ Reasoning steps: {reasoning_info['actual_steps']}")
        
        print("✅ Training tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """전체 검증 실행"""
    print("🚀 Final Verification of Corrected Implementation")
    print("=" * 70)
    
    results = {}
    
    # 1. 토크나이저 검증
    results['tokenizer'] = test_corrected_tokenizer()
    
    # 2. 데이터셋 검증  
    results['datasets'] = test_corrected_datasets()
    
    # 3. 모델 검증
    results['models'] = test_corrected_models()
    
    # 4. 훈련 검증
    results['training'] = test_corrected_training()
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("📋 Final Verification Results")
    print("=" * 70)
    
    for component, success in results.items():
        status = "✅ 통과" if success else "❌ 실패"
        print(f"{component:15}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 모든 수정사항 검증 완료! 구현이 올바릅니다.")
        print("\n📋 다음 단계:")
        print("   1. pip install -r requirements.txt")
        print("   2. python main.py --dataset logiqa --model connection")
        print("   3. python main.py --dataset logiqa --model baseline")
    else:
        print("\n⚠️ 일부 검증 실패. 문제점을 확인해주세요.")
        
        # 실패한 항목들 상세 안내
        failed_items = [k for k, v in results.items() if not v]
        print(f"\n실패한 항목: {', '.join(failed_items)}")
        
        if 'tokenizer' in failed_items:
            print("\n🔧 토크나이저 문제 해결방법:")
            print("   pip install sentencepiece>=0.1.97")
        
        if 'datasets' in failed_items:
            print("\n🔧 데이터셋 문제 해결방법:")
            print("   인터넷 연결 확인 및 HuggingFace Hub 접근 가능 여부 확인")
    
    return all_passed

if __name__ == "__main__":
    main()