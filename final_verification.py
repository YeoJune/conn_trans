# final_verification.py - 통합 시스템 검증 스크립트
"""
통합 모델 사이즈 시스템의 검증 스크립트
"""
import torch
import warnings
warnings.filterwarnings("ignore")

def test_tokenizer_system():
    """토크나이저 시스템 테스트"""
    print("🔍 Testing tokenizer system...")
    
    try:
        from transformers import T5Tokenizer
        
        tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
        
        # 모든 데이터셋 task prefix 테스트
        test_inputs = [
            "reason: The sky is blue. question: What color is the sky?",  # LogiQA
            "solve: John has 5 apples. He gives 2 to Mary. How many does he have left?",  # GSM8K
            "strategy: Can a person fly without any tools?",  # StrategyQA
            "infer: premise: All birds can fly. hypothesis: Penguins can fly."  # MultiNLI
        ]
        
        for input_text in test_inputs:
            inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", 
                             truncation=True, max_length=128)
            decoded = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
            print(f"   ✅ '{input_text[:30]}...' -> {inputs.input_ids.shape}")
        
        print("✅ Tokenizer tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False

def test_unified_configs():
    """통합 설정 시스템 테스트"""
    print("\n🔍 Testing unified config system...")
    
    try:
        # 모든 데이터셋 설정 테스트
        datasets_to_test = [
            ("logiqa", "micro"),
            ("gsm8k", "micro"), 
            ("strategyqa", "nano"),
            ("multinli", "base")  # 큰 데이터셋
        ]
        
        for dataset_name, model_size in datasets_to_test:
            print(f"   Testing {dataset_name} with {model_size} model...")
            
            if dataset_name == "logiqa":
                from configs.logiqa_config import get_config
            elif dataset_name == "gsm8k":
                from configs.gsm8k_config import get_config
            elif dataset_name == "strategyqa":
                from configs.strategyqa_config import get_config
            elif dataset_name == "multinli":
                from configs.multinli_config import get_config
            
            config = get_config(model_size)
            
            # 필수 속성 체크
            required_attrs = [
                'dataset_name', 'd_model', 'num_slots', 'bilinear_rank',
                'max_reasoning_steps', 'learning_rate', 'batch_size', 
                'num_epochs', 'dropout', 'orthogonal_weight'
            ]
            
            for attr in required_attrs:
                if not hasattr(config, attr):
                    raise ValueError(f"Missing attribute: {attr}")
            
            # 파라미터 수 계산 테스트
            params = config.get_estimated_params()
            print(f"      ✅ {dataset_name}-{model_size}: {params['total']:,} params")
        
        print("✅ Config system tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """데이터셋 로딩 테스트"""
    print("\n🔍 Testing dataset loading...")
    
    try:
        from dataset.tokenizer_utils import get_tokenizer_and_dataset
        
        # 각 데이터셋 빠른 테스트 (작은 모델 사용)
        test_configs = [
            ("logiqa", "micro"),
            ("strategyqa", "nano"),
            # ("multinli", "tiny"),  # 시간이 오래 걸리므로 선택적
            # ("gsm8k", "micro")     # 선택적
        ]
        
        for dataset_name, model_size in test_configs:
            print(f"   Testing {dataset_name} dataset loading...")
            
            # Config 가져오기
            if dataset_name == "logiqa":
                from configs.logiqa_config import get_config
            elif dataset_name == "strategyqa":
                from configs.strategyqa_config import get_config
            
            config = get_config(model_size)
            
            try:
                tokenizer, train_dataset, eval_dataset = get_tokenizer_and_dataset(dataset_name, config)
                
                # 첫 번째 샘플 테스트
                sample = train_dataset[0]
                required_keys = ['input_ids', 'attention_mask', 'labels', 'target_text']
                
                for key in required_keys:
                    if key not in sample:
                        raise ValueError(f"Missing key: {key}")
                
                print(f"      ✅ {dataset_name}: {len(train_dataset)} train, {len(eval_dataset)} eval")
                print(f"         Sample shapes: input_ids={sample['input_ids'].shape}, labels={sample['labels'].shape}")
                
            except Exception as e:
                print(f"      ⚠️ {dataset_name} loading failed: {e}")
        
        print("✅ Dataset loading tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        return False

def test_model_architectures():
    """모델 아키텍처 테스트"""
    print("\n🔍 Testing model architectures...")
    
    try:
        from models.connection_transformer import ConnectionTransformer
        from models.baseline_transformer import BaselineTransformer
        
        # 다양한 모델 사이즈 테스트
        test_configs = [
            {"d_model": 32, "num_slots": 8, "bilinear_rank": 2},    # nano
            {"d_model": 64, "num_slots": 16, "bilinear_rank": 4},   # micro
            {"d_model": 128, "num_slots": 32, "bilinear_rank": 8},  # tiny
        ]
        
        vocab_size = 32128
        
        for i, config in enumerate(test_configs):
            size_names = ["nano", "micro", "tiny"]
            size_name = size_names[i]
            
            print(f"   Testing {size_name} models...")
            
            # Connection Transformer 테스트
            conn_model = ConnectionTransformer(
                vocab_size=vocab_size,
                d_model=config["d_model"],
                num_slots=config["num_slots"],
                bilinear_rank=config["bilinear_rank"],
                max_reasoning_steps=2,
                max_seq_len=64,
                pad_token_id=0
            )
            
            # Baseline Transformer 테스트
            baseline_model = BaselineTransformer(
                vocab_size=vocab_size,
                d_model=config["d_model"],
                num_layers=2,
                ffn_multiplier=4,
                max_seq_len=64,
                pad_token_id=0
            )
            
            # Forward pass 테스트
            batch_size = 2
            seq_len = 8
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))  # 작은 vocab으로 테스트
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
            
            with torch.no_grad():
                # Connection Transformer
                conn_output = conn_model(input_ids, attention_mask)
                print(f"      ✅ Connection {size_name}: {conn_output.shape}")
                
                # Baseline Transformer
                baseline_output = baseline_model(input_ids, attention_mask)
                print(f"      ✅ Baseline {size_name}: {baseline_output.shape}")
                
                # Reasoning trace 테스트
                conn_output_with_trace = conn_model(input_ids, attention_mask, return_reasoning_trace=True)
                logits, reasoning_info = conn_output_with_trace
                print(f"      ✅ Reasoning trace: {reasoning_info['actual_steps']} steps")
                
                # Orthogonal regularization 테스트
                if hasattr(conn_model, 'lightweight_orthogonal_regularization_loss'):
                    orth_loss = conn_model.lightweight_orthogonal_regularization_loss()
                    print(f"      ✅ Orthogonal loss: {orth_loss.item():.4f}")
        
        print("✅ Model architecture tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_integration():
    """훈련 통합 테스트"""
    print("\n🔍 Testing training integration...")
    
    try:
        from training.trainer import Trainer
        from models.connection_transformer import ConnectionTransformer
        from configs.strategyqa_config import get_config
        from transformers import T5Tokenizer
        
        # 가장 작은 설정으로 테스트
        config = get_config("nano")
        tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
        
        # 매우 작은 모델
        model = ConnectionTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            num_slots=8,
            bilinear_rank=2,
            max_reasoning_steps=1,
            max_seq_len=32,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Trainer 초기화
        trainer = Trainer(model, config, model_type="connection")
        trainer.set_tokenizer(tokenizer)
        
        # 손실 계산 테스트
        batch_size = 2
        seq_len = 8
        vocab_size = tokenizer.vocab_size
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(trainer.device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(trainer.device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(trainer.device)
        
        with torch.no_grad():
            logits, reasoning_info = model(input_ids, attention_mask, return_reasoning_trace=True)
            loss = trainer.calculate_loss(logits, labels, reasoning_info)
            
            print(f"   ✅ Loss calculation: {loss.item():.4f}")
            print(f"   ✅ Reasoning steps: {reasoning_info['actual_steps']}")
            
            # Orthogonal loss 테스트
            if hasattr(model, 'lightweight_orthogonal_regularization_loss'):
                orth_loss = model.lightweight_orthogonal_regularization_loss()
                print(f"   ✅ Orthogonal regularization: {orth_loss.item():.4f}")
        
        print("✅ Training integration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Training integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """전체 검증 실행"""
    print("🚀 Connection Transformer Unified System Verification")
    print("=" * 70)
    
    results = {}
    
    # 1. 토크나이저 시스템 검증
    results['tokenizer'] = test_tokenizer_system()
    
    # 2. 통합 설정 시스템 검증
    results['configs'] = test_unified_configs()
    
    # 3. 데이터셋 로딩 검증
    results['datasets'] = test_dataset_loading()
    
    # 4. 모델 아키텍처 검증
    results['models'] = test_model_architectures()
    
    # 5. 훈련 통합 검증
    results['training'] = test_training_integration()
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("📋 Unified System Verification Results")
    print("=" * 70)
    
    for component, success in results.items():
        status = "✅ 통과" if success else "❌ 실패"
        print(f"{component:15}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 통합 시스템 검증 완료! 모든 구성요소가 정상 작동합니다.")
        print("\n📋 권장 실험 순서:")
        print("   1. 작은 데이터셋 테스트:")
        print("      python main.py --dataset strategyqa --model connection --model_size nano")
        print("      python main.py --dataset logiqa --model connection --model_size micro")
        print("   2. 큰 데이터셋 실험:")
        print("      python main.py --dataset multinli --model connection --model_size base")
        print("   3. 베이스라인 비교:")
        print("      python main.py --dataset multinli --model baseline --model_size base")
    else:
        print("\n⚠️ 일부 검증 실패. 문제점을 확인해주세요.")
        
        failed_items = [k for k, v in results.items() if not v]
        print(f"\n실패한 항목: {', '.join(failed_items)}")
        
        print("\n🔧 문제 해결 가이드:")
        if 'tokenizer' in failed_items:
            print("   - pip install transformers>=4.21.0 sentencepiece>=0.1.97")
        if 'datasets' in failed_items:
            print("   - 인터넷 연결 확인 및 HuggingFace Hub 접근 가능 여부 확인")
        if 'configs' in failed_items:
            print("   - configs/ 디렉토리의 모든 파일이 올바르게 생성되었는지 확인")
        if 'models' in failed_items:
            print("   - models/ 디렉토리의 파일들과 PyTorch 버전 확인")
        if 'training' in failed_items:
            print("   - training/ 디렉토리와 CUDA 환경 확인")
    
    return all_passed

if __name__ == "__main__":
    main()