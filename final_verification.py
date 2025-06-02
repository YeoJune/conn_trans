# final_verification.py
import torch
import warnings
import sys
from pathlib import Path

warnings.filterwarnings("ignore")

class SystemVerifier:
    """Comprehensive system verification"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_total = 0
        self.errors = []
    
    def run_test(self, test_name, test_func):
        """Run a single test and track results"""
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
        """Test all essential imports"""
        try:
            # Core libraries
            import torch
            import transformers
            
            # Our modules
            from models.connection_transformer import ConnectionTransformer
            from models.baseline_transformer import BaselineTransformer
            from training.trainer import Trainer
            from dataset.tokenizer_utils import get_tokenizer_and_dataset
            from utils.metrics import calculate_accuracy
            from utils.visualization import plot_training_curves
            
            # Config modules
            import configs.strategyqa_config
            import configs.logiqa_config
            import configs.gsm8k_config
            import configs.multinli_config
            
            print("   All imports successful")
            return True
            
        except ImportError as e:
            print(f"   Import error: {e}")
            return False
    
    def test_config_system(self):
        """Test configuration system"""
        try:
            from configs.strategyqa_config import get_config
            
            # Test different sizes
            for size in ["x-small", "small", "base"]:
                config = get_config(size)
                
                # Verify required attributes
                required_attrs = [
                    'd_model', 'num_slots', 'bilinear_rank', 'max_reasoning_steps',
                    'learning_rate', 'batch_size', 'num_epochs', 'dataset_name'
                ]
                
                for attr in required_attrs:
                    if not hasattr(config, attr):
                        print(f"   Missing attribute: {attr}")
                        return False
            
            print("   Config system working correctly")
            return True
            
        except Exception as e:
            print(f"   Config error: {e}")
            return False
    
    def test_model_creation(self):
        """Test model creation and basic forward pass"""
        try:
            from models.connection_transformer import ConnectionTransformer
            from models.baseline_transformer import BaselineTransformer
            from transformers import T5Tokenizer
            
            # Load tokenizer
            tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
            vocab_size = tokenizer.vocab_size
            
            # Test Connection Transformer
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
            
            # Test Baseline Transformer
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
            
            # Test forward passes
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
                
                # Test with reasoning trace
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
        """Test dataset loading"""
        try:
            from dataset.tokenizer_utils import get_tokenizer_and_dataset
            from configs.strategyqa_config import get_config
            
            # Use smallest config for fast testing
            config = get_config("x-small")
            
            # Test dataset loading
            tokenizer, train_dataset, eval_dataset = get_tokenizer_and_dataset("strategyqa", config)
            
            # Verify datasets
            if len(train_dataset) == 0:
                print("   Empty train dataset")
                return False
            
            if len(eval_dataset) == 0:
                print("   Empty eval dataset")
                return False
            
            # Test data loading
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
        """Test training setup (without actual training)"""
        try:
            from training.trainer import Trainer
            from models.connection_transformer import ConnectionTransformer
            from configs.strategyqa_config import get_config
            from transformers import T5Tokenizer
            
            # Setup
            config = get_config("x-small")
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
            
            # Create trainer
            trainer = Trainer(model, config, model_type="connection")
            trainer.set_tokenizer(tokenizer)
            
            # Verify trainer setup
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
    
    def test_metrics_and_utils(self):
        """Test metrics and utility functions"""
        try:
            from utils.metrics import calculate_accuracy, extract_final_answer, exact_match_score
            from utils.visualization import plot_training_curves
            
            # Test metric functions
            predictions = ["Yes", "A", "42", "entailment"]
            targets = ["Yes", "A", "42", "entailment"]
            
            # Test accuracy calculation
            acc = calculate_accuracy(predictions, targets, "strategyqa")
            if acc != 1.0:
                print(f"   Wrong accuracy: {acc} (expected 1.0)")
                return False
            
            # Test answer extraction
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
            
            print("   Metrics and utils working correctly")
            return True
            
        except Exception as e:
            print(f"   Utils error: {e}")
            return False
    
    def test_end_to_end_compatibility(self):
        """Test end-to-end compatibility with minimal training step"""
        try:
            from training.trainer import Trainer
            from models.connection_transformer import ConnectionTransformer
            from dataset.tokenizer_utils import get_tokenizer_and_dataset
            from configs.strategyqa_config import get_config
            
            # Use x-small config for speed
            config = get_config("x-small")
            config.num_epochs = 1  # Just one epoch
            config.batch_size = 2  # Small batch
            
            # Load data
            tokenizer, train_dataset, eval_dataset = get_tokenizer_and_dataset("strategyqa", config)
            
            # Use tiny subset for speed
            from torch.utils.data import Subset
            train_subset = Subset(train_dataset, range(min(4, len(train_dataset))))
            eval_subset = Subset(eval_dataset, range(min(4, len(eval_dataset))))
            
            # Create model
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
            
            # Test trainer initialization
            trainer = Trainer(model, config, model_type="connection")
            trainer.set_tokenizer(tokenizer)
            
            # Test one training step (without full training)
            from torch.utils.data import DataLoader
            from training.data_collator import T5DataCollator
            
            data_collator = T5DataCollator(tokenizer, max_length=config.max_seq_len)
            train_loader = DataLoader(train_subset, batch_size=2, collate_fn=data_collator)
            
            # Setup optimizer
            trainer._setup_optimizer(train_loader)
            
            # Test one batch
            model.train()
            batch = next(iter(train_loader))
            tensors = trainer._extract_batch_tensors(batch)
            
            logits, reasoning_info = trainer._forward_pass(tensors, return_reasoning=True)
            loss = trainer._calculate_loss(logits, tensors['labels'])
            
            # Verify shapes and values
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"   Invalid loss: {loss}")
                return False
            
            if reasoning_info and 'actual_steps' in reasoning_info:
                steps = reasoning_info['actual_steps']
                if steps <= 0 or steps > config.max_reasoning_steps:
                    print(f"   Invalid reasoning steps: {steps}")
                    return False
            
            print("   End-to-end compatibility successful")
            return True
            
        except Exception as e:
            print(f"   E2E error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all verification tests"""
        print("🚀 Connection Transformer System Verification")
        print("=" * 60)
        
        tests = [
            ("Basic Imports", self.test_basic_imports),
            ("Config System", self.test_config_system),
            ("Model Creation", self.test_model_creation),
            ("Dataset Loading", self.test_dataset_loading),
            ("Training Setup", self.test_training_setup),
            ("Metrics & Utils", self.test_metrics_and_utils),
            ("End-to-End Compatibility", self.test_end_to_end_compatibility)
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Final report
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_total} passed")
        
        if self.tests_passed == self.tests_total:
            print("🎉 All tests PASSED! System is ready.")
            print("\n🚀 You can now run:")
            print("   python main.py --dataset strategyqa --model connection --model_size x-small")
            return True
        else:
            print(f"❌ {self.tests_total - self.tests_passed} tests FAILED:")
            for error in self.errors:
                print(f"   • {error}")
            print("\n🔧 Please fix the issues above before training.")
            return False
    
    def quick_test(self):
        """Quick essential tests only"""
        print("⚡ Quick System Check")
        print("-" * 30)
        
        essential_tests = [
            ("Imports", self.test_basic_imports),
            ("Config", self.test_config_system),
            ("Model", self.test_model_creation)
        ]
        
        for test_name, test_func in essential_tests:
            if not self.run_test(test_name, test_func):
                print(f"\n❌ Quick test failed at {test_name}")
                return False
        
        print(f"\n✅ Quick test passed! ({len(essential_tests)}/{len(essential_tests)})")
        return True

def main():
    """Main verification entry point"""
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