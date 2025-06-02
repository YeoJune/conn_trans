# main.py
import torch
import argparse
import os
import sys
from pathlib import Path

def get_config(dataset_name, model_size):
    """Unified config loading"""
    config_map = {
        "strategyqa": "configs.strategyqa_config",
        "logiqa": "configs.logiqa_config", 
        "gsm8k": "configs.gsm8k_config",
        "multinli": "configs.multinli_config"
    }
    
    if dataset_name not in config_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    try:
        module = __import__(config_map[dataset_name], fromlist=['get_config'])
        return module.get_config(model_size)
    except ImportError as e:
        print(f"❌ Failed to import config for {dataset_name}: {e}")
        sys.exit(1)

def create_model(model_type, config):
    """Unified model creation"""
    if model_type == "connection":
        from models.connection_transformer import ConnectionTransformer
        return ConnectionTransformer(
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
            num_heads=config.num_heads,
            dropout=config.dropout
        )
    
    elif model_type == "baseline":
        from models.baseline_transformer import BaselineTransformer, calculate_matching_config_enc_dec
        
        # Calculate matching configuration
        baseline_config = calculate_matching_config_enc_dec(config)
        
        return BaselineTransformer(
            src_vocab_size=config.vocab_size,
            tgt_vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_encoder_layers=baseline_config['num_encoder_layers'],
            num_decoder_layers=baseline_config['num_decoder_layers'],
            num_heads=config.num_heads,
            ffn_multiplier=baseline_config['ffn_multiplier'],
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            src_pad_token_id=config.pad_token_id,
            tgt_pad_token_id=config.pad_token_id
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    parser = argparse.ArgumentParser(description="Train Connection Transformer")
    parser.add_argument("--dataset", 
                       choices=["strategyqa", "logiqa", "gsm8k", "multinli"], 
                       required=True,
                       help="Dataset to use")
    parser.add_argument("--model", 
                       choices=["connection", "baseline"], 
                       required=True,
                       help="Model type to train")
    parser.add_argument("--model_size", 
                       choices=["x-small", "small", "base", "large"], 
                       default="small",
                       help="Model size configuration")
    parser.add_argument("--output_dir", 
                       type=str, 
                       default="./outputs",
                       help="Output directory for results")
    parser.add_argument("--resume_from", 
                       type=str, 
                       default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--dry_run", 
                       action="store_true",
                       help="Just verify setup without training")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Connection Transformer Experiment")
    print(f"   Dataset: {args.dataset}")
    print(f"   Model: {args.model}")
    print(f"   Size: {args.model_size}")
    print(f"   Output: {output_dir}")
    print("-" * 50)
    
    try:
        # Load configuration
        print("📋 Loading configuration...")
        config = get_config(args.dataset, args.model_size)
        config.update(output_dir=str(output_dir))
        
        print(f"✅ Config loaded:")
        print(f"   d_model={config.d_model}")
        print(f"   batch_size={config.batch_size}")
        print(f"   learning_rate={config.learning_rate}")
        print(f"   num_epochs={config.num_epochs}")
        
        if hasattr(config, 'num_slots'):
            print(f"   num_slots={config.num_slots}")
            print(f"   bilinear_rank={config.bilinear_rank}")
        
        # Load data
        print("\n📦 Loading dataset...")
        from dataset.tokenizer_utils import get_tokenizer_and_dataset
        tokenizer, train_dataset, eval_dataset = get_tokenizer_and_dataset(args.dataset, config)
        
        print(f"✅ Dataset loaded:")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Eval: {len(eval_dataset)} samples")
        print(f"   Vocab size: {config.vocab_size:,}")
        
        # Create model
        print(f"\n🏗️ Creating {args.model} model...")
        model = create_model(args.model, config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Model created: {total_params:,} parameters")
        
        if args.dry_run:
            print("\n🔍 Dry run completed successfully!")
            return 0
        
        # Setup trainer
        print(f"\n🎯 Setting up trainer...")
        from training.trainer import Trainer
        trainer = Trainer(model, config, model_type=args.model)
        trainer.set_tokenizer(tokenizer)
        
        # Train
        print(f"\n🚀 Starting training...")
        best_accuracy = trainer.train(train_dataset, eval_dataset)
        
        print(f"\n✅ Training completed!")
        print(f"   Best accuracy: {best_accuracy:.4f}")
        print(f"   Results saved in: {output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
