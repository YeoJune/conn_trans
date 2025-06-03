# main.py
import torch
import argparse
import os
import sys
from pathlib import Path

def get_config(dataset_name, model_size):
    """통합 설정 로딩"""
    config_map = {
        "strategyqa": "configs.strategyqa_config",
        "logiqa": "configs.logiqa_config", 
        "gsm8k": "configs.gsm8k_config",
        "multinli": "configs.multinli_config",
        "eli5": "configs.eli5_config",
        "commongen": "configs.commongen_config"
    }
    
    if dataset_name not in config_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    try:
        module = __import__(config_map[dataset_name], fromlist=['get_config'])
        config = module.get_config(model_size)
        
        # 필수 필드 추가
        config.dataset_name = dataset_name
        config.model_size = model_size

        #config.auto_balance() # 자동 균형 조정
        
        return config
    except ImportError as e:
        print(f"❌ Failed to import config for {dataset_name}: {e}")
        sys.exit(1)

def create_model(model_type, config):
    """통합 모델 생성"""
    if model_type == "connection":
        from models.connection_transformer import ConnectionTransformer
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
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
    elif model_type == "baseline":
        from models.baseline_transformer import BaselineTransformer, calculate_matching_config_enc_dec
        
        baseline_config = calculate_matching_config_enc_dec(config)
        
        model = BaselineTransformer(
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
    
    # Pre-trained weights 로딩
    model.load_pretrained_weights(config.tokenizer_name)
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Connection Transformer")
    parser.add_argument("--dataset", 
                       choices=["strategyqa", "logiqa", "gsm8k", "multinli", "eli5", "commongen"], 
                       required=True,
                       help="Dataset to use")
    parser.add_argument("--model", 
                       choices=["connection", "baseline"], 
                       required=True,
                       help="Model type to train")
    parser.add_argument("--model_size", 
                       choices=["micro", "small", "base", "large"], 
                       default="micro",
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
    
    # 출력 디렉토리 설정
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Connection Transformer Training")
    print(f"   Dataset: {args.dataset}")
    print(f"   Model: {args.model}")
    print(f"   Size: {args.model_size}")
    print(f"   Output: {output_dir}")
    print("-" * 50)
    
    try:
        # 설정 로드
        print("📋 Loading configuration...")
        config = get_config(args.dataset, args.model_size)
        config.output_dir = str(output_dir)
        
        print(f"✅ Config loaded:")
        print(f"   d_model={config.d_model}")
        print(f"   batch_size={config.batch_size}")
        print(f"   learning_rate={config.learning_rate}")
        print(f"   num_epochs={config.num_epochs}")
        
        if hasattr(config, 'num_slots'):
            print(f"   num_slots={config.num_slots}")
            print(f"   bilinear_rank={config.bilinear_rank}")
        
        # 데이터 로드
        print("\n📦 Loading dataset...")
        from dataset.tokenizer_utils import get_tokenizer_and_dataset
        tokenizer, train_dataset, eval_dataset = get_tokenizer_and_dataset(args.dataset, config)
        
        print(f"✅ Dataset loaded:")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Eval: {len(eval_dataset)} samples")
        print(f"   Vocab size: {config.vocab_size:,}")
        
        # 모델 생성
        print(f"\n🏗️ Creating {args.model} model...")
        model = create_model(args.model, config)
        
        # 파라미터 카운트
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Model created: {total_params:,} parameters")
        
        if args.dry_run:
            print("\n🔍 Dry run completed successfully!")
            return 0
        
        # 트레이너 설정
        print(f"\n🎯 Setting up trainer...")
        from training.trainer import Trainer
        trainer = Trainer(model, config, model_type=args.model)
        trainer.set_tokenizer(tokenizer)
        
        # 훈련 시작
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