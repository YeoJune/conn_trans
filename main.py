# main.py
import torch
import argparse
import os
import warnings
from models.connection_transformer import ConnectionTransformer
from models.baseline_transformer import BaselineTransformer, calculate_matching_config_enc_dec
from training.trainer import Trainer
from dataset.tokenizer_utils import get_tokenizer_and_dataset
import configs.logiqa_config as logiqa_cfg
import configs.gsm8k_config as gsm8k_cfg
import configs.strategyqa_config as strategyqa_cfg
import configs.multinli_config as multinli_cfg

def setup_environment():
    """환경 설정"""
    warnings.filterwarnings("ignore", category=UserWarning)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"🚀 CUDA: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ Using CPU")
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

def main():
    parser = argparse.ArgumentParser(description="Connection Transformer Experiments")
    
    parser.add_argument("--dataset", 
                       choices=["logiqa", "gsm8k", "strategyqa", "multinli"], 
                       required=True)
    parser.add_argument("--model", 
                       choices=["connection", "baseline"], 
                       required=True)
    parser.add_argument("--model_size", 
                       choices=["nano", "micro", "tiny", "small", "base"], 
                       default="micro")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--no_save", action="store_true")
    
    args = parser.parse_args()
    
    # 환경 설정
    setup_environment()
    
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"📁 Output: {args.output_dir}")
    
    # Config 로드
    config_map = {
        "logiqa": logiqa_cfg,
        "gsm8k": gsm8k_cfg,
        "strategyqa": strategyqa_cfg,
        "multinli": multinli_cfg
    }
    
    config = config_map[args.dataset].get_config(model_size=args.model_size)
    config.output_dir = args.output_dir
    
    print(f"\n📋 {args.dataset} ({args.model_size}) - {args.model}")
    print(f"   d_model: {config.d_model}, batch_size: {config.batch_size}")
    print(f"   learning_rate: {config.learning_rate}, epochs: {config.num_epochs}")
    
    # 데이터 로드
    print(f"\n🔄 Loading data...")
    tokenizer, train_dataset, eval_dataset = get_tokenizer_and_dataset(args.dataset, config)
    
    # 모델 생성
    print(f"\n🏗️ Building {args.model} model...")
    
    if args.model == "connection":
        model = ConnectionTransformer(
            src_vocab_size=tokenizer.vocab_size,
            tgt_vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            num_slots=config.num_slots,
            bilinear_rank=config.bilinear_rank,
            max_reasoning_steps=config.max_reasoning_steps,
            convergence_threshold=config.convergence_threshold,
            max_seq_len=config.max_seq_len,
            dropout=getattr(config, 'dropout', 0.1),
            src_pad_token_id=tokenizer.pad_token_id,
            tgt_pad_token_id=tokenizer.pad_token_id,
            num_decoder_layers=getattr(config, 'num_decoder_layers', 3),
            num_heads=getattr(config, 'num_heads', 4)
        )
        
    else:  # baseline
        print("\n🔍 Calculating matching baseline...")
        baseline_config = calculate_matching_config_enc_dec(config)
        
        model = BaselineTransformer(
            src_vocab_size=tokenizer.vocab_size,
            tgt_vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            num_encoder_layers=baseline_config['num_encoder_layers'],
            num_decoder_layers=baseline_config['num_decoder_layers'],
            ffn_multiplier=baseline_config['ffn_multiplier'],
            max_seq_len=config.max_seq_len,
            dropout=getattr(config, 'dropout', 0.1),
            src_pad_token_id=tokenizer.pad_token_id,
            tgt_pad_token_id=tokenizer.pad_token_id
        )
        
        config.baseline_config = baseline_config
    
    # 훈련
    print(f"\n🚀 Training...")
    print("=" * 50)
    
    trainer = Trainer(model, config, model_type=args.model)
    trainer.set_tokenizer(tokenizer)
    
    best_accuracy = trainer.train(train_dataset, eval_dataset, resume_from=args.resume)
    
    print(f"\n✅ Completed!")
    print(f"   Dataset: {args.dataset}")
    print(f"   Model: {args.model} ({args.model_size})")
    print(f"   Best accuracy: {best_accuracy:.4f}")
    
    # Connection 분석
    if args.model == "connection":
        print(f"\n🔍 Connection analysis...")
        analysis = model.get_connection_analysis()
        print(f"   Max strength: {analysis['max_connection']:.4f}")
        print(f"   Mean strength: {analysis['mean_connection']:.4f}")
        print(f"   Sparsity: {analysis['sparsity_ratio']:.2%}")
        
        # 간단한 시각화
        if not args.no_save:
            try:
                from utils.visualization import visualize_connection_matrix
                visualize_connection_matrix(
                    model, 
                    save_path=os.path.join(args.output_dir, f"connections_{args.dataset}_{args.model_size}.png")
                )
                print(f"   📊 Visualization saved")
            except Exception as e:
                print(f"   ⚠️ Visualization error: {e}")

if __name__ == "__main__":
    main()