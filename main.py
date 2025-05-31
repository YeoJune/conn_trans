# main.py
import torch
import argparse
import os
import warnings
from models.connection_transformer import ConnectionTransformer
from models.baseline_transformer import BaselineTransformer, calculate_matching_config
from training.trainer import Trainer
from data.tokenizer_utils import get_tokenizer_and_dataset
import configs.logiqa_config as logiqa_cfg
import configs.gsm8k_config as gsm8k_cfg
import configs.strategyqa_config as strategyqa_cfg

def setup_environment():
    """환경 설정"""
    # 경고 무시
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # GPU 설정
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    # 시드 설정
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

def main():
    parser = argparse.ArgumentParser(description="Connection Transformer Experiments")
    parser.add_argument("--dataset", choices=["logiqa", "gsm8k", "strategyqa"], required=True,
                       help="Dataset to use for training")
    parser.add_argument("--model", choices=["connection", "baseline"], required=True,
                       help="Model type to train")
    parser.add_argument("--model_size", choices=["base", "large"], default="base",
                       help="Model size configuration")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Directory to save outputs")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save checkpoints and results")
    
    args = parser.parse_args()
    
    # 환경 설정
    setup_environment()
    
    # 출력 디렉토리 생성
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"📁 Output directory: {args.output_dir}")
    
    # Config 선택
    config_map = {
        "logiqa": logiqa_cfg,
        "gsm8k": gsm8k_cfg,
        "strategyqa": strategyqa_cfg
    }
    
    if args.dataset not in config_map:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    config = config_map[args.dataset].get_config(model_size=args.model_size)
    print(f"\n📋 Configuration for {args.dataset} ({args.model_size}):")
    print(f"   Model: {args.model}")
    print(f"   d_model: {config.d_model}")
    print(f"   num_slots: {config.num_slots}")
    print(f"   bilinear_rank: {config.bilinear_rank}")
    print(f"   max_reasoning_steps: {config.max_reasoning_steps}")
    print(f"   batch_size: {config.batch_size}")
    print(f"   learning_rate: {config.learning_rate}")
    
    # 토크나이저 & 데이터셋 생성
    print(f"\n🔄 Loading data and tokenizer...")
    tokenizer, train_dataset, eval_dataset = get_tokenizer_and_dataset(args.dataset, config)
    
    # Config에 vocab_size 설정 (baseline config 계산용)
    config.vocab_size = tokenizer.vocab_size
    
    # 모델 생성
    print(f"\n🏗️ Building {args.model} model...")
    
    if args.model == "connection":
        model = ConnectionTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            num_slots=config.num_slots,
            bilinear_rank=config.bilinear_rank,
            max_reasoning_steps=config.max_reasoning_steps,
            convergence_threshold=config.convergence_threshold,
            max_seq_len=config.max_seq_len,
            dropout=getattr(config, 'dropout', 0.1)
        )
        
    elif args.model == "baseline":
        # Calculate matching configuration for fair comparison
        print("\n🔍 Calculating matching baseline configuration...")
        baseline_config = calculate_matching_config(config)
        
        model = BaselineTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            num_layers=baseline_config['num_layers'],
            ffn_multiplier=baseline_config['ffn_multiplier'],
            max_seq_len=config.max_seq_len,
            dropout=getattr(config, 'dropout', 0.1)
        )
        
        # 설정에 baseline 정보 추가
        config.baseline_config = baseline_config
    
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # 훈련
    print(f"\n🚀 Starting training...")
    print("=" * 70)
    
    trainer = Trainer(model, config, model_type=args.model)
    
    # Tokenizer를 trainer에 설정 (예측 생성용)
    trainer.tokenizer = tokenizer
    
    best_accuracy = trainer.train(
        train_dataset, 
        eval_dataset, 
        resume_from=args.resume
    )
    
    print(f"\n✅ Training completed!")
    print(f"   Dataset: {args.dataset}")
    print(f"   Model: {args.model}")
    print(f"   Best accuracy: {best_accuracy:.4f}")
    
    # 추가 분석 (Connection Transformer만)
    if args.model == "connection":
        print(f"\n🔍 Analyzing connection patterns...")
        from utils.visualization import visualize_connection_matrix, analyze_reasoning_patterns
        
        if not args.no_save:
            # Connection matrix 시각화
            visualize_connection_matrix(
                model, 
                save_path=os.path.join(args.output_dir, f"connection_matrix_{args.dataset}_{args.model_size}.png"),
                title_suffix=f" ({args.dataset})"
            )
            
            # 추론 패턴 분석
            analyze_reasoning_patterns(
                model,
                save_path=os.path.join(args.output_dir, f"reasoning_patterns_{args.dataset}_{args.model_size}.png")
            )
        
        # Connection 통계 출력
        analysis = model.get_connection_analysis()
        print(f"   Connection Statistics:")
        print(f"     Max strength: {analysis['max_connection']:.4f}")
        print(f"     Mean strength: {analysis['mean_connection']:.4f}")
        print(f"     Sparsity ratio: {analysis['sparsity_ratio']:.2%}")

if __name__ == "__main__":
    main()