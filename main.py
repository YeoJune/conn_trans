# main.py

import torch
import argparse
import os
from models.connection_transformer import ConnectionTransformer
from models.baseline_transformer import BaselineTransformer
from training.trainer import Trainer
from dataset.tokenizer_utils import get_tokenizer_and_dataset

# Config imports
import configs.logiqa_config as logiqa_cfg
import configs.gsm8k_config as gsm8k_cfg
import configs.strategyqa_config as strategyqa_cfg
import configs.multinli_config as multinli_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["logiqa", "gsm8k", "strategyqa", "multinli"], required=True)
    parser.add_argument("--model", choices=["connection", "baseline"], required=True)
    parser.add_argument("--model_size", choices=["nano", "micro", "tiny", "base"], default="micro")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()
    
    # 환경 설정
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Config 로드
    config_map = {
        "logiqa": logiqa_cfg.get_config,
        "gsm8k": gsm8k_cfg.get_config,
        "strategyqa": strategyqa_cfg.get_config,
        "multinli": multinli_cfg.get_config
    }
    
    config = config_map[args.dataset](args.model_size)
    config.update(output_dir=args.output_dir)
    
    print(f"📋 {args.dataset} + {args.model} + {args.model_size}")
    print(f"   d_model={config.d_model}, batch_size={config.batch_size}")
    
    # 데이터 로드
    tokenizer, train_dataset, eval_dataset = get_tokenizer_and_dataset(args.dataset, config)
    
    # 모델 생성
    if args.model == "connection":
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
    else:
        # 간단한 baseline 매칭
        model = BaselineTransformer(
            src_vocab_size=config.vocab_size,
            tgt_vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_encoder_layers=4,
            num_decoder_layers=config.num_decoder_layers,
            ffn_multiplier=4,
            max_seq_len=config.max_seq_len,
            src_pad_token_id=config.pad_token_id,
            tgt_pad_token_id=config.pad_token_id
        )
    
    # 훈련
    trainer = Trainer(model, config, model_type=args.model)
    trainer.set_tokenizer(tokenizer)
    
    best_accuracy = trainer.train(train_dataset, eval_dataset)
    
    print(f"✅ 완료! 최고 정확도: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()