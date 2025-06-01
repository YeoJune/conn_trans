# training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from training.data_collator import T5DataCollator
import time
import os
import json

class Trainer:
    def __init__(self, model, config, model_type="connection"):
        self.model = model
        self.config = config
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        
        # GPU 설정
        self.model.to(self.device)
        if config.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Mixed precision 설정
        self.use_bf16 = getattr(config, 'bf16', True) and torch.cuda.is_bf16_supported()
        self.use_fp16 = getattr(config, 'fp16', False) and not self.use_bf16
        self.scaler = torch.cuda.amp.GradScaler() if self.use_fp16 else None
        
        # 메트릭 추적
        self.train_losses = []
        self.eval_accuracies = []
        self.reasoning_steps_history = []
        
        # 기본 설정
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        
        print(f"🚀 Trainer: {model_type} model, {self.device}, "
              f"{'bf16' if self.use_bf16 else 'fp16' if self.use_fp16 else 'fp32'}")

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        if hasattr(self.model, 'pad_token_id'):
            self.model.pad_token_id = tokenizer.pad_token_id
    
    def setup_optimizer_and_scheduler(self, train_loader):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        accumulated_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Encoder-Decoder 입력 처리
                src_input_ids = batch['input_ids'].to(self.device)
                src_attention_mask = batch['attention_mask'].to(self.device)
                tgt_input_ids = batch.get('decoder_input_ids', batch['input_ids']).to(self.device)
                tgt_attention_mask = batch.get('decoder_attention_mask', batch['attention_mask']).to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass with autocast
                autocast_kwargs = {'device_type': 'cuda', 'dtype': torch.bfloat16 if self.use_bf16 else torch.float16}
                
                if self.use_bf16 or self.use_fp16:
                    with torch.amp.autocast(**autocast_kwargs):
                        logits = self.model(src_input_ids, tgt_input_ids, src_attention_mask, tgt_attention_mask)
                        loss = self.calculate_loss(logits, labels)
                        loss = loss / self.gradient_accumulation_steps
                else:
                    logits = self.model(src_input_ids, tgt_input_ids, src_attention_mask, tgt_attention_mask)
                    loss = self.calculate_loss(logits, labels)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.use_fp16 and self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulated_loss += loss.item()
                
                # Gradient step
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_fp16 and self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    total_loss += accumulated_loss
                    accumulated_loss = 0
                
                num_batches += 1
                
                # 로그
                if batch_idx % self.config.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"  Epoch {epoch} [{batch_idx:4d}/{len(train_loader)}] "
                          f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f} LR: {lr:.2e}")
                
            except torch.cuda.OutOfMemoryError:
                print(f"🚨 OOM at batch {batch_idx}, skipping...")
                torch.cuda.empty_cache()
                continue
        
        return total_loss / max(num_batches // self.gradient_accumulation_steps, 1)
    
    def calculate_loss(self, logits, labels):
        loss_fct = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=getattr(self.config, 'label_smoothing', 0.1)
        )
        
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        
        loss = loss_fct(flat_logits, flat_labels)
        
        # Connection Transformer 정규화
        if self.model_type == "connection":
            if hasattr(self.model, 'orthogonal_regularization_loss'):
                orth_loss = self.model.orthogonal_regularization_loss()
                orth_weight = getattr(self.config, 'orthogonal_weight', 0.01)
                loss += orth_weight * orth_loss
        
        return loss
    
    def evaluate(self, eval_loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in eval_loader:
                try:
                    src_input_ids = batch['input_ids'].to(self.device)
                    src_attention_mask = batch['attention_mask'].to(self.device)
                    tgt_input_ids = batch.get('decoder_input_ids', batch['input_ids']).to(self.device)
                    tgt_attention_mask = batch.get('decoder_attention_mask', batch['attention_mask']).to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    if self.use_bf16 or self.use_fp16:
                        autocast_kwargs = {'device_type': 'cuda', 'dtype': torch.bfloat16 if self.use_bf16 else torch.float16}
                        with torch.amp.autocast(**autocast_kwargs):
                            logits = self.model(src_input_ids, tgt_input_ids, src_attention_mask, tgt_attention_mask)
                            loss = self.calculate_loss(logits, labels)
                    else:
                        logits = self.model(src_input_ids, tgt_input_ids, src_attention_mask, tgt_attention_mask)
                        loss = self.calculate_loss(logits, labels)
                    
                    total_loss += loss.item()
                    
                    # 예측 생성 (첫 몇 개만)
                    predicted_ids = torch.argmax(logits, dim=-1)
                    batch_size = min(src_input_ids.size(0), 4)
                    
                    for i in range(batch_size):
                        pred_tokens = predicted_ids[i].cpu()
                        pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                        target_text = batch['target_text'][i] if 'target_text' in batch else 'N/A'
                        
                        predictions.append(pred_text.strip())
                        targets.append(str(target_text).strip())
                
                except Exception as e:
                    print(f"⚠️ Eval error: {e}")
                    continue
        
        avg_loss = total_loss / len(eval_loader) if len(eval_loader) > 0 else 0
        
        # 정확도 계산
        try:
            from utils.metrics import calculate_accuracy
            accuracy = calculate_accuracy(predictions, targets, self.config.dataset_name) if predictions else 0.0
        except:
            accuracy = 0.0
        
        return avg_loss, accuracy, predictions[:5], targets[:5]
    
    def train(self, train_dataset, eval_dataset, resume_from=None):
        # 데이터 콜레이터 설정
        data_collator = T5DataCollator(tokenizer=self.tokenizer, max_length=self.config.max_seq_len)
        
        # 데이터 로더
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=getattr(self.config, 'num_workers', 2), collate_fn=data_collator
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=getattr(self.config, 'num_workers', 2), collate_fn=data_collator
        )
        
        print(f"📊 Train: {len(train_loader)} batches, Eval: {len(eval_loader)} batches")
        
        # 옵티마이저 설정
        self.setup_optimizer_and_scheduler(train_loader)
        
        # 훈련 시작
        best_accuracy = 0.0
        
        print(f"\n🚀 Training {self.config.num_epochs} epochs")
        print("="*50)
        
        for epoch in range(self.config.num_epochs):
            # 훈련
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 평가
            eval_loss, accuracy, predictions, targets = self.evaluate(eval_loader)
            
            # 메트릭 기록
            self.train_losses.append(train_loss)
            self.eval_accuracies.append(accuracy)
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Eval Loss:  {eval_loss:.4f}")
            print(f"  Accuracy:   {accuracy:.4f}")
            
            # 최고 성능 저장
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_checkpoint(epoch, accuracy, is_best=True)
                print(f"  💾 New best: {best_accuracy:.4f}")
            
            print("-" * 50)
        
        print(f"\n✅ Training completed! Best accuracy: {best_accuracy:.4f}")
        self.save_training_results(best_accuracy, predictions, targets)
        
        return best_accuracy
    
    def save_checkpoint(self, epoch, accuracy, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': self.config.to_dict()
        }
        
        filename = f'{"best" if is_best else f"checkpoint_epoch_{epoch}"}_{self.model_type}_{self.config.dataset_name}.pt'
        torch.save(checkpoint, os.path.join(self.config.output_dir, filename))
    
    def save_training_results(self, best_accuracy, predictions, targets):
        results = {
            'model_type': self.model_type,
            'dataset': self.config.dataset_name,
            'best_accuracy': best_accuracy,
            'train_losses': self.train_losses,
            'eval_accuracies': self.eval_accuracies,
            'sample_predictions': predictions,
            'sample_targets': targets,
            'timestamp': time.strftime("%Y%m%d_%H%M%S")
        }
        
        filename = f'results_{self.model_type}_{self.config.dataset_name}_{results["timestamp"]}.json'
        with open(os.path.join(self.config.output_dir, filename), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📊 Results saved to {filename}")
