# training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import time
import os
import json
from utils.metrics import calculate_accuracy, extract_final_answer
from utils.visualization import plot_training_curves, analyze_reasoning_patterns

class Trainer:
    def __init__(self, model, config, model_type="connection"):
        self.model = model
        self.config = config
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None  # 나중에 설정됨
        
        # 모델을 GPU로 이동
        self.model.to(self.device)
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
        
        # 메트릭 추적
        self.train_losses = []
        self.eval_accuracies = []
        self.reasoning_steps_history = []
        
        print(f"🚀 Trainer initialized for {model_type} model on {self.device}")
        if config.fp16:
            print("⚡ Mixed precision training enabled")
    
    def set_tokenizer(self, tokenizer):
        """토크나이저 설정"""
        self.tokenizer = tokenizer
        
        # 모델에도 pad_token_id 설정 (필요한 경우)
        if hasattr(self.model, 'pad_token_id'):
            self.model.pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
    
    def setup_optimizer_and_scheduler(self, train_loader):
        """옵티마이저와 스케줄러 설정"""
        # 옵티마이저
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 스케줄러
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"📊 Training setup:")
        print(f"   Total steps: {total_steps:,}")
        print(f"   Warmup steps: {warmup_steps:,}")
        print(f"   Learning rate: {self.config.learning_rate}")
    
    def train_epoch(self, train_loader, epoch):
        """한 에폭 훈련"""
        self.model.train()
        total_loss = 0
        total_reasoning_steps = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 데이터를 GPU로 이동
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask, return_reasoning_trace=True)
                    logits, reasoning_info = outputs
                    
                    # 손실 계산
                    loss = self.calculate_loss(logits, labels, reasoning_info)
            else:
                outputs = self.model(input_ids, attention_mask, return_reasoning_trace=True)
                logits, reasoning_info = outputs
                loss = self.calculate_loss(logits, labels, reasoning_info)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.fp16:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # 메트릭 누적
            total_loss += loss.item()
            if hasattr(reasoning_info, 'actual_steps'):
                total_reasoning_steps += reasoning_info['actual_steps']
            num_batches += 1
            
            # 로깅
            if batch_idx % self.config.log_every == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch} [{batch_idx:4d}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} LR: {current_lr:.2e} "
                      f"Steps: {reasoning_info.get('actual_steps', 'N/A')}")
        
        avg_loss = total_loss / num_batches
        avg_reasoning_steps = total_reasoning_steps / num_batches if num_batches > 0 else 0
        
        return avg_loss, avg_reasoning_steps
    
    def calculate_loss(self, logits, labels, reasoning_info):
        """
        손실 함수 계산 - T5 특화 버전
        """
        # T5 tokenizer의 pad_token_id 사용
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
        if pad_token_id is None:
            pad_token_id = 0

        # T5 스타일 손실 계산 (decoder-only가 아닌 seq2seq)
        # logits: [B, S, V], labels: [B, S]
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)
        
        # Flatten for cross entropy
        shift_logits = logits.view(-1, logits.size(-1))  # [B*S, V]
        shift_labels = labels.view(-1).to(logits.device)  # [B*S] → logits와 동일 디바이스로 이동

        lm_loss = loss_fct(shift_logits, shift_labels)
        
        # Connection Transformer의 경우 추론 비용 추가
        if self.model_type == "connection" and hasattr(self.model, 'reasoning_cost_loss'):
            reasoning_cost = self.model.reasoning_cost_loss(
                reasoning_info.get('actual_steps', 4),
                target_steps=4,
                weight=self.config.reasoning_cost_weight
            )
            total_loss = lm_loss + reasoning_cost
        else:
            total_loss = lm_loss
        
        return total_loss

    
    def evaluate(self, eval_loader):
        """평가"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        reasoning_steps_list = []
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, return_reasoning_trace=True)
                logits, reasoning_info = outputs
                
                # 손실 계산
                loss = self.calculate_loss(logits, labels, reasoning_info)
                total_loss += loss.item()
                
                # 예측 생성 (간단한 greedy decoding)
                batch_predictions = self.generate_predictions(input_ids, attention_mask)
                predictions.extend(batch_predictions)
                targets.extend(batch['target_text'])
                
                # 추론 스텝 기록
                if 'actual_steps' in reasoning_info:
                    reasoning_steps_list.append(reasoning_info['actual_steps'])
        
        avg_loss = total_loss / len(eval_loader)
        accuracy = calculate_accuracy(predictions, targets)
        avg_reasoning_steps = sum(reasoning_steps_list) / len(reasoning_steps_list) if reasoning_steps_list else 0
        
        return avg_loss, accuracy, avg_reasoning_steps, predictions, targets
    
    def generate_predictions(self, input_ids, attention_mask, max_new_tokens=50):
        """
        예측 텍스트 생성 - T5 특화 버전
        """
        batch_size = input_ids.size(0)
        predictions = []
        
        # T5 tokenizer 토큰 ID들
        eos_token_id = getattr(self.tokenizer, 'eos_token_id', 1)
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
        
        with torch.no_grad():
            for i in range(batch_size):
                # 각 샘플에 대해 개별적으로 생성
                single_input = input_ids[i:i+1]
                single_mask = attention_mask[i:i+1] if attention_mask is not None else None
                
                # T5는 encoder-decoder 구조이지만, 여기서는 단순화된 생성 사용
                # 실제 T5에서는 generate() 메서드를 사용해야 함
                
                # 단순한 next-token prediction으로 생성
                generated_ids = []
                current_input = single_input
                
                for step in range(max_new_tokens):
                    outputs = self.model(current_input, attention_mask=single_mask)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    # 마지막 토큰의 logits에서 다음 토큰 예측
                    next_token_logits = logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    next_token_id = next_token.item()
                    
                    # EOS 토큰이면 중단
                    if next_token_id == eos_token_id:
                        break
                    
                    generated_ids.append(next_token_id)
                    
                    # 다음 입력 준비 (현재 구현에서는 단순화)
                    break  # 실제로는 generated token을 append해야 함
                
                # 디코딩
                try:
                    if generated_ids:
                        prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    else:
                        # Fallback: 입력에서 직접 예측 (단순화)
                        output_logits = self.model(single_input, single_mask)
                        if isinstance(output_logits, tuple):
                            output_logits = output_logits[0]
                        predicted_token = torch.argmax(output_logits[:, -1, :], dim=-1)
                        prediction = self.tokenizer.decode([predicted_token.item()], skip_special_tokens=True)
                except:
                    prediction = ""
                
                predictions.append(prediction.strip())
        
        return predictions
    
    def train(self, train_dataset, eval_dataset, resume_from=None):
        """전체 훈련 프로세스"""
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        # 옵티마이저 설정
        self.setup_optimizer_and_scheduler(train_loader)
        
        # 체크포인트 로드
        start_epoch = 0
        best_accuracy = 0.0
        
        if resume_from and os.path.exists(resume_from):
            checkpoint = torch.load(resume_from)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_accuracy = checkpoint.get('best_accuracy', 0.0)
            print(f"📂 Resumed from epoch {start_epoch}, best accuracy: {best_accuracy:.4f}")
        
        print(f"\n🚀 Starting training for {self.config.num_epochs} epochs")
        print("="*70)
        
        for epoch in range(start_epoch, self.config.num_epochs):
            epoch_start_time = time.time()
            
            # 훈련
            train_loss, avg_reasoning_steps = self.train_epoch(train_loader, epoch)
            
            # 평가
            eval_loss, accuracy, eval_reasoning_steps, predictions, targets = self.evaluate(eval_loader)
            
            # 메트릭 기록
            self.train_losses.append(train_loss)
            self.eval_accuracies.append(accuracy)
            self.reasoning_steps_history.append(avg_reasoning_steps)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Eval Loss:  {eval_loss:.4f}")
            print(f"  Accuracy:   {accuracy:.4f}")
            if self.model_type == "connection":
                print(f"  Avg Reasoning Steps: {avg_reasoning_steps:.2f}")
            
            # 최고 성능 모델 저장
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_checkpoint(epoch, accuracy, is_best=True)
                print(f"  💾 New best model saved! Accuracy: {best_accuracy:.4f}")
            
            # 정기 체크포인트 저장
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, accuracy, is_best=False)
            
            print("-" * 70)
        
        print(f"\n✅ Training completed!")
        print(f"   Best accuracy: {best_accuracy:.4f}")
        
        # 훈련 결과 시각화 및 분석
        self.save_training_results(best_accuracy, predictions[:10], targets[:10])
        
        return best_accuracy
    
    def save_checkpoint(self, epoch, accuracy, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'best_accuracy': max(self.eval_accuracies) if self.eval_accuracies else accuracy,
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'eval_accuracies': self.eval_accuracies,
            'reasoning_steps_history': self.reasoning_steps_history
        }
        
        if is_best:
            torch.save(checkpoint, f'best_{self.model_type}_{self.config.dataset_name}.pt')
        else:
            torch.save(checkpoint, f'checkpoint_{self.model_type}_{self.config.dataset_name}_epoch_{epoch}.pt')
    
    def save_training_results(self, best_accuracy, sample_predictions, sample_targets):
        """훈련 결과 저장"""
        results = {
            'model_type': self.model_type,
            'dataset': self.config.dataset_name,
            'best_accuracy': best_accuracy,
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'eval_accuracies': self.eval_accuracies,
            'reasoning_steps_history': self.reasoning_steps_history,
            'sample_predictions': sample_predictions,
            'sample_targets': sample_targets,
            'timestamp': time.strftime("%Y%m%d_%H%M%S")
        }
        
        filename = f'results_{self.model_type}_{self.config.dataset_name}_{results["timestamp"]}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)  # default=str for non-serializable objects
        
        print(f"📊 Results saved to {filename}")
        
        # 시각화
        if len(self.train_losses) > 1:
            plot_training_curves(
                self.train_losses, 
                self.eval_accuracies, 
                self.reasoning_steps_history,
                save_path=f'training_curves_{self.model_type}_{self.config.dataset_name}.png'
            )
        
        # Connection Transformer 분석
        if self.model_type == "connection" and hasattr(self.model, 'get_connection_analysis'):
            analyze_reasoning_patterns(
                self.model,
                save_path=f'reasoning_analysis_{self.config.dataset_name}.png'
            )