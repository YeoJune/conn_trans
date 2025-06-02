# training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from .data_collator import T5DataCollator
from utils.metrics import calculate_accuracy
from utils.result_manager import ResultManager
import time
from contextlib import nullcontext

class Trainer:
    """간소화되고 체계적인 Connection Transformer 훈련기"""
    
    def __init__(self, model, config, model_type="connection"):
        self.model = model
        self.config = config
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        
        # 결과 관리자 설정
        self.result_manager = ResultManager(
            base_dir=getattr(config, 'output_dir', './outputs'),
            model_type=model_type,
            dataset=config.dataset_name,
            model_size=getattr(config, 'model_size', 'unknown')
        )
        
        # 모델 및 정밀도 설정
        self._setup_model()
        self._setup_precision()
        
        print(f"🚀 Trainer: {model_type} | {self.device} | {self.precision_str}")
    
    def _setup_model(self):
        """모델을 디바이스로 이동 및 그래디언트 체크포인팅 설정"""
        self.model.to(self.device)
        
        if hasattr(self.config, 'gradient_checkpointing') and self.config.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
    
    def _setup_precision(self):
        """혼합 정밀도 훈련 설정"""
        self.use_bf16 = (
            getattr(self.config, 'bf16', True) and 
            torch.cuda.is_available() and 
            torch.cuda.is_bf16_supported()
        )
        self.use_fp16 = (
            getattr(self.config, 'fp16', False) and 
            not self.use_bf16 and 
            torch.cuda.is_available()
        )
        
        self.scaler = torch.cuda.amp.GradScaler() if self.use_fp16 else None
        self.precision_str = 'bf16' if self.use_bf16 else 'fp16' if self.use_fp16 else 'fp32'
        
        # Autocast 컨텍스트
        if self.use_bf16:
            self.autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)
        elif self.use_fp16:
            self.autocast_ctx = torch.amp.autocast('cuda', dtype=torch.float16)
        else:
            self.autocast_ctx = nullcontext()
    
    def set_tokenizer(self, tokenizer):
        """토크나이저 설정"""
        self.tokenizer = tokenizer
        if hasattr(self.model, 'pad_token_id'):
            self.model.pad_token_id = tokenizer.pad_token_id
    
    def _setup_optimizer(self, train_loader):
        """옵티마이저 및 스케줄러 설정"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * getattr(self.config, 'warmup_ratio', 0.1))
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
    
    def _extract_batch_tensors(self, batch):
        """배치 텐서 추출 및 디바이스 이동"""
        return {
            'src_input_ids': batch['input_ids'].to(self.device),
            'src_attention_mask': batch['attention_mask'].to(self.device),
            'tgt_input_ids': batch.get('decoder_input_ids', batch['input_ids']).to(self.device),
            'tgt_attention_mask': batch.get('decoder_attention_mask', batch['attention_mask']).to(self.device),
            'labels': batch['labels'].to(self.device)
        }
    
    def _forward_pass(self, tensors, return_reasoning=False):
        """autocast를 사용한 순전파"""
        with self.autocast_ctx:
            if return_reasoning and hasattr(self.model, 'forward'):
                try:
                    output = self.model(
                        tensors['src_input_ids'], 
                        tensors['tgt_input_ids'],
                        tensors['src_attention_mask'], 
                        tensors['tgt_attention_mask'],
                        return_reasoning_trace=True
                    )
                    if isinstance(output, tuple):
                        logits, reasoning_info = output
                        return logits, reasoning_info
                except:
                    pass
            
            # 표준 순전파
            logits = self.model(
                tensors['src_input_ids'], 
                tensors['tgt_input_ids'],
                tensors['src_attention_mask'], 
                tensors['tgt_attention_mask']
            )
            return logits, None
    
    def _calculate_loss(self, logits, labels):
        """정규화를 포함한 총 손실 계산"""
        loss_fct = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=getattr(self.config, 'label_smoothing', 0.1)
        )
        
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        loss = loss_fct(flat_logits, flat_labels)
        
        # Connection Transformer 정규화 추가
        if (self.model_type == "connection" and 
            hasattr(self.model, 'orthogonal_regularization_loss')):
            orth_loss = self.model.orthogonal_regularization_loss()
            orth_weight = getattr(self.config, 'orthogonal_weight', 0.01)
            loss += orth_weight * orth_loss
        
        return loss
    
    def _backward_pass(self, loss):
        """그래디언트 스케일링을 사용한 역전파"""
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def _optimizer_step(self):
        """그래디언트 클리핑을 사용한 옵티마이저 스텝"""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    def train_epoch(self, train_loader, epoch):
        """단일 에포크 훈련"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        reasoning_steps_epoch = []
        log_every = getattr(self.config, 'log_every', 50)
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # 텐서 추출
                tensors = self._extract_batch_tensors(batch)
                
                # 순전파
                logits, reasoning_info = self._forward_pass(tensors, return_reasoning=True)
                loss = self._calculate_loss(logits, tensors['labels'])
                
                # 역전파
                self._backward_pass(loss)
                self._optimizer_step()
                
                # 메트릭 추적
                total_loss += loss.item()
                num_batches += 1
                
                # 추론 단계 추적 (Connection Transformer)
                if reasoning_info and 'actual_steps' in reasoning_info:
                    reasoning_steps_epoch.append(reasoning_info['actual_steps'])
                
                # 로깅
                if batch_idx % log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    msg = f"  Epoch {epoch} [{batch_idx:4d}/{len(train_loader)}] Loss: {loss.item():.4f} LR: {lr:.2e}"
                    print(msg)
                    self.result_manager.log_training(msg)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    error_msg = f"🚨 OOM at batch {batch_idx}, skipping..."
                    print(error_msg)
                    self.result_manager.log_training(error_msg)
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_reasoning_steps = sum(reasoning_steps_epoch) / len(reasoning_steps_epoch) if reasoning_steps_epoch else None
        
        return avg_loss, avg_reasoning_steps
    
    def evaluate(self, eval_loader):
        """모델 평가 - 전체 데이터에 대해 정확도 계산"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        sample_predictions = []  # 시각화용 샘플
        sample_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                try:
                    # 텐서 추출
                    tensors = self._extract_batch_tensors(batch)
                    
                    # 순전파
                    logits, _ = self._forward_pass(tensors)
                    loss = self._calculate_loss(logits, tensors['labels'])
                    
                    total_loss += loss.item()
                    
                    # 전체 배치에 대해 예측 생성
                    predicted_ids = torch.argmax(logits, dim=-1)
                    batch_size = logits.size(0)
                    
                    for i in range(batch_size):
                        pred_tokens = predicted_ids[i].cpu()
                        pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                        target_text = batch.get('target_text', [''])[i] if i < len(batch.get('target_text', [])) else ''
                        
                        # 전체 예측 저장 (정확도 계산용)
                        all_predictions.append(pred_text.strip())
                        all_targets.append(str(target_text).strip())
                        
                        # 처음 몇 개만 샘플로 저장 (시각화용)
                        if len(sample_predictions) < 10:  # 샘플 개수 증가
                            sample_predictions.append(pred_text.strip())
                            sample_targets.append(str(target_text).strip())
                
                except Exception as e:
                    print(f"⚠️ Eval error: {str(e)[:50]}...")
                    continue
        
        avg_loss = total_loss / max(len(eval_loader), 1)
        
        # 전체 데이터에 대해 정확도 계산
        accuracy = calculate_accuracy(all_predictions, all_targets, self.config.dataset_name) if all_predictions else 0.0
        
        # 시각화용으로는 샘플만 반환
        return avg_loss, accuracy, sample_predictions, sample_targets
    
    def train(self, train_dataset, eval_dataset):
        """주 훈련 루프"""
        # 설정 저장
        self.result_manager.save_config(self.config)
        
        # 데이터 로더 설정
        data_collator = T5DataCollator(self.tokenizer, max_length=self.config.max_seq_len)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            collate_fn=data_collator,
            num_workers=getattr(self.config, 'num_workers', 0),
            pin_memory=torch.cuda.is_available()
        )
        
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            collate_fn=data_collator,
            num_workers=getattr(self.config, 'num_workers', 0),
            pin_memory=torch.cuda.is_available()
        )
        
        # 옵티마이저 설정
        self._setup_optimizer(train_loader)
        
        info_msg = f"📊 Train: {len(train_loader)} batches | Eval: {len(eval_loader)} batches"
        start_msg = f"🚀 Training {self.config.num_epochs} epochs"
        
        print(info_msg)
        print(start_msg + "\n" + "="*50)
        self.result_manager.log_training(info_msg)
        self.result_manager.log_training(start_msg)
        
        best_accuracy = 0.0
        final_predictions = []
        final_targets = []
        
        for epoch in range(self.config.num_epochs):
            # 훈련
            train_loss, avg_reasoning_steps = self.train_epoch(train_loader, epoch)
            
            # 평가
            eval_loss, accuracy, predictions, targets = self.evaluate(eval_loader)
            
            # 메트릭 업데이트
            self.result_manager.update_metrics(
                epoch=epoch,
                train_loss=train_loss,
                eval_loss=eval_loss,
                accuracy=accuracy,
                reasoning_steps=avg_reasoning_steps
            )
            
            # 로깅
            results_msg = f"\nEpoch {epoch + 1}/{self.config.num_epochs}\n"
            results_msg += f"  Train Loss: {train_loss:.4f}\n"
            results_msg += f"  Eval Loss:  {eval_loss:.4f}\n"
            results_msg += f"  Accuracy:   {accuracy:.4f}\n"
            
            if avg_reasoning_steps is not None:
                results_msg += f"  Avg Steps:  {avg_reasoning_steps:.1f}\n"
            
            print(results_msg)
            self.result_manager.log_training(results_msg)
            
            # 최고 모델 저장
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                final_predictions = predictions
                final_targets = targets
                
                self.result_manager.save_checkpoint(
                    self.model, self.optimizer, epoch, accuracy, is_best=True
                )
                
                best_msg = f"  💾 New best: {best_accuracy:.4f}"
                print(best_msg)
                self.result_manager.log_training(best_msg)
            
            print("-" * 50)
        
        complete_msg = f"\n✅ Training completed! Best accuracy: {best_accuracy:.4f}"
        print(complete_msg)
        self.result_manager.log_training(complete_msg)
        
        # 최종 분석 수행
        self.result_manager.finalize_training(
            best_accuracy, self.model, final_predictions, final_targets
        )
        
        return best_accuracy