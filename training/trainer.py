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

class Trainer:
    def __init__(self, model, config, model_type="connection"):
        self.model = model
        self.config = config
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        
        # GPU 메모리 확인
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU Memory before model loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        self.model.to(self.device)
        
        if torch.cuda.is_available():
            print(f"GPU Memory after model loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Gradient checkpointing
        if config.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
        # 🔥 T5 최적화: Mixed precision 설정
        self.use_fp16 = getattr(config, 'fp16', False)
        self.use_bf16 = getattr(config, 'bf16', True) and torch.cuda.is_bf16_supported()
        
        if self.use_bf16:
            print("⚡ BFloat16 training enabled (T5 optimized)")
            self.scaler = None  # bf16은 scaler 불필요
        elif self.use_fp16:
            print("⚡ Float16 training enabled")
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            print("🔧 Float32 training (safer for T5)")
            self.scaler = None
        
        # 메트릭 추적
        self.train_losses = []
        self.eval_accuracies = []
        self.reasoning_steps_history = []
        self.orthogonal_losses = []
        
        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        
        # T5 특화 설정
        if not hasattr(config, 'orthogonal_weight'):
            config.orthogonal_weight = 0.01
        
        print(f"🚀 T5-Optimized Trainer initialized for {model_type} model on {self.device}")
        print(f"   Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"   Precision: {'bf16' if self.use_bf16 else 'fp16' if self.use_fp16 else 'fp32'}")
        
        if model_type == "connection":
            print(f"   Orthogonal regularization weight: {config.orthogonal_weight}")
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total trainable parameters: {total_params:,}")

    def setup_data_collator(self):
        """T5용 데이터 콜레이터 설정"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set before setting up data collator")
        
        return T5DataCollator(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.config.max_seq_len,
            return_tensors="pt"
        )
    
    def set_tokenizer(self, tokenizer):
        """토크나이저 설정"""
        self.tokenizer = tokenizer
        
        # 모델에도 pad_token_id 설정
        if hasattr(self.model, 'pad_token_id'):
            self.model.pad_token_id = tokenizer.pad_token_id
    
    def setup_optimizer_and_scheduler(self, train_loader):
        """T5 최적화된 옵티마이저와 스케줄러"""
        # T5는 더 높은 학습률 필요 (HuggingFace 문서 권장)
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
        
        print(f"📊 T5 Training setup:")
        print(f"   Total steps: {total_steps:,}")
        print(f"   Warmup steps: {warmup_steps:,}")
        print(f"   Learning rate: {self.config.learning_rate} (T5 optimized)")
    
    def train_epoch(self, train_loader, epoch):
        """T5 최적화된 훈련 에폭"""
        self.model.train()
        total_loss = 0
        total_reasoning_steps = 0
        total_orthogonal_loss = 0
        num_batches = 0
        accumulated_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # 🔥 T5 최적화: 적절한 precision 사용
                if self.use_bf16:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = self.model(input_ids, attention_mask, return_reasoning_trace=True)
                        logits, reasoning_info = outputs
                        loss = self.calculate_loss(logits, labels, reasoning_info)
                        loss = loss / self.gradient_accumulation_steps
                elif self.use_fp16:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(input_ids, attention_mask, return_reasoning_trace=True)
                        logits, reasoning_info = outputs
                        loss = self.calculate_loss(logits, labels, reasoning_info)
                        loss = loss / self.gradient_accumulation_steps
                else:
                    outputs = self.model(input_ids, attention_mask, return_reasoning_trace=True)
                    logits, reasoning_info = outputs
                    loss = self.calculate_loss(logits, labels, reasoning_info)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.use_fp16 and self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulated_loss += loss.item()
                
                # Orthogonal loss 추적
                if self.model_type == "connection" and hasattr(self.model, 'orthogonal_regularization_loss'):
                    with torch.no_grad():
                        orth_loss = self.model.orthogonal_regularization_loss()
                        total_orthogonal_loss += orth_loss.item()
                
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
                
                if isinstance(reasoning_info, dict) and 'actual_steps' in reasoning_info:
                    total_reasoning_steps += reasoning_info['actual_steps']
                num_batches += 1
                
                # 메모리 정리 (T5는 메모리 많이 사용)
                if batch_idx % getattr(self.config, 'empty_cache_every', 25) == 0:
                    torch.cuda.empty_cache()
                
                if batch_idx % self.config.log_every == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    actual_steps = reasoning_info.get('actual_steps', 'N/A') if isinstance(reasoning_info, dict) else 'N/A'
                    memory_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    
                    log_msg = (f"  Epoch {epoch} [{batch_idx:4d}/{len(train_loader)}] "
                              f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f} "
                              f"LR: {current_lr:.2e} Steps: {actual_steps} "
                              f"GPU: {memory_used:.1f}GB")
                    
                    if self.model_type == "connection" and total_orthogonal_loss > 0:
                        avg_orth_loss = total_orthogonal_loss / max(num_batches, 1)
                        log_msg += f" Orth: {avg_orth_loss:.4f}"
                    
                    print(log_msg)
                
            except torch.cuda.OutOfMemoryError as oom_error:
                print(f"🚨 OOM Error at batch {batch_idx}: {oom_error}")
                print(f"   Clearing cache and skipping batch...")
                torch.cuda.empty_cache()
                if hasattr(self.optimizer, 'zero_grad'):
                    self.optimizer.zero_grad()
                continue
            
            except Exception as other_error:
                print(f"⚠️ Error at batch {batch_idx}: {other_error}")
                continue
        
        avg_loss = total_loss / max(num_batches // self.gradient_accumulation_steps, 1)
        avg_reasoning_steps = total_reasoning_steps / num_batches if num_batches > 0 else 0
        avg_orthogonal_loss = total_orthogonal_loss / num_batches if num_batches > 0 else 0
        
        return avg_loss, avg_reasoning_steps, avg_orthogonal_loss
    
    def calculate_loss(self, logits, labels, reasoning_info):
        """
        T5 최적화된 손실 함수 계산
        """
        # T5 tokenizer의 pad_token_id 사용 (기본값: 0)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else 0
        
        # 🔥 T5 중요: CrossEntropyLoss에서 -100인 토큰은 자동으로 무시됨
        loss_fct = nn.CrossEntropyLoss(
            ignore_index=-100,  # T5에서 padding은 -100으로 처리
            label_smoothing=getattr(self.config, 'label_smoothing', 0.1)  # T5에 효과적
        )
        
        # Logits reshape: [batch_size * seq_len, vocab_size]
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        
        lm_loss = loss_fct(flat_logits, flat_labels)
        
        # Connection Transformer 정규화 추가
        total_loss = lm_loss
        
        if self.model_type == "connection":
            # 1. Reasoning cost loss
            if hasattr(self.model, 'reasoning_cost_loss'):
                reasoning_cost = self.model.reasoning_cost_loss(
                    reasoning_info.get('actual_steps', 4),
                    target_steps=4,
                    weight=self.config.reasoning_cost_weight
                )
                total_loss += reasoning_cost
            
            # 2. Orthogonal regularization loss
            if hasattr(self.model, 'orthogonal_regularization_loss'):
                orthogonal_loss = self.model.orthogonal_regularization_loss()
                orthogonal_weight = getattr(self.config, 'orthogonal_weight', 0.01)
                total_loss += orthogonal_weight * orthogonal_loss
        
        return total_loss
    
    def evaluate(self, eval_loader):
        """T5 최적화된 평가 (배치 크기 문제 해결)"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        reasoning_steps_list = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                try:
                    # 텐서 데이터 GPU로 이동
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                    labels = batch['labels'].to(self.device, non_blocking=True)
                    
                    # 실제 배치 크기 확인 (중요!)
                    actual_batch_size = input_ids.size(0)
                    
                    # T5 최적화된 forward pass
                    if self.use_bf16:
                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            outputs = self.model(input_ids, attention_mask, return_reasoning_trace=True)
                            logits, reasoning_info = outputs
                            loss = self.calculate_loss(logits, labels, reasoning_info)
                    elif self.use_fp16:
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            outputs = self.model(input_ids, attention_mask, return_reasoning_trace=True)
                            logits, reasoning_info = outputs
                            loss = self.calculate_loss(logits, labels, reasoning_info)
                    else:
                        outputs = self.model(input_ids, attention_mask, return_reasoning_trace=True)
                        logits, reasoning_info = outputs
                        loss = self.calculate_loss(logits, labels, reasoning_info)
                    
                    total_loss += loss.item()
                    
                    # 예측 생성 (T5 디코딩)
                    predicted_ids = torch.argmax(logits, dim=-1)
                    
                    # target_text 안전하게 처리
                    batch_target_texts = []
                    if 'target_text' in batch:
                        if isinstance(batch['target_text'], (list, tuple)):
                            batch_target_texts = list(batch['target_text'])
                        else:
                            # 단일 문자열인 경우 배치 크기만큼 복제
                            batch_target_texts = [str(batch['target_text'])] * actual_batch_size
                    else:
                        # target_text가 없는 경우 기본값
                        batch_target_texts = ['N/A'] * actual_batch_size
                    
                    # 배치 크기 안전 확인
                    if len(batch_target_texts) < actual_batch_size:
                        # 부족한 만큼 'N/A'로 채우기
                        batch_target_texts.extend(['N/A'] * (actual_batch_size - len(batch_target_texts)))
                    elif len(batch_target_texts) > actual_batch_size:
                        # 넘치는 만큼 자르기
                        batch_target_texts = batch_target_texts[:actual_batch_size]
                    
                    # 배치의 각 샘플 처리 (메모리 절약을 위해 최대 4개만)
                    num_samples_to_process = min(actual_batch_size, 4)
                    
                    for i in range(num_samples_to_process):
                        try:
                            # 예측 디코딩
                            pred_tokens = predicted_ids[i]
                            
                            # -100 토큰 제거 (T5에서는 보통 필요 없지만 안전하게)
                            if hasattr(pred_tokens, 'cpu'):
                                pred_tokens_clean = pred_tokens.cpu()
                            else:
                                pred_tokens_clean = pred_tokens
                            
                            # 패딩 토큰 제거
                            if self.tokenizer.pad_token_id is not None:
                                mask = pred_tokens_clean != self.tokenizer.pad_token_id
                                pred_tokens_clean = pred_tokens_clean[mask]
                            
                            # 텍스트 디코딩
                            pred_text = self.tokenizer.decode(pred_tokens_clean, skip_special_tokens=True)
                            
                            # 타겟 텍스트 가져오기
                            target_text = batch_target_texts[i] if i < len(batch_target_texts) else 'N/A'
                            
                            # 결과 저장
                            predictions.append(pred_text.strip() if pred_text else "")
                            targets.append(str(target_text).strip())
                            
                        except Exception as decode_error:
                            print(f"      ⚠️ Decode error for sample {i}: {decode_error}")
                            predictions.append("DECODE_ERROR")
                            targets.append("N/A")
                    
                    # 추론 스텝 기록
                    if isinstance(reasoning_info, dict) and 'actual_steps' in reasoning_info:
                        reasoning_steps_list.append(reasoning_info['actual_steps'])
                    
                    # 메모리 정리
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                    
                    # 진행 상황 로그 (선택적)
                    if batch_idx % 50 == 0:
                        print(f"      Eval batch {batch_idx}/{len(eval_loader)}, "
                              f"Loss: {loss.item():.4f}, "
                              f"Samples: {len(predictions)}")
                        
                except torch.cuda.OutOfMemoryError:
                    print(f"🚨 OOM during evaluation at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                    
                except Exception as e:
                    print(f"⚠️ Evaluation error at batch {batch_idx}: {e}")
                    print(f"   Batch keys: {batch.keys() if hasattr(batch, 'keys') else 'Not a dict'}")
                    if hasattr(batch, 'get'):
                        print(f"   Input shape: {batch.get('input_ids', torch.tensor([])).shape}")
                        print(f"   Labels shape: {batch.get('labels', torch.tensor([])).shape}")
                    continue
        
        avg_loss = total_loss / len(eval_loader) if len(eval_loader) > 0 else 0
        
        # 정확도 계산 (안전하게)
        try:
            if predictions and targets and len(predictions) == len(targets):
                from utils.metrics import calculate_accuracy
                accuracy = calculate_accuracy(predictions, targets, self.config.dataset_name)
            else:
                print(f"⚠️ Prediction/target mismatch: {len(predictions)} vs {len(targets)}")
                accuracy = 0.0
        except Exception as acc_error:
            print(f"⚠️ Accuracy calculation error: {acc_error}")
            accuracy = 0.0
        
        avg_reasoning_steps = sum(reasoning_steps_list) / len(reasoning_steps_list) if reasoning_steps_list else 0
        
        # 결과 요약 출력
        print(f"   📊 Evaluation summary:")
        print(f"      Total samples processed: {len(predictions)}")
        print(f"      Average loss: {avg_loss:.4f}")
        print(f"      Accuracy: {accuracy:.4f}")
        if reasoning_steps_list:
            print(f"      Average reasoning steps: {avg_reasoning_steps:.2f}")
        
        return avg_loss, accuracy, avg_reasoning_steps, predictions[:10], targets[:10]
    
    def train(self, train_dataset, eval_dataset, resume_from=None):
        """전체 훈련 프로세스 (T5 최적화, 배치 크기 문제 해결)"""
        
        # T5용 데이터 콜레이터 설정
        from training.data_collator import T5DataCollator
        data_collator = T5DataCollator(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.config.max_seq_len,
            return_tensors="pt"
        )
        
        # 데이터 로더 생성 (데이터 콜레이터 적용)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=getattr(self.config, 'num_workers', 2),
            pin_memory=getattr(self.config, 'pin_memory', True),
            collate_fn=data_collator  # 여기가 중요!
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=getattr(self.config, 'num_workers', 2),
            pin_memory=getattr(self.config, 'pin_memory', True),
            collate_fn=data_collator  # 여기도!
        )
        
        print(f"📊 Data loaders created:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Eval batches: {len(eval_loader)}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Data collator: T5DataCollator (배치 크기 문제 해결)")
        
        # 첫 번째 배치 테스트
        try:
            print(f"\n🔍 Testing first batch...")
            first_batch = next(iter(train_loader))
            print(f"   Batch keys: {list(first_batch.keys())}")
            
            for key, value in first_batch.items():
                if torch.is_tensor(value):
                    print(f"   {key}: {value.shape}")
                elif isinstance(value, (list, tuple)):
                    print(f"   {key}: list of {len(value)} items")
                else:
                    print(f"   {key}: {type(value)}")
            
            print(f"   ✅ First batch test passed")
            
        except Exception as batch_error:
            print(f"   ❌ First batch test failed: {batch_error}")
            raise RuntimeError("Data loader setup failed")
        
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
        
        print(f"\n🚀 Starting T5-optimized training for {self.config.num_epochs} epochs")
        print("="*70)
        
        for epoch in range(start_epoch, self.config.num_epochs):
            epoch_start_time = time.time()
            
            # 훈련
            train_loss, avg_reasoning_steps, avg_orthogonal_loss = self.train_epoch(train_loader, epoch)
            
            # 평가
            eval_loss, accuracy, eval_reasoning_steps, predictions, targets = self.evaluate(eval_loader)
            
            # 메트릭 기록
            self.train_losses.append(train_loss)
            self.eval_accuracies.append(accuracy)
            self.reasoning_steps_history.append(avg_reasoning_steps)
            
            if self.model_type == "connection":
                self.orthogonal_losses.append(avg_orthogonal_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Eval Loss:  {eval_loss:.4f}")
            print(f"  Accuracy:   {accuracy:.4f}")
            if self.model_type == "connection":
                print(f"  Avg Reasoning Steps: {avg_reasoning_steps:.2f}")
                print(f"  Orthogonal Loss: {avg_orthogonal_loss:.4f}")
                
                # Connection 품질 분석
                if hasattr(self.model, 'get_connection_analysis'):
                    analysis = self.model.get_connection_analysis()
                    print(f"  Connection Quality:")
                    print(f"    Max strength: {analysis['max_connection']:.4f}")
                    print(f"    Mean strength: {analysis['mean_connection']:.4f}")
                    if 'orthogonality_quality' in analysis:
                        print(f"    Orthogonality quality: {analysis['orthogonality_quality']:.4f}")
            
            # 최고 성능 모델 저장
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_checkpoint(epoch, accuracy, is_best=True)
                print(f"  💾 New best model saved! Accuracy: {best_accuracy:.4f}")
            
            # 정기 체크포인트 저장
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, accuracy, is_best=False)
            
            print("-" * 70)
        
        print(f"\n✅ T5-optimized training completed!")
        print(f"   Best accuracy: {best_accuracy:.4f}")
        
        # 훈련 결과 저장
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
            'reasoning_steps_history': self.reasoning_steps_history,
            'model_type': self.model_type,
            'precision': 'bf16' if self.use_bf16 else 'fp16' if self.use_fp16 else 'fp32'
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.output_dir, f'best_{self.model_type}_{self.config.dataset_name}.pt'))
        else:
            torch.save(checkpoint, os.path.join(self.config.output_dir, f'checkpoint_{self.model_type}_{self.config.dataset_name}_epoch_{epoch}.pt'))

    def save_training_results(self, best_accuracy, sample_predictions, sample_targets):
        """T5 최적화된 훈련 결과 저장"""
        
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
            'timestamp': time.strftime("%Y%m%d_%H%M%S"),
            't5_optimizations': {
                'precision': 'bf16' if self.use_bf16 else 'fp16' if self.use_fp16 else 'fp32',
                'tokenizer': self.config.tokenizer_name,
                'learning_rate': self.config.learning_rate,
                'gradient_clip': self.config.gradient_clip,
                'label_smoothing': getattr(self.config, 'label_smoothing', 0.1)
            }
        }
        
        # Connection Transformer 전용 메트릭
        if self.model_type == "connection":
            results['orthogonal_losses'] = getattr(self, 'orthogonal_losses', [])
            
            # 최종 connection 분석
            if hasattr(self.model, 'get_connection_analysis'):
                final_analysis = self.model.get_connection_analysis()
                results['final_connection_analysis'] = {
                    'max_connection': final_analysis['max_connection'],
                    'mean_connection': final_analysis['mean_connection'],
                    'sparsity_ratio': final_analysis['sparsity_ratio']
                }
                if 'orthogonality_quality' in final_analysis:
                    results['final_connection_analysis']['orthogonality_quality'] = final_analysis['orthogonality_quality']
                    results['final_connection_analysis']['orthogonality_error'] = final_analysis['orthogonality_error']
        
        # 결과 파일 저장
        filename = os.path.join(self.config.output_dir, f'results_{self.model_type}_{self.config.dataset_name}_{results["timestamp"]}.json')
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📊 T5-optimized results saved to {filename}")
        
        # 🎨 간소화된 시각화 생성
        if len(self.train_losses) > 1:
            try:
                from utils.visualization import plot_training_curves, plot_accuracy_breakdown
                
                # 훈련 곡선
                plot_training_curves(
                    self.train_losses, 
                    self.eval_accuracies, 
                    self.reasoning_steps_history if self.model_type == "connection" else None,
                    save_path=os.path.join(self.config.output_dir, f'training_curves_{self.model_type}_{self.config.dataset_name}.png')
                )
                
                # 정확도 분석
                if sample_predictions and sample_targets:
                    plot_accuracy_breakdown(
                        sample_predictions,
                        sample_targets,
                        self.config.dataset_name,
                        save_path=os.path.join(self.config.output_dir, f'accuracy_breakdown_{self.model_type}_{self.config.dataset_name}.png')
                    )
                
            except ImportError as e:
                print(f"⚠️ Visualization not available: {e}")
        
        # Connection Transformer 전용 분석
        if self.model_type == "connection" and hasattr(self.model, 'get_connection_analysis'):
            try:
                from utils.visualization import visualize_connection_matrix, analyze_reasoning_patterns
                
                # Connection matrix 시각화
                visualize_connection_matrix(
                    self.model,
                    save_path=os.path.join(self.config.output_dir, f'connection_matrix_{self.config.dataset_name}.png'),
                    title_suffix=f" ({self.config.dataset_name})"
                )
                
                # 추론 패턴 분석
                analyze_reasoning_patterns(
                    self.model,
                    save_path=os.path.join(self.config.output_dir, f'reasoning_patterns_{self.config.dataset_name}.png')
                )
                
            except ImportError as e:
                print(f"⚠️ Connection analysis visualization not available: {e}")