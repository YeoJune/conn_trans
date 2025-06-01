# training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from .data_collator import T5DataCollator
from utils.metrics import calculate_accuracy
import time
import os
import json
from contextlib import nullcontext

class Trainer:
    """Simplified and robust trainer for Connection Transformer"""
    
    def __init__(self, model, config, model_type="connection"):
        self.model = model
        self.config = config
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        
        # Setup model and precision
        self._setup_model()
        self._setup_precision()
        
        # Training state
        self.metrics = {
            'train_losses': [],
            'eval_losses': [],
            'eval_accuracies': [],
            'reasoning_steps': []
        }
        
        print(f"🚀 Trainer: {model_type} | {self.device} | {self.precision_str}")
    
    def _setup_model(self):
        """Setup model on device with gradient checkpointing"""
        self.model.to(self.device)
        
        if hasattr(self.config, 'gradient_checkpointing') and self.config.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
    
    def _setup_precision(self):
        """Setup mixed precision training"""
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
        
        # Autocast context
        if self.use_bf16:
            self.autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)
        elif self.use_fp16:
            self.autocast_ctx = torch.amp.autocast('cuda', dtype=torch.float16)
        else:
            self.autocast_ctx = nullcontext()
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for the trainer"""
        self.tokenizer = tokenizer
        if hasattr(self.model, 'pad_token_id'):
            self.model.pad_token_id = tokenizer.pad_token_id
    
    def _setup_optimizer(self, train_loader):
        """Setup optimizer and scheduler"""
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
        """Extract and move batch tensors to device"""
        return {
            'src_input_ids': batch['input_ids'].to(self.device),
            'src_attention_mask': batch['attention_mask'].to(self.device),
            'tgt_input_ids': batch.get('decoder_input_ids', batch['input_ids']).to(self.device),
            'tgt_attention_mask': batch.get('decoder_attention_mask', batch['attention_mask']).to(self.device),
            'labels': batch['labels'].to(self.device)
        }
    
    def _forward_pass(self, tensors, return_reasoning=False):
        """Single forward pass with autocast"""
        with self.autocast_ctx:
            if return_reasoning and hasattr(self.model, 'forward'):
                # Try to get reasoning info for Connection Transformer
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
            
            # Standard forward pass
            logits = self.model(
                tensors['src_input_ids'], 
                tensors['tgt_input_ids'],
                tensors['src_attention_mask'], 
                tensors['tgt_attention_mask']
            )
            return logits, None
    
    def _calculate_loss(self, logits, labels):
        """Calculate total loss including regularization"""
        # Cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=getattr(self.config, 'label_smoothing', 0.1)
        )
        
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        loss = loss_fct(flat_logits, flat_labels)
        
        # Add regularization for Connection Transformer
        if (self.model_type == "connection" and 
            hasattr(self.model, 'orthogonal_regularization_loss')):
            orth_loss = self.model.orthogonal_regularization_loss()
            orth_weight = getattr(self.config, 'orthogonal_weight', 0.01)
            loss += orth_weight * orth_loss
        
        return loss
    
    def _backward_pass(self, loss):
        """Backward pass with gradient scaling"""
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def _optimizer_step(self):
        """Optimizer step with gradient clipping"""
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
        """Train single epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        log_every = getattr(self.config, 'log_every', 50)
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Extract tensors
                tensors = self._extract_batch_tensors(batch)
                
                # Forward pass
                logits, reasoning_info = self._forward_pass(tensors, return_reasoning=True)
                loss = self._calculate_loss(logits, tensors['labels'])
                
                # Backward pass
                self._backward_pass(loss)
                self._optimizer_step()
                
                # Track metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Track reasoning steps for Connection Transformer
                if reasoning_info and 'actual_steps' in reasoning_info:
                    self.metrics['reasoning_steps'].append(reasoning_info['actual_steps'])
                
                # Logging
                if batch_idx % log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"  Epoch {epoch} [{batch_idx:4d}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f} LR: {lr:.2e}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"🚨 OOM at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self, eval_loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in eval_loader:
                try:
                    # Extract tensors
                    tensors = self._extract_batch_tensors(batch)
                    
                    # Forward pass
                    logits, _ = self._forward_pass(tensors)
                    loss = self._calculate_loss(logits, tensors['labels'])
                    
                    total_loss += loss.item()
                    
                    # Generate predictions (sample first 4 items)
                    predicted_ids = torch.argmax(logits, dim=-1)
                    batch_size = min(logits.size(0), 4)
                    
                    for i in range(batch_size):
                        pred_tokens = predicted_ids[i].cpu()
                        pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                        target_text = batch.get('target_text', [''])[i]
                        
                        predictions.append(pred_text.strip())
                        targets.append(str(target_text).strip())
                
                except Exception as e:
                    print(f"⚠️ Eval error: {str(e)[:50]}...")
                    continue
        
        avg_loss = total_loss / max(len(eval_loader), 1)
        accuracy = calculate_accuracy(predictions, targets, self.config.dataset_name) if predictions else 0.0
        
        return avg_loss, accuracy, predictions[:5], targets[:5]
    
    def train(self, train_dataset, eval_dataset):
        """Main training loop with visualization"""
        # Setup data loaders
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
        
        # Setup optimizer
        self._setup_optimizer(train_loader)
        
        print(f"📊 Train: {len(train_loader)} batches | Eval: {len(eval_loader)} batches")
        print(f"🚀 Training {self.config.num_epochs} epochs\n" + "="*50)
        
        best_accuracy = 0.0
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            eval_loss, accuracy, predictions, targets = self.evaluate(eval_loader)
            
            # Update metrics
            self.metrics['train_losses'].append(train_loss)
            self.metrics['eval_losses'].append(eval_loss)
            self.metrics['eval_accuracies'].append(accuracy)
            
            # Logging
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Eval Loss:  {eval_loss:.4f}")
            print(f"  Accuracy:   {accuracy:.4f}")
            
            # Track reasoning efficiency for Connection Transformer
            avg_steps = None
            if self.metrics['reasoning_steps']:
                avg_steps = sum(self.metrics['reasoning_steps'][-50:]) / len(self.metrics['reasoning_steps'][-50:])
                print(f"  Avg Steps:  {avg_steps:.1f}")
            
            # Generate visualizations during training
            self._generate_training_visualizations(epoch, predictions, targets, avg_steps)
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self._save_checkpoint(epoch, accuracy, is_best=True)
                print(f"  💾 New best: {best_accuracy:.4f}")
            
            print("-" * 50)
        
        print(f"\n✅ Training completed! Best accuracy: {best_accuracy:.4f}")
        self._save_results(best_accuracy, predictions, targets)
        
        return best_accuracy
    
    def _save_checkpoint(self, epoch, accuracy, is_best=False):
        """Save model checkpoint"""
        os.makedirs(getattr(self.config, 'output_dir', './outputs'), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': vars(self.config)
        }
        
        filename = f'{"best" if is_best else f"epoch_{epoch}"}_{self.model_type}_{self.config.dataset_name}.pt'
        filepath = os.path.join(getattr(self.config, 'output_dir', './outputs'), filename)
        torch.save(checkpoint, filepath)
    
    def _generate_training_visualizations(self, epoch, predictions, targets, avg_steps):
        """Generate and save visualizations during training"""
        output_dir = getattr(self.config, 'output_dir', './outputs')
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        try:
            from utils.visualization import plot_training_curves, visualize_connection_matrix, plot_accuracy_breakdown
            
            # 1. Training curves (every epoch)
            if len(self.metrics['train_losses']) > 0:
                reasoning_steps_avg = None
                if self.metrics['reasoning_steps']:
                    # Convert to epoch-wise averages
                    reasoning_steps_avg = []
                    steps_per_epoch = len(self.metrics['reasoning_steps']) // max(len(self.metrics['train_losses']), 1)
                    for i in range(len(self.metrics['train_losses'])):
                        start_idx = i * steps_per_epoch
                        end_idx = min((i + 1) * steps_per_epoch, len(self.metrics['reasoning_steps']))
                        if start_idx < len(self.metrics['reasoning_steps']):
                            epoch_avg = sum(self.metrics['reasoning_steps'][start_idx:end_idx]) / max(end_idx - start_idx, 1)
                            reasoning_steps_avg.append(epoch_avg)
                
                plot_training_curves(
                    train_losses=self.metrics['train_losses'],
                    eval_accuracies=self.metrics['eval_accuracies'],
                    reasoning_steps=reasoning_steps_avg,
                    save_path=os.path.join(vis_dir, f'training_curves_epoch_{epoch+1}.png')
                )
            
            # 2. Connection Matrix (for Connection Transformer, every 2 epochs or when best)
            if (self.model_type == "connection" and 
                hasattr(self.model, 'get_connection_analysis') and
                (epoch % 2 == 0 or epoch == self.config.num_epochs - 1)):
                
                visualize_connection_matrix(
                    self.model,
                    save_path=os.path.join(vis_dir, f'connection_matrix_epoch_{epoch+1}.png')
                )
                
                # Save connection analysis as text
                analysis = self.model.get_connection_analysis()
                analysis_path = os.path.join(vis_dir, f'connection_analysis_epoch_{epoch+1}.txt')
                with open(analysis_path, 'w') as f:
                    f.write(f"Connection Analysis - Epoch {epoch+1}\n")
                    f.write("=" * 40 + "\n")
                    f.write(f"Sparsity Ratio: {analysis.get('sparsity_ratio', 0):.4f}\n")
                    f.write(f"Max Connection: {analysis.get('max_connection', 0):.4f}\n")
                    f.write(f"Mean Connection: {analysis.get('mean_connection', 0):.4f}\n")
                    if 'orthogonality_quality' in analysis:
                        f.write(f"Orthogonality Quality: {analysis['orthogonality_quality']:.4f}\n")
                    if avg_steps is not None:
                        f.write(f"Average Reasoning Steps: {avg_steps:.2f}\n")
            
            # 3. Accuracy breakdown (every epoch)
            if predictions and targets:
                plot_accuracy_breakdown(
                    predictions=predictions,
                    targets=targets,
                    dataset_type=self.config.dataset_name,
                    save_path=os.path.join(vis_dir, f'accuracy_breakdown_epoch_{epoch+1}.png')
                )
            
            # 4. Log visualization info
            if epoch % 2 == 0:  # Don't spam logs
                print(f"  📈 Visualizations saved to {vis_dir}/")
                
        except Exception as e:
            print(f"  ⚠️ Visualization error: {str(e)[:50]}...")
    
    def _save_final_analysis(self, best_accuracy):
        """Save comprehensive final analysis"""
        output_dir = getattr(self.config, 'output_dir', './outputs')
        analysis_dir = os.path.join(output_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        try:
            # Final training report
            report_path = os.path.join(analysis_dir, f'training_report_{self.model_type}_{self.config.dataset_name}.md')
            with open(report_path, 'w') as f:
                f.write(f"# Training Report: {self.model_type.title()} on {self.config.dataset_name.upper()}\n\n")
                f.write(f"**Final Accuracy**: {best_accuracy:.4f}\n\n")
                
                f.write("## Configuration\n")
                f.write(f"- Model: {self.model_type}\n")
                f.write(f"- Dataset: {self.config.dataset_name}\n")
                f.write(f"- d_model: {self.config.d_model}\n")
                f.write(f"- Batch size: {self.config.batch_size}\n")
                f.write(f"- Learning rate: {self.config.learning_rate}\n")
                f.write(f"- Epochs: {self.config.num_epochs}\n")
                
                if hasattr(self.config, 'num_slots'):
                    f.write(f"- Slots: {self.config.num_slots}\n")
                    f.write(f"- Bilinear rank: {self.config.bilinear_rank}\n")
                    f.write(f"- Max reasoning steps: {self.config.max_reasoning_steps}\n")
                
                f.write("\n## Training Progress\n")
                f.write(f"- Final train loss: {self.metrics['train_losses'][-1]:.4f}\n")
                f.write(f"- Final eval loss: {self.metrics['eval_losses'][-1]:.4f}\n")
                f.write(f"- Best accuracy: {best_accuracy:.4f}\n")
                
                if self.metrics['reasoning_steps']:
                    avg_steps = sum(self.metrics['reasoning_steps']) / len(self.metrics['reasoning_steps'])
                    f.write(f"- Average reasoning steps: {avg_steps:.2f}\n")
                
                f.write("\n## Model Analysis\n")
                if self.model_type == "connection" and hasattr(self.model, 'get_connection_analysis'):
                    analysis = self.model.get_connection_analysis()
                    f.write(f"- Connection sparsity: {analysis.get('sparsity_ratio', 0):.4f}\n")
                    f.write(f"- Max connection strength: {analysis.get('max_connection', 0):.4f}\n")
                    if 'orthogonality_quality' in analysis:
                        f.write(f"- Orthogonality quality: {analysis['orthogonality_quality']:.4f}\n")
                
                total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                f.write(f"- Total parameters: {total_params:,}\n")
            
            print(f"📋 Training report saved: {report_path}")
            
        except Exception as e:
            print(f"⚠️ Analysis error: {e}")
    
    def _save_results(self, best_accuracy, predictions, targets):
        """Save training results with enhanced analysis"""
        output_dir = getattr(self.config, 'output_dir', './outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Enhanced results structure
        results = {
            'model_type': self.model_type,
            'dataset': self.config.dataset_name,
            'best_accuracy': best_accuracy,
            'metrics': self.metrics,
            'sample_predictions': predictions,
            'sample_targets': targets,
            'config': {
                'd_model': self.config.d_model,
                'num_slots': getattr(self.config, 'num_slots', 0),
                'bilinear_rank': getattr(self.config, 'bilinear_rank', 0),
                'max_reasoning_steps': getattr(self.config, 'max_reasoning_steps', 0),
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'num_epochs': self.config.num_epochs,
                'num_decoder_layers': getattr(self.config, 'num_decoder_layers', 0),
                'num_heads': getattr(self.config, 'num_heads', 0)
            },
            'timestamp': time.strftime("%Y%m%d_%H%M%S")
        }
        
        # Add model analysis for Connection Transformer
        if self.model_type == "connection" and hasattr(self.model, 'get_connection_analysis'):
            results['connection_analysis'] = self.model.get_connection_analysis()
        
        # Save JSON results
        filename = f'results_{self.model_type}_{self.config.dataset_name}_{results["timestamp"]}.json'
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📊 Results saved to {filename}")
        
        # Save final analysis
        self._save_final_analysis(best_accuracy)
