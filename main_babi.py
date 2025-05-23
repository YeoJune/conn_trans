import torch
import torch.nn as nn  # <--- nn 임포트 추가!
import torch.nn.functional as F  # <--- F 임포트 추가! (train_babi_model에서 사용)
import torch.optim as optim  # <--- optim 임포트 추가! (train_babi_model에서 사용)
import numpy as np
import time
import json
import warnings
import math
from torch.utils.data import DataLoader  # DataLoader 임포트 확인

# 설정 파일 및 모듈 임포트
from configs.babi_config import get_babi_config
from data_processing.babi_dataset import BabiDataset
from models.base_conn_trans import ConnectionTransformer  # 부모 클래스로 사용
from models.conn_trans_ffn import ConnTransWithFFN  # 부모 클래스로 사용
from models.standard_transformer import StandardTransformer  # 부모 클래스로 사용
# train_babi_model은 이 파일 내에 정의하거나 training.trainer에서 가져옴
from utils.visualization import visualize_connection_matrix, analyze_reasoning_evolution, print_comparison_results


# JSON 인코더
class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (torch.Tensor)): return o.cpu().tolist()
        if isinstance(o, (torch.float32, torch.float64, torch.float16)): return float(o)
        if isinstance(o, (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8)): return int(o)
        return super(NpEncoder, self).default(o)


# === bAbI 태스크를 위한 모델 수정 ===
class BabiConnectionTransformer(ConnectionTransformer):
    def __init__(self, vocab_size, d_model, num_slots, num_reasoning_steps, max_seq_len,
                 connection_init_std, spectral_radius_limit, **kwargs):  # **kwargs 추가
        # 부모 클래스 __init__ 호출 시, 부모가 받는 인자만 전달
        super().__init__(vocab_size, d_model, num_slots, num_reasoning_steps, max_seq_len,
                         connection_init_std, spectral_radius_limit)  # 여기서 **kwargs 제거

        if hasattr(self, 'qa_outputs_start'): del self.qa_outputs_start
        if hasattr(self, 'qa_outputs_end'): del self.qa_outputs_end
        self.classifier = nn.Linear(d_model, vocab_size)  # nn 사용
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None: nn.init.zeros_(self.classifier.bias)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔹 BabiConnectionTransformer: {total_params:,} trainable parameters (vocab: {vocab_size})")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_reasoning_trace=False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        X_input = self.token_embedding(input_ids) + self.pos_embedding(positions)
        Q_input = self.W_q_input(X_input)
        K_slots = self.W_k_slots(self.H)
        V_input = self.W_v_input(X_input)
        A_compress = F.softmax(Q_input @ K_slots.T / math.sqrt(self.d_model), dim=-1)  # F 사용
        IR_activation = A_compress.transpose(-1, -2) @ V_input
        H_state = self.H.unsqueeze(0).expand(batch_size, -1, -1) + IR_activation
        reasoning_trace_states = [H_state.clone()] if return_reasoning_trace else []
        for step in range(self.num_reasoning_steps):
            if self.d_model != self.num_slots and self.numerical_warnings < 1:
                self.numerical_warnings += 1
            Influence = H_state @ self.C
            H_state = H_state + Influence
            H_state = self.reasoning_norms[step](H_state)
            if return_reasoning_trace:
                reasoning_trace_states.append(H_state.clone())
        Q_output = self.W_q_output(X_input)
        K_final = self.W_k_final(H_state)
        V_final = self.W_v_final(H_state)
        A_expand = F.softmax(Q_output @ K_final.transpose(-1, -2) / math.sqrt(self.d_model), dim=-1)  # F 사용
        Y_output = A_expand @ V_final
        last_token_output = Y_output[:, -1, :]
        logits = self.classifier(last_token_output)
        if return_reasoning_trace:
            return logits, reasoning_trace_states
        else:
            return logits

    def get_reasoning_trace(self, input_ids, attention_mask=None, token_type_ids=None):
        self.eval()
        with torch.no_grad():
            logits, trace_states = self.forward(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_reasoning_trace=True
            )
        norms = []
        if trace_states:
            norms = [torch.norm(state, dim=-1).mean().item() for state in trace_states]
        return trace_states, norms


class BabiConnTransWithFFN(BabiConnectionTransformer):
    def __init__(self, vocab_size, d_model, num_slots, num_reasoning_steps, max_seq_len,
                 connection_init_std, spectral_radius_limit,
                 ffn_dim_multiplier=4, dropout=0.1, **kwargs):  # **kwargs 추가
        super().__init__(vocab_size, d_model, num_slots, num_reasoning_steps, max_seq_len,
                         connection_init_std, spectral_radius_limit)  # **kwargs 제거
        ffn_dim = d_model * ffn_dim_multiplier
        self.reasoning_ffn = nn.Sequential(  # nn 사용
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),  # nn 사용
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout)  # nn 사용
        )
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔸 BabiConnTransWithFFN: {total_params:,} trainable parameters (vocab: {vocab_size})")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_reasoning_trace=False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        X_input = self.token_embedding(input_ids) + self.pos_embedding(positions)
        Q_input = self.W_q_input(X_input)
        K_slots = self.W_k_slots(self.H)
        V_input = self.W_v_input(X_input)
        A_compress = F.softmax(Q_input @ K_slots.T / math.sqrt(self.d_model), dim=-1)  # F 사용
        IR_activation = A_compress.transpose(-1, -2) @ V_input
        H_state = self.H.unsqueeze(0).expand(batch_size, -1, -1) + IR_activation
        reasoning_trace_states = [H_state.clone()] if return_reasoning_trace else []
        for step in range(self.num_reasoning_steps):
            Influence = H_state @ self.C
            H_state_after_conn = H_state + Influence
            H_state_norm_before_ffn = self.reasoning_norms[step](H_state_after_conn)
            ffn_output = self.reasoning_ffn(H_state_norm_before_ffn)
            H_state = H_state_norm_before_ffn + ffn_output
            if return_reasoning_trace:
                reasoning_trace_states.append(H_state.clone())
        Q_output = self.W_q_output(X_input)
        K_final = self.W_k_final(H_state)
        V_final = self.W_v_final(H_state)
        A_expand = F.softmax(Q_output @ K_final.transpose(-1, -2) / math.sqrt(self.d_model), dim=-1)  # F 사용
        Y_output = A_expand @ V_final
        last_token_output = Y_output[:, -1, :]
        logits = self.classifier(last_token_output)
        if return_reasoning_trace:
            return logits, reasoning_trace_states
        else:
            return logits


class BabiStandardTransformer(StandardTransformer):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, ffn_dim_multiplier, dropout, max_seq_len,
                 **kwargs):  # **kwargs 추가
        super().__init__(vocab_size, d_model, num_heads, num_layers, ffn_dim_multiplier, dropout,
                         max_seq_len)  # **kwargs 제거
        if hasattr(self, 'qa_outputs_start'): del self.qa_outputs_start
        if hasattr(self, 'qa_outputs_end'): del self.qa_outputs_end
        self.classifier_babi = nn.Linear(d_model, vocab_size)  # nn 사용
        nn.init.xavier_uniform_(self.classifier_babi.weight)
        if self.classifier_babi.bias is not None: nn.init.zeros_(self.classifier_babi.bias)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔶 BabiStandardTransformer: {total_params:,} trainable parameters (vocab: {vocab_size})")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):  # **kwargs 추가
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        src_key_padding_mask = ~attention_mask if attention_mask is not None else None
        transformer_output = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        normed_output = self.norm(transformer_output)
        last_token_output = normed_output[:, -1, :]
        logits = self.classifier_babi(last_token_output)
        return logits


# bAbI용 train_model
def train_babi_model(model, train_loader, val_loader, config, device='cuda', model_name="Model"):
    model = model.to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),  # optim 사용
                            lr=config["learning_rate"], weight_decay=config["weight_decay"])

    num_training_steps = len(train_loader) * config["max_epochs"]
    actual_warmup_steps = min(config["warmup_steps"], num_training_steps // 10) if num_training_steps > 0 else config[
        "warmup_steps"]
    pct_start_val = float(actual_warmup_steps) / num_training_steps if num_training_steps > 0 else 0.1
    scheduler = optim.lr_scheduler.OneCycleLR(  # optim 사용
        optimizer, max_lr=config["learning_rate"],
        total_steps=num_training_steps if num_training_steps > 0 else None,
        pct_start=pct_start_val,
    ) if num_training_steps > 0 else None

    best_val_acc = 0.0
    print(f"\n🚀 Training {model_name} for bAbI on {device}...")
    print("=" * 50)

    for epoch in range(config["max_epochs"]):
        model.train()
        total_train_loss = 0;
        train_correct = 0;
        train_total_samples = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            answer_ids = batch['answer_ids'].to(device)

            if answer_ids.size(1) == 0: continue
            first_answer_token = answer_ids[:, 0]

            logits = model(input_ids, attention_mask=attention_mask)

            loss = F.cross_entropy(logits, first_answer_token)  # F 사용

            if hasattr(model, 'C') and model.C is not None and "connection_regularization" in config:
                c_reg = config["connection_regularization"] * torch.norm(model.C, 'fro') ** 2
                loss = loss + c_reg
            if hasattr(model, 'enforce_spectral_radius') and model.training:
                model.enforce_spectral_radius(config.get("spectral_radius_limit"))

            optimizer.zero_grad();
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            if scheduler: scheduler.step()

            total_train_loss += loss.item() * input_ids.size(0)
            predicted = torch.argmax(logits, dim=1)
            train_correct += (predicted == first_answer_token).sum().item()
            train_total_samples += input_ids.size(0)

            if batch_idx > 0 and batch_idx % (len(train_loader) // 10 if len(train_loader) >= 10 else 1) == 0:
                current_lr = scheduler.get_last_lr()[0] if scheduler else config["learning_rate"]
                print(f"  E{epoch + 1} B{batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")

        avg_train_loss = total_train_loss / train_total_samples if train_total_samples > 0 else 0
        train_acc = train_correct / train_total_samples if train_total_samples > 0 else 0

        model.eval()
        val_correct = 0;
        val_total_samples = 0;
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                answer_ids = batch['answer_ids'].to(device)
                if answer_ids.size(1) == 0: continue
                first_answer_token = answer_ids[:, 0]

                logits = model(input_ids, attention_mask=attention_mask)
                loss_val = F.cross_entropy(logits, first_answer_token)  # F 사용
                total_val_loss += loss_val.item() * input_ids.size(0)
                predicted = torch.argmax(logits, dim=1)
                val_correct += (predicted == first_answer_token).sum().item()
                val_total_samples += input_ids.size(0)

        avg_val_loss = total_val_loss / val_total_samples if val_total_samples > 0 else 0
        val_acc = val_correct / val_total_samples if val_total_samples > 0 else 0

        epoch_time = time.time() - start_time
        print(f"  Epoch {epoch + 1}/{config['max_epochs']} ({epoch_time:.1f}s)")
        print(f"    Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"    Val Loss:   {avg_val_loss:.4f}, Val Acc:   {val_acc:.4f}")

        if hasattr(model, 'get_connection_stats'):
            try:
                stats = model.get_connection_stats()
                print(f"    ConnStats: SR(I+C)={stats.get('spectral_radius_I_plus_C', float('nan')):.3f}, "
                      f"Frob(C)={stats.get('frobenius_norm', float('nan')):.3f}")
            except Exception as e_stat_print_epoch:
                print(f"Error printing conn stats epoch for bAbI: {e_stat_print_epoch}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_babi_model_{model_name.replace(" ", "_")}.pt')
            print(f"    💾 New best bAbI model saved (Val Acc: {best_val_acc:.4f})")
        print("-" * 30)

    print(f"✅ {model_name} bAbI training completed. Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc


def main_babi():
    CFG_BABI = get_babi_config()
    babi_task_id = 1
    babi_type_prefix_to_load = CFG_BABI.get("babi_hf_type_prefix", "en-10k")

    print(f"🚀 Connection Transformer - bAbI Task qa{babi_task_id} (Type Prefix: {babi_type_prefix_to_load}) Experiment")
    print("=" * 70)

    if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True
    torch.manual_seed(42);
    np.random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    print(f"\n📦 bAbI Task qa{babi_task_id} (Type Prefix: {babi_type_prefix_to_load}) Data Loading...")
    try:
        train_dataset = BabiDataset(task_id=babi_task_id,
                                    babi_type_prefix=babi_type_prefix_to_load,
                                    split='train',
                                    max_seq_len=CFG_BABI["max_seq_len"])
        val_dataset = BabiDataset(task_id=babi_task_id,
                                  babi_type_prefix=babi_type_prefix_to_load,
                                  split='validation',
                                  max_seq_len=CFG_BABI["max_seq_len"])
        print("✅ bAbI Data loading successful")
    except Exception as e:
        print(f"❌ bAbI Data loading failed: {e}")
        import traceback;
        traceback.print_exc()
        return {}

    cpu_cores = torch.multiprocessing.cpu_count() if hasattr(torch.multiprocessing, 'cpu_count') else 2
    nw = 0 if not torch.cuda.is_available() else min(2, cpu_cores // 2 if cpu_cores > 1 else 0)
    train_loader = DataLoader(train_dataset, batch_size=CFG_BABI["batch_size"], shuffle=True, num_workers=nw,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=CFG_BABI["batch_size"], shuffle=False, num_workers=nw,
                            pin_memory=torch.cuda.is_available())

    vocab_size = train_dataset.vocab_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(
        f"  ✅ Device: {device}, Vocab: {vocab_size:,}, Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    results_babi = {}

    models_to_train_babi = [
        ("Babi Connection Transformer", BabiConnectionTransformer),
        ("Babi Standard Transformer", BabiStandardTransformer),
        ("Babi ConnTrans + FFN", BabiConnTransWithFFN),
    ]

    sample_item_babi = val_dataset[0] if len(val_dataset) > 0 else None
    sample_batch_babi = None
    if sample_item_babi:
        sample_batch_babi = {k: v.unsqueeze(0) for k, v in sample_item_babi.items() if isinstance(v, torch.Tensor)}
        if 'answer_text' in sample_item_babi: sample_batch_babi['answer_text'] = [sample_item_babi['answer_text']]
        if 'token_type_ids' not in sample_batch_babi and 'input_ids' in sample_batch_babi:
            sample_batch_babi['token_type_ids'] = torch.zeros_like(sample_batch_babi['input_ids'])
        print(f"  📊 Prepared sample batch from bAbI validation set for trace.")

    for model_name, model_class in models_to_train_babi:
        print("\n" + "=" * 60 + f"\n▶️ bAbI EXPERIMENT: {model_name}" + "\n" + "=" * 60)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # 모델 초기화 시 필요한 모든 인자 전달
        # StandardTransformer와 ConnTrans 계열 모델이 받는 인자가 다를 수 있으므로,
        # **kwargs를 사용하거나, 각 모델에 맞게 명시적으로 전달해야 함.
        # 여기서는 모든 config 값을 일단 전달하고, 모델 내부에서 필요한 것만 사용하도록 함.
        model_instance = model_class(
            vocab_size=vocab_size,
            d_model=CFG_BABI["d_model"],
            num_slots=CFG_BABI["num_slots"],  # StandardTransformer는 이 인자를 사용하지 않음
            num_reasoning_steps=CFG_BABI["num_reasoning_steps"],  # StandardTransformer는 num_layers로 사용
            max_seq_len=CFG_BABI["max_seq_len"],
            connection_init_std=CFG_BABI.get("connection_init_std", 0.01),  # ConnTrans 계열용
            spectral_radius_limit=CFG_BABI.get("spectral_radius_limit", 0.95),  # ConnTrans 계열용
            # StandardTransformer 및 ConnTransWithFFN에 필요한 추가 인자
            num_heads=CFG_BABI.get("num_heads", 8),
            num_layers=CFG_BABI.get("num_transformer_layers", CFG_BABI["num_reasoning_steps"]),
            ffn_dim_multiplier=CFG_BABI.get("ffn_dim_multiplier", 4),
            dropout=CFG_BABI.get("dropout", 0.1)
        )

        acc_val = train_babi_model(model_instance, train_loader, val_loader, CFG_BABI, device, model_name)
        results_babi[model_name] = acc_val

        if isinstance(acc_val, float) and acc_val > 0.0 and sample_batch_babi:
            if hasattr(model_instance, 'C') and model_instance.C is not None:
                visualize_connection_matrix(model_instance, f"{model_name.replace(' ', '_')}_babi_C.png",
                                            f" ({model_name} bAbI)")
            if hasattr(model_instance, 'get_reasoning_trace'):
                analyze_reasoning_evolution(model_instance, sample_batch_babi,
                                            f"{model_name.replace(' ', '_')}_babi_evo.png", model_name)
        del model_instance
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print_comparison_results(results_babi, metric_name="Val Acc (bAbI)")

    print(f"\n💾 Saving bAbI Experimental Results...")
    babi_exp_results = {
        "experiment_type": "babi_conn_trans_comparison_refactored",
        "dataset_info": {"task_id": babi_task_id, "type_prefix": babi_type_prefix_to_load},
        "config_hyperparameters": CFG_BABI,
        "results_metric": "Validation Accuracy",
        "model_accuracies": results_babi,
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }
    babi_results_filename = f"babi_qa{babi_task_id}_{babi_type_prefix_to_load}_results_{babi_exp_results['timestamp']}.json"
    try:
        with open(babi_results_filename, "w") as f:
            json.dump(babi_exp_results, f, indent=2, cls=NpEncoder)
        print(f"  📄 bAbI Results: {babi_results_filename}")
    except Exception as e_json_babi:
        print(f"⚠️ Error saving bAbI JSON: {e_json_babi}")

    print(f"\n✨ bAbI Experiment Completed!")
    return results_babi


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    main_babi()

    # SQuAD 실험은 main_squad.py에서 별도 실행
    # print("\n\n" + "="*20 + " Moving to SQuAD Experiments " + "="*20)
    # from main_squad import main_squad
    # main_squad()

    print("\n\n" + "=" * 20 + " All Specified Experiments Finished " + "=" * 20)