# training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math  # math 임포트 추가 (F.softmax에서 사용)


# CONFIG를 직접 참조하는 대신, 함수 인자로 받도록 수정
# from configs.base_config import BASE_CONFIG # 더 이상 직접 참조 안 함

def train_model(model, train_loader, val_loader, config, device='cuda', model_name="Model"):
    model = model.to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    num_training_steps = len(train_loader) * config["max_epochs"]
    # warmup_steps가 num_training_steps보다 크지 않도록 보장
    actual_warmup_steps = min(config["warmup_steps"], num_training_steps // 10) if num_training_steps > 0 else config[
        "warmup_steps"]
    pct_start_val = float(actual_warmup_steps) / num_training_steps if num_training_steps > 0 else 0.1

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        total_steps=num_training_steps if num_training_steps > 0 else None,  # total_steps가 0이면 에러 발생 가능
        pct_start=pct_start_val,
    ) if num_training_steps > 0 else None  # num_training_steps가 0이면 스케줄러 사용 안 함

    best_val_span_acc = 0.0

    print(f"\n🚀 Training {model_name} for SQuAD on {device}...")
    print(f"   Total training steps: {num_training_steps if num_training_steps > 0 else 'N/A (no scheduler)'}")
    print("=" * 50)

    for epoch in range(config["max_epochs"]):
        model.train()
        total_train_loss = 0
        train_span_correct = 0
        train_total_samples = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)  # SQuADDataset에서 반환
            if token_type_ids is not None: token_type_ids = token_type_ids.to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            start_logits, end_logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            seq_len = input_ids.size(1)
            # SQuAD에서 start/end position이 CLS (인덱스 0)인 경우는 답변이 없거나 스팬 밖에 있는 경우.
            # 이 경우 loss 계산에서 제외하거나, CLS를 예측하도록 학습. 여기서는 CLS도 예측 대상.
            # ignore_index는 패딩된 토큰에 대한 것이 아니라, 타겟 레이블 자체를 무시할 때 사용.
            # SQuAD 1.1은 항상 답변이 있으므로, start/end_positions는 유효한 토큰 인덱스여야 함.
            # (SQuADDataset에서 CLS로 설정된 경우는 해당 스팬에 답변이 없다는 의미)

            # start_positions.clamp_(0, seq_len - 1) # 데이터셋에서 이미 처리되었을 것으로 가정
            # end_positions.clamp_(0, seq_len - 1)

            loss_fct = nn.CrossEntropyLoss()  # ignore_index 사용 안 함 (모든 위치에 대해 학습)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            loss = total_loss

            if hasattr(model, 'C') and model.C is not None and "connection_regularization" in config:
                c_reg = config["connection_regularization"] * torch.norm(model.C, 'fro') ** 2
                loss = loss + c_reg

            if hasattr(model, 'enforce_spectral_radius') and model.training:
                model.enforce_spectral_radius(config.get("spectral_radius_limit"))  # config에서 가져옴

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            if scheduler: scheduler.step()

            total_train_loss += loss.item() * input_ids.size(0)
            pred_start = torch.argmax(start_logits, dim=1)
            pred_end = torch.argmax(end_logits, dim=1)
            train_span_correct += ((pred_start == start_positions) & (pred_end == end_positions)).sum().item()
            train_total_samples += input_ids.size(0)

            if batch_idx > 0 and batch_idx % (len(train_loader) // 10 if len(train_loader) >= 10 else 1) == 0:
                current_lr = scheduler.get_last_lr()[0] if scheduler else config["learning_rate"]
                print(f"  E{epoch + 1} B{batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")

        avg_train_loss = total_train_loss / train_total_samples if train_total_samples > 0 else 0
        train_span_acc = train_span_correct / train_total_samples if train_total_samples > 0 else 0

        model.eval()
        val_span_correct = 0
        val_total_samples = 0
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids', None)
                if token_type_ids is not None: token_type_ids = token_type_ids.to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                start_logits, end_logits = model(input_ids, attention_mask=attention_mask,
                                                 token_type_ids=token_type_ids)

                # start_positions.clamp_(0, input_ids.size(1) - 1)
                # end_positions.clamp_(0, input_ids.size(1) - 1)

                loss_fct_val = nn.CrossEntropyLoss()
                start_loss_val = loss_fct_val(start_logits, start_positions)
                end_loss_val = loss_fct_val(end_logits, end_positions)
                total_loss_val = (start_loss_val + end_loss_val) / 2
                total_val_loss += total_loss_val.item() * input_ids.size(0)

                pred_start_val = torch.argmax(start_logits, dim=1)
                pred_end_val = torch.argmax(end_logits, dim=1)
                val_span_correct += ((pred_start_val == start_positions) & (pred_end_val == end_positions)).sum().item()
                val_total_samples += input_ids.size(0)

        avg_val_loss = total_val_loss / val_total_samples if val_total_samples > 0 else 0
        val_span_acc = val_span_correct / val_total_samples if val_total_samples > 0 else 0

        epoch_time = time.time() - start_time
        print(f"  Epoch {epoch + 1}/{config['max_epochs']} ({epoch_time:.1f}s)")
        print(f"    Train Loss: {avg_train_loss:.4f}, Train Span Acc: {train_span_acc:.4f}")
        print(f"    Val Loss:   {avg_val_loss:.4f}, Val Span Acc:   {val_span_acc:.4f}")

        if hasattr(model, 'get_connection_stats'):
            try:
                stats = model.get_connection_stats()
                print(f"    ConnStats: SR(I+C)={stats.get('spectral_radius_I_plus_C', float('nan')):.3f}, "
                      f"Frob(C)={stats.get('frobenius_norm', float('nan')):.3f}")
            except Exception as e_stat_print_epoch:
                print(f"Error printing conn stats epoch: {e_stat_print_epoch}")

        if val_span_acc > best_val_span_acc:
            best_val_span_acc = val_span_acc
            torch.save(model.state_dict(), f'best_squad_model_{model_name.replace(" ", "_")}.pt')
            print(f"    💾 New best SQuAD model saved (Val Span Acc: {best_val_span_acc:.4f})")
        print("-" * 30)

    print(f"✅ {model_name} SQuAD training completed. Best Val Span Acc: {best_val_span_acc:.4f}")
    return best_val_span_acc