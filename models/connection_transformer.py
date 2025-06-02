# models/connection_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConnectionTransformer(nn.Module):
    """
    Connection Transformer with Encoder-Decoder Architecture
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_slots=128,
                 bilinear_rank=32, max_reasoning_steps=6,
                 convergence_threshold=0.01, max_seq_len=512,
                 dropout=0.1, src_pad_token_id=0, tgt_pad_token_id=0,
                 num_decoder_layers=6, num_heads=8):
        super().__init__()
        
        self.d_model = d_model
        self.num_slots = num_slots
        self.bilinear_rank = bilinear_rank
        self.max_reasoning_steps = max_reasoning_steps
        self.convergence_threshold = convergence_threshold
        self.src_pad_token_id = src_pad_token_id
        self.tgt_pad_token_id = tgt_pad_token_id
        self.num_heads = num_heads
        
        # === ENCODER COMPONENTS ===
        # Source embeddings
        self.src_token_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_token_id)
        self.src_pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Fixed semantic slots (H) - orthogonal initialization
        self.register_buffer('H', self._create_orthogonal_slots(num_slots, d_model))
        
        # Bilinear connection matrices - will be orthogonally initialized
        self.W_source = nn.Parameter(torch.zeros(num_slots, num_slots, d_model, bilinear_rank))
        self.W_target = nn.Parameter(torch.zeros(num_slots, num_slots, bilinear_rank, d_model))
        
        # Encoder cross-attention projection matrices
        self.W_q_input = nn.Linear(d_model, d_model, bias=False)
        self.W_k_slots = nn.Linear(d_model, d_model, bias=False)
        self.W_v_input = nn.Linear(d_model, d_model, bias=False)
        
        # Layer normalization for reasoning steps
        self.reasoning_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(max_reasoning_steps)
        ])
        
        # === DECODER COMPONENTS ===
        # Target embeddings
        self.tgt_token_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_token_id)
        self.tgt_pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Orthogonal regularization parameters
        self.orthogonal_weight = 0.01
        
        self._init_parameters()
        
        # Report parameter count
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔹 Connection Transformer (Encoder-Decoder): {total_params:,} parameters")
        print(f"   - Encoder slots & bilinear: {self.W_source.numel() + self.W_target.numel():,}")
        print(f"   - Decoder layers: {num_decoder_layers}")

    def _create_orthogonal_slots(self, num_slots, d_model):
        """Create orthogonal semantic slots for independent semantic spaces."""
        if num_slots <= d_model:
            Q, _ = torch.qr(torch.randn(d_model, num_slots))
            H = Q.T  # (num_slots, d_model)
        else:
            H = torch.zeros(num_slots, d_model)
            for start in range(0, num_slots, d_model):
                end = min(start + d_model, num_slots)
                group_size = end - start
                Q, _ = torch.qr(torch.randn(d_model, group_size))
                H[start:end] = Q.T
        
        return H
    
    def _init_parameters(self):
        """Orthogonal initialization for all parameters"""
        # Source embeddings
        nn.init.normal_(self.src_token_embedding.weight, std=0.02)
        nn.init.normal_(self.src_pos_embedding.weight, std=0.02)
        
        # Target embeddings
        nn.init.normal_(self.tgt_token_embedding.weight, std=0.02)
        nn.init.normal_(self.tgt_pos_embedding.weight, std=0.02)
        
        # Padding token embeddings을 0으로 설정
        if self.src_pad_token_id is not None:
            with torch.no_grad():
                self.src_token_embedding.weight[self.src_pad_token_id].fill_(0)
        
        if self.tgt_pad_token_id is not None:
            with torch.no_grad():
                self.tgt_token_embedding.weight[self.tgt_pad_token_id].fill_(0)
        
        # Bilinear connections - Orthogonal initialization
        self._orthogonal_init_bilinear()
        
        # Encoder cross-attention projections
        for module in [self.W_q_input, self.W_k_slots, self.W_v_input]:
            nn.init.orthogonal_(module.weight)
        
        # Output projection
        nn.init.orthogonal_(self.output_projection.weight)
    
    def _orthogonal_init_bilinear(self):
        """Proper orthogonal initialization for bilinear matrices"""
        with torch.no_grad():
            for i in range(self.num_slots):
                for j in range(self.num_slots):
                    if i != j:  # Only non-self connections
                        # W_source[i,j]: [D, r] - orthogonal columns
                        W_src = self.W_source[i, j]  # [D, r]
                        if W_src.size(0) >= W_src.size(1):  # D >= r
                            Q, _ = torch.qr(torch.randn(W_src.size(0), W_src.size(0)))
                            self.W_source[i, j] = Q[:, :W_src.size(1)]
                        else:  # r > D (rare case)
                            nn.init.orthogonal_(W_src)
                        
                        # W_target[i,j]: [r, D] - orthogonal rows
                        W_tgt = self.W_target[i, j]  # [r, D]
                        if W_tgt.size(1) >= W_tgt.size(0):  # D >= r
                            Q, _ = torch.qr(torch.randn(W_tgt.size(1), W_tgt.size(1)))
                            self.W_target[i, j] = Q[:W_tgt.size(0), :]
                        else:  # r > D (rare case)
                            nn.init.orthogonal_(W_tgt)
    
    def bilinear_transform(self, H_state):
        """Orthogonal-regularized bilinear slot-to-slot influences"""
        batch_size, num_slots, d_model = H_state.shape
        device = H_state.device
        
        # Vectorized bilinear transformation
        H_expanded = H_state.unsqueeze(2).expand(batch_size, num_slots, num_slots, d_model)
        
        # First transformation: [B,N,N,D] × [N,N,D,r] -> [B,N,N,r]
        intermediate = torch.einsum('bijd,ijdr->bijr', H_expanded, self.W_source)
        
        # Second transformation: [B,N,N,r] × [N,N,r,D] -> [B,N,N,D]
        output = torch.einsum('bijr,ijrd->bijd', intermediate, self.W_target)
        
        # Remove self-connections
        mask = torch.eye(num_slots, device=device, dtype=torch.bool)
        output[:, mask] = 0
        
        # Sum over source slots
        influence = output.sum(dim=1)  # [B, N, D]
        
        return influence
    
    def encode(self, src_input_ids, src_attention_mask=None, return_reasoning_trace=False):
        """
        Encoder: Input tokens → Semantic slots with bilinear reasoning
        
        Args:
            src_input_ids: [batch_size, src_seq_len]
            src_attention_mask: [batch_size, src_seq_len]
            return_reasoning_trace: bool
            
        Returns:
            semantic_slots: [batch_size, num_slots, d_model]
            reasoning_info: dict (optional)
        """
        batch_size, src_seq_len = src_input_ids.shape
        device = src_input_ids.device
        
        # === STEP 1: SOURCE INPUT PROCESSING ===
        positions = torch.arange(src_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        X_src = self.src_token_embedding(src_input_ids) + self.src_pos_embedding(positions)
        X_src = self.dropout(X_src)
        
        # === STEP 2: INPUT → SEMANTIC SLOT COMPRESSION ===
        Q_input = self.W_q_input(X_src)
        K_slots = self.W_k_slots(self.H)
        V_input = self.W_v_input(X_src)
        
        A_compress = F.softmax(Q_input @ K_slots.T / math.sqrt(self.d_model), dim=-1)
        
        if src_attention_mask is not None:
            mask_expanded = src_attention_mask.unsqueeze(-1).float()
            A_compress = A_compress * mask_expanded
            A_compress = A_compress / (A_compress.sum(dim=1, keepdim=True) + 1e-8)
        
        IR_activation = A_compress.transpose(-1, -2) @ V_input
        H_state = self.H.unsqueeze(0).expand(batch_size, -1, -1) + IR_activation
        
        reasoning_trace = [H_state.clone()] if return_reasoning_trace else []
        
        # === STEP 3: BILINEAR REASONING ===
        actual_steps = 0
        final_change_magnitude = torch.zeros(batch_size, self.num_slots, device=device)
        
        for step in range(self.max_reasoning_steps):
            influence = self.bilinear_transform(H_state)
            
            step_update = F.relu(influence)
            H_state = H_state + step_update
            H_state = self.reasoning_norms[step](H_state)
            
            if return_reasoning_trace:
                reasoning_trace.append(H_state.clone())
            
            change_magnitude = torch.norm(step_update, dim=-1)
            final_change_magnitude = change_magnitude
            
            converged_mask = change_magnitude <= self.convergence_threshold
            all_converged = converged_mask.all()
            
            actual_steps = step + 1
            
            if all_converged:
                break
        
        if return_reasoning_trace:
            reasoning_info = {
                'actual_steps': actual_steps,
                'reasoning_trace': reasoning_trace,
                'final_change_magnitude': final_change_magnitude
            }
            return H_state, reasoning_info
        else:
            return H_state
    
    def decode(self, tgt_input_ids, semantic_slots, tgt_attention_mask=None):
        """
        Decoder: Target tokens generation with semantic slots memory
        
        Args:
            tgt_input_ids: [batch_size, tgt_seq_len]
            semantic_slots: [batch_size, num_slots, d_model] (from encoder)
            tgt_attention_mask: [batch_size, tgt_seq_len]
            
        Returns:
            logits: [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        batch_size, tgt_seq_len = tgt_input_ids.shape
        device = tgt_input_ids.device
        
        # Target embeddings
        positions = torch.arange(tgt_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        tgt_embeddings = self.tgt_token_embedding(tgt_input_ids) + self.tgt_pos_embedding(positions)
        tgt_embeddings = self.dropout(tgt_embeddings)
        
        # Create causal mask for decoder self-attention
        causal_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=device), diagonal=1).bool()
        
        # Create target key padding mask
        if tgt_attention_mask is not None:
            tgt_key_padding_mask = (tgt_attention_mask == 0)
        else:
            tgt_key_padding_mask = None
        
        # Apply transformer decoder layers
        decoder_output = self.decoder(
            tgt=tgt_embeddings,
            memory=semantic_slots,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Output projection
        output = self.output_norm(decoder_output)
        output = self.dropout(output)
        logits = self.output_projection(output)
        
        return logits
    
    def forward(self, src_input_ids, tgt_input_ids, src_attention_mask=None, 
                tgt_attention_mask=None, return_reasoning_trace=False):
        """
        Full forward pass: Encoder + Decoder
        
        Args:
            src_input_ids: [batch_size, src_seq_len]
            tgt_input_ids: [batch_size, tgt_seq_len]
            src_attention_mask: [batch_size, src_seq_len]
            tgt_attention_mask: [batch_size, tgt_seq_len]
            return_reasoning_trace: bool
            
        Returns:
            logits: [batch_size, tgt_seq_len, tgt_vocab_size]
            reasoning_info: dict (optional)
        """
        # Encode source
        if return_reasoning_trace:
            semantic_slots, reasoning_info = self.encode(
                src_input_ids, src_attention_mask, return_reasoning_trace=True
            )
        else:
            semantic_slots = self.encode(src_input_ids, src_attention_mask)
        
        # Decode target
        logits = self.decode(tgt_input_ids, semantic_slots, tgt_attention_mask)
        
        if return_reasoning_trace:
            return logits, reasoning_info
        else:
            return logits
    
    def orthogonal_regularization_loss(self):
        """Orthogonal regularization loss (same as before)"""
        device = self.W_source.device
        
        mask = torch.eye(self.num_slots, device=device, dtype=torch.bool)
        
        # W_source orthogonality
        gram_source = torch.einsum('ijdr,ijdq->ijrq', self.W_source, self.W_source)
        gram_source_valid = gram_source[~mask]
        
        identity = torch.eye(self.bilinear_rank, device=device, dtype=gram_source.dtype)
        identity_batch = identity.unsqueeze(0).expand(gram_source_valid.size(0), -1, -1)
        
        source_loss = F.mse_loss(gram_source_valid, identity_batch)
        
        # W_target orthogonality
        gram_target = torch.einsum('ijrd,ijqd->ijrq', self.W_target, self.W_target)
        gram_target_valid = gram_target[~mask]
        
        target_loss = F.mse_loss(gram_target_valid, identity_batch)
        
        return (source_loss + target_loss) / 2
    
    def get_connection_analysis(self):
        """Enhanced analysis including orthogonality quality"""
        with torch.no_grad():
            connection_magnitudes = torch.zeros(self.num_slots, self.num_slots)
            orthogonality_errors = []
            
            for i in range(self.num_slots):
                for j in range(self.num_slots):
                    if i != j:
                        # Connection strength
                        combined = self.W_source[i, j] @ self.W_target[i, j]
                        magnitude = torch.norm(combined, 'fro').item()
                        connection_magnitudes[i, j] = magnitude
                        
                        # Orthogonality quality for W_source
                        W_src = self.W_source[i, j]
                        if W_src.size(0) >= W_src.size(1):
                            gram = W_src.T @ W_src
                            identity = torch.eye(W_src.size(1), device=W_src.device)
                            error = torch.norm(gram - identity, 'fro').item()
                            orthogonality_errors.append(error)
                        
                        # Orthogonality quality for W_target
                        W_tgt = self.W_target[i, j]
                        if W_tgt.size(1) >= W_tgt.size(0):
                            gram = W_tgt @ W_tgt.T
                            identity = torch.eye(W_tgt.size(0), device=W_tgt.device)
                            error = torch.norm(gram - identity, 'fro').item()
                            orthogonality_errors.append(error)
            
            avg_orthogonality_error = sum(orthogonality_errors) / len(orthogonality_errors) if orthogonality_errors else 0.0
            
            return {
                'connection_matrix': connection_magnitudes,
                'sparsity_ratio': (connection_magnitudes < 0.01).float().mean().item(),
                'max_connection': connection_magnitudes.max().item(),
                'mean_connection': connection_magnitudes.mean().item(),
                'orthogonality_error': avg_orthogonality_error,
                'orthogonality_quality': 1.0 / (1.0 + avg_orthogonality_error)
            }
    
    def reasoning_cost_loss(self, actual_steps, target_steps=4, weight=0.001):
        """Regularization loss for reasoning efficiency"""
        if isinstance(actual_steps, int):
            actual_steps = torch.tensor(actual_steps, dtype=torch.float32, device=next(self.parameters()).device)
        target = torch.full_like(actual_steps, target_steps, dtype=torch.float32)
        return weight * F.mse_loss(actual_steps.float(), target)

    def load_pretrained_weights(self, model_name="google-t5/t5-base"):
        """T5 pre-trained weights 로딩 (d_model 크기 안전 처리)"""
        try:
            from transformers import T5Model
            pretrained = T5Model.from_pretrained(model_name)
            
            # 1. Token embeddings (d_model 차원 처리)
            pretrained_embed = pretrained.shared.weight.data  # [vocab_size, pretrained_d_model]
            current_vocab_size = self.src_token_embedding.weight.size(0)
            current_d_model = self.src_token_embedding.weight.size(1)
            pretrained_vocab_size = pretrained_embed.size(0)
            pretrained_d_model = pretrained_embed.size(1)
            
            print(f"🔍 Dimensions: current=({current_vocab_size}, {current_d_model}), pretrained=({pretrained_vocab_size}, {pretrained_d_model})")
            
            if current_d_model == pretrained_d_model:
                # d_model이 같으면 vocab_size만 맞춰서 복사
                min_vocab_size = min(current_vocab_size, pretrained_vocab_size)
                self.src_token_embedding.weight.data[:min_vocab_size] = pretrained_embed[:min_vocab_size].clone()
                self.tgt_token_embedding.weight.data[:min_vocab_size] = pretrained_embed[:min_vocab_size].clone()
                print(f"✅ Token embeddings: {min_vocab_size} tokens, d_model={current_d_model}")
            else:
                # d_model이 다르면 차원 맞춰서 복사 (작은 쪽까지만)
                min_vocab_size = min(current_vocab_size, pretrained_vocab_size)
                min_d_model = min(current_d_model, pretrained_d_model)
                self.src_token_embedding.weight.data[:min_vocab_size, :min_d_model] = pretrained_embed[:min_vocab_size, :min_d_model].clone()
                self.tgt_token_embedding.weight.data[:min_vocab_size, :min_d_model] = pretrained_embed[:min_vocab_size, :min_d_model].clone()
                print(f"✅ Token embeddings: {min_vocab_size} tokens, {min_d_model}/{current_d_model} dimensions")
            
            # 2. Position embeddings (T5 스타일, d_model 크기 맞춤)
            max_pos = min(self.src_pos_embedding.weight.size(0), 512)
            pos_init = torch.randn(max_pos, current_d_model) * 0.02  # 우리 d_model 크기로
            self.src_pos_embedding.weight.data[:max_pos] = pos_init
            self.tgt_pos_embedding.weight.data[:max_pos] = pos_init
            print(f"✅ Position embeddings: {max_pos} positions, d_model={current_d_model}")
            
            # 3. Output projection (d_model 차원 처리)
            if hasattr(pretrained, 'lm_head'):
                pretrained_proj = pretrained.lm_head.weight.data  # [vocab_size, pretrained_d_model]
                current_vocab_out = self.output_projection.weight.size(0)
                current_d_model_out = self.output_projection.weight.size(1)
                pretrained_vocab_out = pretrained_proj.size(0)
                pretrained_d_model_out = pretrained_proj.size(1)
                
                if current_d_model_out == pretrained_d_model_out:
                    # d_model이 같으면 vocab만 맞춰서
                    min_vocab_out = min(current_vocab_out, pretrained_vocab_out)
                    self.output_projection.weight.data[:min_vocab_out] = pretrained_proj[:min_vocab_out].clone()
                    print(f"✅ Output projection: {min_vocab_out} tokens, d_model={current_d_model_out}")
                else:
                    # d_model이 다르면 둘 다 맞춰서
                    min_vocab_out = min(current_vocab_out, pretrained_vocab_out)
                    min_d_model_out = min(current_d_model_out, pretrained_d_model_out)
                    self.output_projection.weight.data[:min_vocab_out, :min_d_model_out] = pretrained_proj[:min_vocab_out, :min_d_model_out].clone()
                    print(f"✅ Output projection: {min_vocab_out} tokens, {min_d_model_out}/{current_d_model_out} dimensions")
            
            print(f"🎯 Pre-trained initialization from {model_name} completed")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to load pre-trained weights: {e}")
            return False