# models/connection_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConnectionTransformer(nn.Module):
    """
    Connection Transformer with Orthogonal Regularized Bilinear Connections
    """
    
    def __init__(self, vocab_size, d_model=256, num_slots=128,
                 bilinear_rank=32, max_reasoning_steps=6,
                 convergence_threshold=0.01, max_seq_len=512,
                 dropout=0.1, pad_token_id=0):
        super().__init__()
        
        self.d_model = d_model
        self.num_slots = num_slots
        self.bilinear_rank = bilinear_rank
        self.max_reasoning_steps = max_reasoning_steps
        self.convergence_threshold = convergence_threshold
        self.pad_token_id = pad_token_id
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Fixed semantic slots (H) - orthogonal initialization
        self.register_buffer('H', self._create_orthogonal_slots(num_slots, d_model))
        
        # Bilinear connection matrices - will be orthogonally initialized
        self.W_source = nn.Parameter(torch.zeros(num_slots, num_slots, d_model, bilinear_rank))
        self.W_target = nn.Parameter(torch.zeros(num_slots, num_slots, bilinear_rank, d_model))
        
        # Cross-attention projection matrices
        self.W_q_input = nn.Linear(d_model, d_model, bias=False)
        self.W_k_slots = nn.Linear(d_model, d_model, bias=False)
        self.W_v_input = nn.Linear(d_model, d_model, bias=False)
        
        self.W_q_output = nn.Linear(d_model, d_model, bias=False)
        self.W_k_final = nn.Linear(d_model, d_model, bias=False)
        self.W_v_final = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Layer normalization for reasoning steps
        self.reasoning_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(max_reasoning_steps)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Orthogonal regularization parameters
        self.orthogonal_weight = 0.01  # Default weight, will be overridden by config
        
        self._init_parameters()
        
        # Report parameter count
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔹 Connection Transformer (Orthogonal): {total_params:,} parameters")
        print(f"   - Bilinear connections: {self.W_source.numel() + self.W_target.numel():,}")
        
        cross_attention_params = sum(
            p.numel() for layer in [
                self.W_q_input, self.W_k_slots, self.W_v_input,
                self.W_q_output, self.W_k_final, self.W_v_final
            ] for p in layer.parameters()
        )
        print(f"   - Cross-attention: {cross_attention_params:,}")

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
        """ Orthogonal initialization for all parameters"""
        # Token embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        
        # Padding token embedding을 0으로 설정
        if self.pad_token_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.pad_token_id].fill_(0)
        
        # Bilinear connections - Orthogonal initialization
        self._orthogonal_init_bilinear()
        
        # Cross-attention projections - orthogonal initialization
        for module in [self.W_q_input, self.W_k_slots, self.W_v_input,
                      self.W_q_output, self.W_k_final, self.W_v_final]:
            nn.init.orthogonal_(module.weight)
        
        # Output projection
        nn.init.orthogonal_(self.output_projection.weight)
    
    def _orthogonal_init_bilinear(self):
        """ Proper orthogonal initialization for bilinear matrices"""
        with torch.no_grad():
            for i in range(self.num_slots):
                for j in range(self.num_slots):
                    if i != j:  # Only non-self connections
                        # W_source[i,j]: [D, r] - orthogonal columns
                        W_src = self.W_source[i, j]  # [D, r]
                        if W_src.size(0) >= W_src.size(1):  # D >= r
                            # Generate orthogonal matrix and take first r columns
                            Q, _ = torch.qr(torch.randn(W_src.size(0), W_src.size(0)))
                            self.W_source[i, j] = Q[:, :W_src.size(1)]
                        else:  # r > D (rare case)
                            nn.init.orthogonal_(W_src)
                        
                        # W_target[i,j]: [r, D] - orthogonal rows
                        W_tgt = self.W_target[i, j]  # [r, D]
                        if W_tgt.size(1) >= W_tgt.size(0):  # D >= r
                            # Generate orthogonal matrix and take first r rows
                            Q, _ = torch.qr(torch.randn(W_tgt.size(1), W_tgt.size(1)))
                            self.W_target[i, j] = Q[:W_tgt.size(0), :]
                        else:  # r > D (rare case)
                            nn.init.orthogonal_(W_tgt)
    
    def bilinear_transform(self, H_state):
        """
        Orthogonal-regularized bilinear slot-to-slot influences
        """
        batch_size, num_slots, d_model = H_state.shape
        device = H_state.device
        
        # Vectorized bilinear transformation (same as before)
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
    
    def orthogonal_regularization_loss(self):
        """
        Orthogonal regularization loss computation
        
        Enforces: W^T @ W ≈ I for proper information preservation
        """
        total_loss = torch.tensor(0.0, device=self.W_source.device)
        count = 0
        
        # W_source orthogonality: W_source[i,j]^T @ W_source[i,j] = I
        for i in range(self.num_slots):
            for j in range(self.num_slots):
                if i != j:  # Only non-self connections
                    W_src = self.W_source[i, j]  # [D, r]
                    
                    if W_src.size(0) >= W_src.size(1):  # D >= r (normal case)
                        # Enforce column orthogonality: W^T @ W = I_r
                        gram_matrix = W_src.T @ W_src  # [r, r]
                        identity = torch.eye(W_src.size(1), device=W_src.device, dtype=W_src.dtype)
                        total_loss += F.mse_loss(gram_matrix, identity)
                        count += 1
        
        # W_target orthogonality: W_target[i,j] @ W_target[i,j]^T = I
        for i in range(self.num_slots):
            for j in range(self.num_slots):
                if i != j:
                    W_tgt = self.W_target[i, j]  # [r, D]
                    
                    if W_tgt.size(1) >= W_tgt.size(0):  # D >= r (normal case)
                        # Enforce row orthogonality: W @ W^T = I_r
                        gram_matrix = W_tgt @ W_tgt.T  # [r, r]
                        identity = torch.eye(W_tgt.size(0), device=W_tgt.device, dtype=W_tgt.dtype)
                        total_loss += F.mse_loss(gram_matrix, identity)
                        count += 1
        
        return total_loss / count if count > 0 else total_loss
    
    def forward(self, input_ids, attention_mask=None, return_reasoning_trace=False):
        """Forward pass with orthogonal-regularized reasoning (unchanged logic)"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # === STEP 1: INPUT PROCESSING ===
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        X_input = self.token_embedding(input_ids) + self.pos_embedding(positions)
        X_input = self.dropout(X_input)
        
        # === STEP 2: INPUT → SEMANTIC SLOT COMPRESSION ===
        Q_input = self.W_q_input(X_input)
        K_slots = self.W_k_slots(self.H)
        V_input = self.W_v_input(X_input)
        
        A_compress = F.softmax(Q_input @ K_slots.T / math.sqrt(self.d_model), dim=-1)
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            A_compress = A_compress * mask_expanded
            A_compress = A_compress / (A_compress.sum(dim=1, keepdim=True) + 1e-8)
        
        IR_activation = A_compress.transpose(-1, -2) @ V_input
        H_state = self.H.unsqueeze(0).expand(batch_size, -1, -1) + IR_activation
        
        reasoning_trace = [H_state.clone()] if return_reasoning_trace else []
        
        # === STEP 3: ADAPTIVE BILINEAR REASONING (with orthogonal regularization) ===
        actual_steps = 0
        final_change_magnitude = torch.zeros(batch_size, self.num_slots, device=device)
        
        for step in range(self.max_reasoning_steps):
            # Orthogonal-regularized bilinear transformation
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
        
        # === STEP 4: SEMANTIC SLOT → OUTPUT EXPANSION ===
        Q_output = self.W_q_output(X_input)
        K_final = self.W_k_final(H_state)
        V_final = self.W_v_final(H_state)
        
        A_expand = F.softmax(Q_output @ K_final.transpose(-1, -2) / math.sqrt(self.d_model), dim=-1)
        Y_output = A_expand @ V_final
        Y_output = self.dropout(Y_output)
        
        # === STEP 5: VOCABULARY PROJECTION ===
        logits = self.output_projection(Y_output)
        
        if return_reasoning_trace:
            reasoning_info = {
                'actual_steps': actual_steps,
                'reasoning_trace': reasoning_trace,
                'final_change_magnitude': final_change_magnitude
            }
            return logits, reasoning_info
        else:
            return logits
    
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
        """Regularization loss for reasoning efficiency (unchanged)"""
        if isinstance(actual_steps, int):
            actual_steps = torch.tensor(actual_steps, dtype=torch.float32, device=next(self.parameters()).device)
        target = torch.full_like(actual_steps, target_steps, dtype=torch.float32)
        return weight * F.mse_loss(actual_steps.float(), target)