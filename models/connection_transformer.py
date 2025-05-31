# models/connection_transformer.py - 수정된 버전
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConnectionTransformer(nn.Module):
    """
    Connection Transformer with bilinear connections and adaptive reasoning.
    수정사항:
    1. pad_token_id 처리 추가
    2. bilinear_transform 최적화
    3. convergence 체크 로직 개선
    4. 메모리 효율성 향상
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
        
        # Fixed semantic slots (H) - never updated, orthogonal initialization
        self.register_buffer('H', self._create_orthogonal_slots(num_slots, d_model))
        
        # Bilinear connection matrices - primary learnable parameters
        self.W_source = nn.Parameter(torch.normal(0, 0.02,
                                    size=(num_slots, num_slots, d_model, bilinear_rank)))
        self.W_target = nn.Parameter(torch.normal(0, 0.02,
                                    size=(num_slots, num_slots, bilinear_rank, d_model)))
        
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
        
        self._init_parameters()
        
        # Report parameter count
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔹 Connection Transformer: {total_params:,} parameters")
        print(f"   - Bilinear connections: {self.W_source.numel() + self.W_target.numel():,}")
        print(f"   - Cross-attention: {sum(p.numel() for p in [self.W_q_input, self.W_k_slots, self.W_v_input, self.W_q_output, self.W_k_final, self.W_v_final]):,}")
        
    def _create_orthogonal_slots(self, num_slots, d_model):
        """Create orthogonal semantic slots for independent semantic spaces."""
        if num_slots <= d_model:
            # Perfect orthogonality when num_slots ≤ d_model
            Q, _ = torch.qr(torch.randn(d_model, num_slots))
            H = Q.T  # (num_slots, d_model)
        else:
            # Partial orthogonality for num_slots > d_model
            H = torch.zeros(num_slots, d_model)
            for start in range(0, num_slots, d_model):
                end = min(start + d_model, num_slots)
                group_size = end - start
                Q, _ = torch.qr(torch.randn(d_model, group_size))
                H[start:end] = Q.T
        
        return H
    
    def _init_parameters(self):
        """Initialize parameters according to specification"""
        # Token embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        
        # Padding token embedding을 0으로 설정
        if self.pad_token_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.pad_token_id].fill_(0)
        
        # Bilinear connections - Xavier initialization
        fan_in = self.d_model
        fan_out = self.bilinear_rank
        std = math.sqrt(2.0 / (fan_in + fan_out))
        nn.init.normal_(self.W_source, std=std)
        nn.init.normal_(self.W_target, std=std)
        
        # Cross-attention projections
        for module in [self.W_q_input, self.W_k_slots, self.W_v_input,
                      self.W_q_output, self.W_k_final, self.W_v_final]:
            nn.init.xavier_uniform_(module.weight)
        
        # Output projection
        nn.init.normal_(self.output_projection.weight, std=0.02)
    
    def bilinear_transform(self, H_state):
        """
        Compute bilinear slot-to-slot influences - 최적화된 버전.
        
        Args:
            H_state: [batch_size, num_slots, d_model]
        
        Returns:
            influence: [batch_size, num_slots, d_model]
        """
        batch_size, num_slots, d_model = H_state.shape
        device = H_state.device
        
        # 벡터화된 bilinear transformation (메모리 효율적)
        influence = torch.zeros_like(H_state)
        
        # 모든 연결을 한 번에 계산 (self-connection 제외)
        for i in range(num_slots):
            # Source slot i의 영향을 모든 다른 slot들에게 전파
            source_state = H_state[:, i, :]  # [B, D]
            
            for j in range(num_slots):
                if i != j:  # Skip self-connections
                    # Bilinear transformation: source -> intermediate -> target
                    intermediate = source_state @ self.W_source[i, j]  # [B, r]
                    transformed = intermediate @ self.W_target[i, j]   # [B, D]
                    influence[:, j, :] += transformed
        
        return influence
    
    def forward(self, input_ids, attention_mask=None, return_reasoning_trace=False):
        """
        Forward pass with adaptive bilinear reasoning.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_reasoning_trace: bool
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            reasoning_info: dict (optional)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # === STEP 1: INPUT PROCESSING ===
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        X_input = self.token_embedding(input_ids) + self.pos_embedding(positions)
        X_input = self.dropout(X_input)
        
        # === STEP 2: INPUT → SEMANTIC SLOT COMPRESSION ===
        Q_input = self.W_q_input(X_input)    # [B, S, D]
        K_slots = self.W_k_slots(self.H)     # [N, D]
        V_input = self.W_v_input(X_input)    # [B, S, D]
        
        # Cross-attention compression
        A_compress = F.softmax(Q_input @ K_slots.T / math.sqrt(self.d_model), dim=-1)
        
        # Attention mask 적용 (padding 토큰 무시)
        if attention_mask is not None:
            # attention_mask: [B, S], True for valid tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
            A_compress = A_compress * mask_expanded
            # Re-normalize
            A_compress = A_compress / (A_compress.sum(dim=1, keepdim=True) + 1e-8)
        
        IR_activation = A_compress.transpose(-1, -2) @ V_input  # [B, N, D]
        
        # Initialize reasoning state
        H_state = self.H.unsqueeze(0).expand(batch_size, -1, -1) + IR_activation
        
        # Store reasoning trace if requested
        reasoning_trace = [H_state.clone()] if return_reasoning_trace else []
        
        # === STEP 3: ADAPTIVE BILINEAR REASONING ===
        actual_steps = 0
        final_change_magnitude = torch.zeros(batch_size, self.num_slots, device=device)
        
        for step in range(self.max_reasoning_steps):
            # Compute bilinear slot-to-slot influences
            influence = self.bilinear_transform(H_state)
            
            # Apply ReLU activation (neuronal firing threshold)
            step_update = F.relu(influence)
            
            # Update slot states
            H_state = H_state + step_update
            
            # Apply layer normalization
            H_state = self.reasoning_norms[step](H_state)
            
            # Store reasoning trace if requested
            if return_reasoning_trace:
                reasoning_trace.append(H_state.clone())
            
            # Adaptive termination criterion
            change_magnitude = torch.norm(step_update, dim=-1)  # [B, N]
            final_change_magnitude = change_magnitude
            
            # Check convergence per sample and per slot
            converged_mask = change_magnitude <= self.convergence_threshold  # [B, N]
            all_converged = converged_mask.all()  # True if all slots in all samples converged
            
            actual_steps = step + 1
            
            # Global convergence check
            if all_converged:
                break
        
        # === STEP 4: SEMANTIC SLOT → OUTPUT EXPANSION ===
        Q_output = self.W_q_output(X_input)    # [B, S, D]
        K_final = self.W_k_final(H_state)      # [B, N, D]
        V_final = self.W_v_final(H_state)      # [B, N, D]
        
        # Cross-attention expansion
        A_expand = F.softmax(Q_output @ K_final.transpose(-1, -2) / math.sqrt(self.d_model), dim=-1)
        Y_output = A_expand @ V_final  # [B, S, D]
        
        # Apply dropout
        Y_output = self.dropout(Y_output)
        
        # === STEP 5: VOCABULARY PROJECTION ===
        logits = self.output_projection(Y_output)  # [B, S, V]
        
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
        """Analyze bilinear connection patterns"""
        connection_magnitudes = torch.zeros(self.num_slots, self.num_slots)
        
        with torch.no_grad():
            for i in range(self.num_slots):
                for j in range(self.num_slots):
                    if i != j:
                        # Compute effective connection strength
                        combined = self.W_source[i, j] @ self.W_target[i, j]  # [D, D]
                        magnitude = torch.norm(combined, 'fro').item()
                        connection_magnitudes[i, j] = magnitude
        
        return {
            'connection_matrix': connection_magnitudes,
            'sparsity_ratio': (connection_magnitudes < 0.01).float().mean().item(),
            'max_connection': connection_magnitudes.max().item(),
            'mean_connection': connection_magnitudes.mean().item()
        }
    
    def reasoning_cost_loss(self, actual_steps, target_steps=4, weight=0.001):
        """Regularization loss for reasoning efficiency"""
        if isinstance(actual_steps, int):
            actual_steps = torch.tensor(actual_steps, dtype=torch.float32, device=next(self.parameters()).device)
        target = torch.full_like(actual_steps, target_steps, dtype=torch.float32)
        return weight * F.mse_loss(actual_steps.float(), target)