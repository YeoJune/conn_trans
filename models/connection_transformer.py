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
        self.W_source = nn.Parameter(torch.zeros(num_slots, num_slots, bilinear_rank))
        self.W_target = nn.Parameter(torch.zeros(num_slots, num_slots, bilinear_rank))
        
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
        """간소화된 orthogonal 초기화"""
        with torch.no_grad():
            for i in range(self.num_slots):
                for j in range(self.num_slots):
                    if i != j:  # Only non-self connections
                        # 단순한 orthogonal 초기화
                        nn.init.orthogonal_(self.W_source[i, j].unsqueeze(0))
                        nn.init.orthogonal_(self.W_target[i, j].unsqueeze(0))
    
    def bilinear_transform(self, H_state):
        """청크 단위로 einsum 처리"""
        # 연결 강도 계산: [N, N, r] * [N, N, r] -> [N, N]
        connection_matrix = torch.sum(self.W_source * self.W_target, dim=-1)  # [N, N]
        
        # 자기 연결 제거
        connection_matrix.fill_diagonal_(0.0)
        
        batch_size, num_slots, d_model = H_state.shape
        
        # einsum 청크 크기 결정 (제한: 128 미만)
        chunk_size = min(127, num_slots)  # einsum 제한 고려
        
        if num_slots <= 127:
            # 한 번에 처리 가능
            influence = torch.einsum('bid,ij->bjd', H_state, connection_matrix)
        else:
            # 청크로 나누어 처리
            influence = torch.zeros_like(H_state)
            
            for j_start in range(0, num_slots, chunk_size):
                j_end = min(j_start + chunk_size, num_slots)
                
                # j_start:j_end 슬롯들이 받는 영향 계산
                # 모든 i 슬롯으로부터의 기여 합산
                chunk_influence = torch.einsum(
                    'bid,ij->bjd', 
                    H_state,  # [B, N, D] - 모든 소스 슬롯
                    connection_matrix[:, j_start:j_end]  # [N, chunk_size] - 타겟 청크로의 연결
                )
                
                influence[:, j_start:j_end, :] = chunk_influence
        
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
        """간소화된 orthogonal regularization"""
        device = self.W_source.device
        
        # 자기 연결 제외 마스크
        mask = torch.eye(self.num_slots, device=device, dtype=torch.bool)
        
        # W_source orthogonality (벡터 간 직교성)
        W_source_valid = self.W_source[~mask]  # [N*(N-1), r]
        gram_source = W_source_valid @ W_source_valid.T
        identity_source = torch.eye(W_source_valid.size(0), device=device)
        source_loss = F.mse_loss(gram_source, identity_source)
        
        # W_target orthogonality
        W_target_valid = self.W_target[~mask]  # [N*(N-1), r]
        gram_target = W_target_valid @ W_target_valid.T
        identity_target = torch.eye(W_target_valid.size(0), device=device)
        target_loss = F.mse_loss(gram_target, identity_target)
        
        return (source_loss + target_loss) / 2
    
    def get_connection_analysis(self):
        """간소화된 bilinear에 맞춘 연결 분석"""
        with torch.no_grad():
            # 연결 강도 계산: [N, N, r] * [N, N, r] -> [N, N]
            connection_magnitudes = torch.sum(self.W_source * self.W_target, dim=-1)
            
            # 자기 연결 제거 (대각선 0으로)
            mask = torch.eye(self.num_slots, device=connection_magnitudes.device, dtype=torch.bool)
            connection_magnitudes = connection_magnitudes.masked_fill(mask, 0.0)
            
            # Orthogonality 분석 (벡터들 간의 직교성)
            orthogonality_errors = []
            
            # W_source 벡터들의 직교성 분석
            W_source_valid = self.W_source[~mask]  # [N*(N-1), r] - 자기 연결 제외
            if W_source_valid.size(0) > 0:
                # 벡터들 간의 내적 행렬 계산
                gram_source = W_source_valid @ W_source_valid.T  # [N*(N-1), N*(N-1)]
                # 대각선은 1이어야 하고, 비대각선은 0에 가까워야 함
                gram_source.fill_diagonal_(0)  # 대각선 제거하고 비대각선 요소만 확인
                source_orth_error = torch.norm(gram_source, 'fro').item()
                orthogonality_errors.append(source_orth_error)
            
            # W_target 벡터들의 직교성 분석
            W_target_valid = self.W_target[~mask]  # [N*(N-1), r]
            if W_target_valid.size(0) > 0:
                gram_target = W_target_valid @ W_target_valid.T
                gram_target.fill_diagonal_(0)
                target_orth_error = torch.norm(gram_target, 'fro').item()
                orthogonality_errors.append(target_orth_error)
            
            # 평균 직교성 오차
            avg_orthogonality_error = sum(orthogonality_errors) / len(orthogonality_errors) if orthogonality_errors else 0.0
            
            # 연결 패턴 분석
            abs_connections = torch.abs(connection_magnitudes)
            threshold = 0.01
            
            # 양수/음수 연결 분석
            positive_connections = (connection_magnitudes > threshold).sum().item()
            negative_connections = (connection_magnitudes < -threshold).sum().item()
            total_possible = self.num_slots * (self.num_slots - 1)  # 자기 연결 제외
            
            # 연결 강도 통계
            non_zero_connections = connection_magnitudes[connection_magnitudes != 0]
            
            return {
                'connection_matrix': connection_magnitudes,
                'sparsity_ratio': (abs_connections < threshold).float().mean().item(),
                'max_connection': abs_connections.max().item(),
                'min_connection': abs_connections.min().item(),
                'mean_connection': abs_connections.mean().item(),
                'std_connection': abs_connections.std().item(),
                'median_connection': abs_connections.median().item(),
                
                # 연결 패턴
                'positive_connections': positive_connections,
                'negative_connections': negative_connections,
                'total_possible_connections': total_possible,
                'active_connection_ratio': (positive_connections + negative_connections) / total_possible,
                
                # 직교성 품질
                'orthogonality_error': avg_orthogonality_error,
                'orthogonality_quality': 1.0 / (1.0 + avg_orthogonality_error),
                
                # 연결 강도 분포
                'connection_range': abs_connections.max().item() - abs_connections.min().item(),
                'connection_entropy': self._calculate_connection_entropy(abs_connections),
                
                # 원시 데이터 (디버깅용)
                'raw_source_weights': self.W_source.clone(),
                'raw_target_weights': self.W_target.clone()
            }

    def _calculate_connection_entropy(self, connection_strengths):
        """연결 강도의 엔트로피 계산 (다양성 측정)"""
        try:
            # 연결 강도를 확률 분포로 변환
            abs_strengths = torch.abs(connection_strengths).flatten()
            abs_strengths = abs_strengths[abs_strengths > 1e-8]  # 0에 가까운 값 제거
            
            if len(abs_strengths) == 0:
                return 0.0
            
            # 정규화하여 확률 분포로 만들기
            probs = abs_strengths / abs_strengths.sum()
            
            # 엔트로피 계산: H = -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
            
            return entropy
        except:
            return 0.0
    
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