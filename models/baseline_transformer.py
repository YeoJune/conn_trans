# models/baseline_transformer.py - 수정된 버전
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaselineTransformer(nn.Module):
    """
    Standard Transformer baseline for fair comparison.
    수정사항:
    1. pad_token_id 처리 추가
    2. attention mask 처리 개선
    3. 초기화 방식 개선
    4. 메모리 효율성 향상
    """
    
    def __init__(self, vocab_size, d_model=256, num_layers=6, 
                 num_heads=8, ffn_multiplier=4, dropout=0.1, 
                 max_seq_len=512, pad_token_id=0):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        
        # Token embeddings (same as Connection Transformer)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ffn_multiplier,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm like modern transformers
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_parameters()
        
        # Report parameter count
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔶 Baseline Transformer: {total_params:,} parameters")
        print(f"   - Transformer layers: {num_layers} x {d_model}d x {ffn_multiplier}ffn")
        
    def _init_parameters(self):
        """Initialize parameters to match Connection Transformer"""
        # Token embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        
        # Padding token embedding을 0으로 설정
        if self.pad_token_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.pad_token_id].fill_(0)
        
        # Output projection  
        nn.init.normal_(self.output_projection.weight, std=0.02)
        
        # Transformer layers는 PyTorch 기본 초기화 사용
        
    def forward(self, input_ids, attention_mask=None, return_reasoning_trace=False):
        """
        Forward pass matching Connection Transformer interface.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] 
            return_reasoning_trace: bool (for compatibility)
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            reasoning_info: dict (for compatibility with Connection Transformer)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token + positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Create attention mask for transformer
        # PyTorch transformer expects True for tokens to IGNORE
        if attention_mask is not None:
            # input attention_mask: [B, S], True for valid tokens
            # transformer needs: [B, S], True for tokens to mask (ignore)
            if attention_mask.dtype != torch.bool:
                src_key_padding_mask = (attention_mask == 0)
            else:
                src_key_padding_mask = ~attention_mask
        else:
            src_key_padding_mask = None
        
        # Apply transformer layers
        transformer_output = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Output projection
        output = self.output_norm(transformer_output)
        output = self.dropout(output)
        logits = self.output_projection(output)
        
        if return_reasoning_trace:
            # Return dummy reasoning info for compatibility
            reasoning_info = {
                'actual_steps': self.num_layers,  # Fixed steps = num layers
                'reasoning_trace': [],  # No intermediate states
                'final_change_magnitude': torch.zeros(batch_size, 1, device=device)
            }
            return logits, reasoning_info
        else:
            return logits
    
    def get_connection_analysis(self):
        """Dummy method for compatibility"""
        return {
            'connection_matrix': torch.zeros(1, 1),
            'sparsity_ratio': 0.0,
            'max_connection': 0.0,
            'mean_connection': 0.0
        }
    
    def reasoning_cost_loss(self, actual_steps, target_steps=4, weight=0.001):
        """Dummy method for compatibility - no reasoning cost for baseline"""
        return torch.tensor(0.0, device=next(self.parameters()).device)

def calculate_matching_config(config):
    """
    Calculate baseline transformer config to match Connection Transformer parameters.
    수정사항: 더 정확한 파라미터 계산
    """
    # Connection Transformer parameters
    N = config.num_slots
    D = config.d_model  
    r = config.bilinear_rank
    V = getattr(config, 'vocab_size', 32000)  # Will be set by tokenizer
    S = config.max_seq_len
    
    # Connection Transformer parameter count
    bilinear_params = 2 * N * N * D * r
    cross_attn_params = 6 * D * D
    embedding_params = (V + S) * D
    output_params = D * V
    layer_norm_params = config.max_reasoning_steps * 2 * D  # LayerNorm has weight + bias
    
    conn_total = bilinear_params + cross_attn_params + embedding_params + output_params + layer_norm_params
    
    print(f"\nConnection Transformer parameters:")
    print(f"  Bilinear: {bilinear_params:,}")
    print(f"  Cross-attention: {cross_attn_params:,}")  
    print(f"  Embeddings: {embedding_params:,}")
    print(f"  Output: {output_params:,}")
    print(f"  LayerNorms: {layer_norm_params:,}")
    print(f"  Total: {conn_total:,}")
    
    # Baseline transformer - shared parameters
    baseline_shared = embedding_params + output_params + 2 * D  # output LayerNorm
    available_for_layers = conn_total - baseline_shared
    
    # Each transformer layer parameters (더 정확한 계산)
    # TransformerEncoderLayer: 
    # - MultiheadAttention: 4*D^2 (q,k,v,out projections)
    # - FFN: 2*D*ffn_dim + ffn_dim + D (linear1, linear2, bias)
    # - LayerNorms: 4*D (2 LayerNorms * 2 params each)
    
    best_config = None
    best_diff = float('inf')
    
    for ffn_mult in [2, 3, 4, 6, 8]:
        ffn_dim = D * ffn_mult
        
        # Attention parameters
        attn_params = 4 * D * D  # q, k, v, out projections
        
        # FFN parameters  
        ffn_params = D * ffn_dim + ffn_dim * D + ffn_dim + D  # linear1 + linear2 + biases
        
        # LayerNorm parameters
        ln_params = 4 * D  # 2 LayerNorms * (weight + bias)
        
        params_per_layer = attn_params + ffn_params + ln_params
        
        num_layers = max(1, available_for_layers // params_per_layer)
        actual_layer_params = num_layers * params_per_layer
        total_baseline = baseline_shared + actual_layer_params
        
        diff = abs(total_baseline - conn_total)
        if diff < best_diff:
            best_diff = diff
            best_config = {
                'num_layers': int(num_layers),
                'ffn_multiplier': ffn_mult,
                'total_params': total_baseline,
                'param_diff': diff
            }
    
    print(f"\nMatched Baseline Transformer:")
    print(f"  Layers: {best_config['num_layers']}")
    print(f"  FFN multiplier: {best_config['ffn_multiplier']}")
    print(f"  Total params: {best_config['total_params']:,}")
    print(f"  Difference: {best_config['param_diff']:,} ({best_config['param_diff']/conn_total*100:.1f}%)")
    
    return best_config