# models/baseline_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaselineTransformer(nn.Module):
    """
    Standard Transformer Encoder-Decoder baseline for fair comparison.
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_encoder_layers=6, 
                 num_decoder_layers=6, num_heads=8, ffn_multiplier=4, dropout=0.1, 
                 max_seq_len=512, src_pad_token_id=0, tgt_pad_token_id=0):
        super().__init__()
        
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.src_pad_token_id = src_pad_token_id
        self.tgt_pad_token_id = tgt_pad_token_id
        
        # Source embeddings
        self.src_token_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_token_id)
        self.src_pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Target embeddings
        self.tgt_token_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_token_id)
        self.tgt_pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ffn_multiplier,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ffn_multiplier,
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
        
        self._init_parameters()
        
        # Report parameter count
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔶 Baseline Transformer (Encoder-Decoder): {total_params:,} parameters")
        print(f"   - Encoder layers: {num_encoder_layers}")
        print(f"   - Decoder layers: {num_decoder_layers}")
        
    def _init_parameters(self):
        """Initialize parameters"""
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
        
        # Output projection
        nn.init.normal_(self.output_projection.weight, std=0.02)
    
    def encode(self, src_input_ids, src_attention_mask=None):
        """
        Encoder forward pass
        
        Args:
            src_input_ids: [batch_size, src_seq_len]
            src_attention_mask: [batch_size, src_seq_len]
            
        Returns:
            encoder_output: [batch_size, src_seq_len, d_model]
        """
        batch_size, src_seq_len = src_input_ids.shape
        device = src_input_ids.device
        
        # Source embeddings
        positions = torch.arange(src_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.src_token_embedding(src_input_ids) + self.src_pos_embedding(positions)
        x = self.dropout(x)
        
        # Create source key padding mask
        if src_attention_mask is not None:
            src_key_padding_mask = (src_attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Apply encoder
        encoder_output = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        return encoder_output
    
    def decode(self, tgt_input_ids, encoder_output, tgt_attention_mask=None, src_attention_mask=None):
        """
        Decoder forward pass
        
        Args:
            tgt_input_ids: [batch_size, tgt_seq_len]
            encoder_output: [batch_size, src_seq_len, d_model]
            tgt_attention_mask: [batch_size, tgt_seq_len]
            src_attention_mask: [batch_size, src_seq_len]
            
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
        
        # Create memory key padding mask
        if src_attention_mask is not None:
            memory_key_padding_mask = (src_attention_mask == 0)
        else:
            memory_key_padding_mask = None
        
        # Apply decoder
        decoder_output = self.decoder(
            tgt=tgt_embeddings,
            memory=encoder_output,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
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
            return_reasoning_trace: bool (for compatibility)
            
        Returns:
            logits: [batch_size, tgt_seq_len, tgt_vocab_size]
            reasoning_info: dict (for compatibility)
        """
        # Encode
        encoder_output = self.encode(src_input_ids, src_attention_mask)
        
        # Decode
        logits = self.decode(tgt_input_ids, encoder_output, tgt_attention_mask, src_attention_mask)
        
        if return_reasoning_trace:
            # Return dummy reasoning info for compatibility
            batch_size = src_input_ids.size(0)
            device = src_input_ids.device
            reasoning_info = {
                'actual_steps': self.num_decoder_layers,  # Fixed steps = num decoder layers
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


def calculate_matching_config_enc_dec(config):
    """
    Calculate baseline transformer config to match Connection Transformer parameters.
    For encoder-decoder architecture.
    """
    # Connection Transformer parameters
    N = config.num_slots
    D = config.d_model  
    r = config.bilinear_rank
    src_V = getattr(config, 'src_vocab_size', 32000)
    tgt_V = getattr(config, 'tgt_vocab_size', 32000)
    S = config.max_seq_len
    
    # Connection Transformer parameter count
    bilinear_params = 2 * N * N * D * r
    cross_attn_params = 3 * D * D  # encoder cross-attention only
    embedding_params = (src_V + tgt_V + 2 * S) * D  # src + tgt embeddings + positions
    output_params = D * tgt_V
    layer_norm_params = config.max_reasoning_steps * 2 * D  # reasoning LayerNorms
    decoder_layers_params = config.num_decoder_layers * (
        4 * D * D +  # self-attention
        4 * D * D +  # cross-attention  
        2 * D * D * 4 +  # FFN
        4 * D  # LayerNorms
    )
    
    conn_total = bilinear_params + cross_attn_params + embedding_params + output_params + layer_norm_params + decoder_layers_params
    
    print(f"\nConnection Transformer (Encoder-Decoder) parameters:")
    print(f"  Bilinear: {bilinear_params:,}")
    print(f"  Cross-attention: {cross_attn_params:,}")  
    print(f"  Embeddings: {embedding_params:,}")
    print(f"  Output: {output_params:,}")
    print(f"  Reasoning LayerNorms: {layer_norm_params:,}")
    print(f"  Decoder layers: {decoder_layers_params:,}")
    print(f"  Total: {conn_total:,}")
    
    # Baseline transformer - shared parameters
    baseline_shared = embedding_params + output_params + 2 * D  # output LayerNorm
    available_for_layers = conn_total - baseline_shared
    
    # Try different encoder/decoder layer combinations
    best_config = None
    best_diff = float('inf')
    
    for ffn_mult in [2, 3, 4, 6, 8]:
        for enc_layers in range(1, 13):
            for dec_layers in range(1, 13):
                # Encoder layer parameters
                enc_attn_params = 4 * D * D  # q, k, v, out projections
                enc_ffn_params = D * D * ffn_mult + D * ffn_mult * D + D * ffn_mult + D  # linear1 + linear2 + biases
                enc_ln_params = 4 * D  # 2 LayerNorms * (weight + bias)
                enc_params_per_layer = enc_attn_params + enc_ffn_params + enc_ln_params
                
                # Decoder layer parameters
                dec_self_attn_params = 4 * D * D  # self-attention
                dec_cross_attn_params = 4 * D * D  # cross-attention
                dec_ffn_params = D * D * ffn_mult + D * ffn_mult * D + D * ffn_mult + D  # FFN
                dec_ln_params = 6 * D  # 3 LayerNorms * (weight + bias)
                dec_params_per_layer = dec_self_attn_params + dec_cross_attn_params + dec_ffn_params + dec_ln_params
                
                total_layer_params = enc_layers * enc_params_per_layer + dec_layers * dec_params_per_layer
                
                if total_layer_params <= available_for_layers:
                    total_baseline = baseline_shared + total_layer_params
                    diff = abs(total_baseline - conn_total)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_config = {
                            'num_encoder_layers': enc_layers,
                            'num_decoder_layers': dec_layers,
                            'ffn_multiplier': ffn_mult,
                            'total_params': total_baseline,
                            'param_diff': diff
                        }
    
    print(f"\nMatched Baseline Transformer (Encoder-Decoder):")
    print(f"  Encoder layers: {best_config['num_encoder_layers']}")
    print(f"  Decoder layers: {best_config['num_decoder_layers']}")
    print(f"  FFN multiplier: {best_config['ffn_multiplier']}")
    print(f"  Total params: {best_config['total_params']:,}")
    print(f"  Difference: {best_config['param_diff']:,} ({best_config['param_diff']/conn_total*100:.1f}%)")
    
    return best_config