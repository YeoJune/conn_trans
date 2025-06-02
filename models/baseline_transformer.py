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
    """간단한 baseline 설정 계산"""
    # Connection Transformer 대략적 파라미터
    conn_params = (
        config.num_slots * config.num_slots * config.d_model * config.bilinear_rank * 2 +  # bilinear
        config.d_model * config.d_model * 3 +  # cross-attention
        config.vocab_size * config.d_model * 2 +  # embeddings
        config.d_model * config.vocab_size  # output
    )
    
    # 간단한 매칭 (정확하지 않아도 됨)
    if config.d_model <= 64:
        return {'num_encoder_layers': 3, 'num_decoder_layers': 3, 'ffn_multiplier': 4}
    elif config.d_model <= 128:
        return {'num_encoder_layers': 4, 'num_decoder_layers': 4, 'ffn_multiplier': 4}
    else:
        return {'num_encoder_layers': 6, 'num_decoder_layers': 6, 'ffn_multiplier': 4}
    
    def load_pretrained_embeddings(self, model_name="google-t5/t5-base"):
        """T5 pre-trained embeddings 로딩"""
        try:
            from transformers import T5Model
            pretrained = T5Model.from_pretrained(model_name)
            
            # 토큰 임베딩 복사
            self.src_token_embedding.weight.data = pretrained.shared.weight.data.clone()
            self.tgt_token_embedding.weight.data = pretrained.shared.weight.data.clone()
            
            print(f"✅ Loaded pre-trained embeddings from {model_name}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load pre-trained embeddings: {e}")
            return False