# models/standard_transformer.py
import torch
import torch.nn as nn


class StandardTransformer(nn.Module):
    """Standard Transformer for SQuAD (Span Prediction)"""

    def __init__(self, vocab_size, d_model, num_heads,
                 num_layers, ffn_dim_multiplier, dropout, max_seq_len):
        super().__init__()

        self.d_model = d_model
        ffn_dim = d_model * ffn_dim_multiplier

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.qa_outputs_start = nn.Linear(d_model, 1)
        self.qa_outputs_end = nn.Linear(d_model, 1)

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print(f"🔶 Standard Transformer (SQuAD): {total_params:,} trainable parameters")

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.qa_outputs_start.weight)
        if self.qa_outputs_start.bias is not None: nn.init.zeros_(self.qa_outputs_start.bias)
        nn.init.xavier_uniform_(self.qa_outputs_end.weight)
        if self.qa_outputs_end.bias is not None: nn.init.zeros_(self.qa_outputs_end.bias)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        src_key_padding_mask = ~attention_mask if attention_mask is not None else None

        transformer_output = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        normed_output = self.norm(transformer_output)

        start_logits = self.qa_outputs_start(normed_output).squeeze(-1)
        end_logits = self.qa_outputs_end(normed_output).squeeze(-1)

        return start_logits, end_logits