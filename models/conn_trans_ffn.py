# models/conn_trans_ffn.py
import torch
import torch.nn as nn
from .base_conn_trans import ConnectionTransformer  # 부모 클래스 임포트


class ConnTransWithFFN(ConnectionTransformer):
    """Connection Transformer with FFN for SQuAD"""

    def __init__(self, vocab_size, d_model, num_slots,
                 num_reasoning_steps, max_seq_len, connection_init_std, spectral_radius_limit,
                 ffn_dim_multiplier=4, dropout=0.1):
        super().__init__(vocab_size, d_model, num_slots, num_reasoning_steps, max_seq_len,
                         connection_init_std, spectral_radius_limit)  # 부모 클래스 초기화

        ffn_dim = d_model * ffn_dim_multiplier
        self.reasoning_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        # self.ffn_norm = nn.LayerNorm(d_model) # 부모의 reasoning_norms 사용 또는 별도 추가

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print(f"🔸 ConnTransWithFFN (SQuAD): {total_params:,} trainable parameters")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                return_reasoning_trace=False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        X_input = self.token_embedding(input_ids) + self.pos_embedding(positions)

        Q_input = self.W_q_input(X_input)
        K_slots = self.W_k_slots(self.H)
        V_input = self.W_v_input(X_input)
        A_compress = F.softmax(Q_input @ K_slots.T / math.sqrt(self.d_model), dim=-1)
        IR_activation = A_compress.transpose(-1, -2) @ V_input
        H_state = self.H.unsqueeze(0).expand(batch_size, -1, -1) + IR_activation
        reasoning_trace_states = [H_state.clone()] if return_reasoning_trace else []

        for step in range(self.num_reasoning_steps):
            Influence = H_state @ self.C
            H_state_after_conn = H_state + Influence
            H_state_norm_before_ffn = self.reasoning_norms[step](H_state_after_conn)

            ffn_output = self.reasoning_ffn(H_state_norm_before_ffn)
            H_state = H_state_norm_before_ffn + ffn_output  # Residual connection for FFN

            if return_reasoning_trace:
                reasoning_trace_states.append(H_state.clone())

        Q_output = self.W_q_output(X_input)
        K_final = self.W_k_final(H_state)
        V_final = self.W_v_final(H_state)
        A_expand = F.softmax(Q_output @ K_final.transpose(-1, -2) / math.sqrt(self.d_model), dim=-1)
        Y_output = A_expand @ V_final
        start_logits = self.qa_outputs_start(Y_output).squeeze(-1)
        end_logits = self.qa_outputs_end(Y_output).squeeze(-1)

        if return_reasoning_trace:
            return start_logits, end_logits, reasoning_trace_states
        else:
            return start_logits, end_logits