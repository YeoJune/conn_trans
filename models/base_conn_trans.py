# models/base_conn_trans.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# from configs.base_config import BASE_CONFIG # 직접 CONFIG 사용 대신 인자로 받도록 수정

class ConnectionTransformer(nn.Module):
    """Connection Transformer for SQuAD (Span Prediction) - Base Class"""

    def __init__(self, vocab_size, d_model, num_slots,
                 num_reasoning_steps, max_seq_len, connection_init_std, spectral_radius_limit):  # config 대신 개별 인자
        super().__init__()

        self.d_model = d_model
        self.num_slots = num_slots
        self.num_reasoning_steps = num_reasoning_steps
        self.spectral_radius_limit = spectral_radius_limit  # 인자로 받음

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.register_buffer('H', torch.normal(0, 1, size=(num_slots, d_model)))
        self.C = nn.Parameter(torch.normal(0, connection_init_std, size=(num_slots, num_slots)))

        self.W_q_input = nn.Linear(d_model, d_model, bias=False)
        self.W_k_slots = nn.Linear(d_model, d_model, bias=False)
        self.W_v_input = nn.Linear(d_model, d_model, bias=False)

        self.W_q_output = nn.Linear(d_model, d_model, bias=False)
        self.W_k_final = nn.Linear(d_model, d_model, bias=False)
        self.W_v_final = nn.Linear(d_model, d_model, bias=False)

        self.qa_outputs_start = nn.Linear(d_model, 1)
        self.qa_outputs_end = nn.Linear(d_model, 1)

        self.reasoning_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_reasoning_steps)
        ])

        self._init_parameters()
        self.numerical_warnings = 0

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print(f"🔹 Base Connection Transformer (SQuAD): {total_params:,} trainable parameters") # main에서 출력

    def _init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        for module in [self.W_q_input, self.W_k_slots, self.W_v_input,
                       self.W_q_output, self.W_k_final, self.W_v_final,
                       self.qa_outputs_start, self.qa_outputs_end]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

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
            if self.d_model != self.num_slots:
                if self.numerical_warnings < 1:
                    # print(f"⚠️ Warning: d_model ({self.d_model}) != num_slots ({self.num_slots}). H_state @ C assumes D=N.")
                    self.numerical_warnings += 1
            Influence = H_state @ self.C
            H_state = H_state + Influence
            H_state = self.reasoning_norms[step](H_state)
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

    def get_connection_stats(self):
        with torch.no_grad():
            C_data = self.C.data
            stats_dict = {'frobenius_norm': torch.norm(C_data, 'fro').item()}
            try:
                I_plus_C = torch.eye(self.num_slots, device=C_data.device) + C_data
                eigenvals = torch.linalg.eigvals(I_plus_C)
                stats_dict['spectral_radius_I_plus_C'] = torch.abs(eigenvals).max().real.item()
                eigenvals_C = torch.linalg.eigvals(C_data)
                stats_dict['spectral_radius_C'] = torch.abs(eigenvals_C).max().real.item()
            except Exception:
                stats_dict['spectral_radius_I_plus_C'] = float('nan')
                stats_dict['spectral_radius_C'] = float('nan')
            return stats_dict

    def enforce_spectral_radius(self, max_radius=None):  # max_radius는 config에서 받음
        if max_radius is None: max_radius = self.spectral_radius_limit  # 클래스 변수 사용

        with torch.no_grad():
            target_matrix = torch.eye(self.num_slots, device=self.C.device) + self.C
            try:
                eigenvals = torch.linalg.eigvals(target_matrix)
                current_radius = torch.abs(eigenvals).max().real
                if current_radius > max_radius:
                    self.C.data *= (max_radius / current_radius * 0.9 + 0.1)
                    if self.numerical_warnings < 3:
                        # print(f"⚠️ (I+C) spectral radius {current_radius:.3f} > {max_radius}. C scaled.")
                        self.numerical_warnings += 1
                    return True
            except Exception as e:
                if self.numerical_warnings < 3:
                    # print(f"⚠️ Spectral radius enforcement failed: {e}")
                    self.numerical_warnings += 1
        return False

    def get_reasoning_trace(self, input_ids, attention_mask=None, token_type_ids=None):
        self.eval()
        with torch.no_grad():
            start_logits, end_logits, trace_states = self.forward(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                return_reasoning_trace=True
            )
        norms = []
        if trace_states:
            norms = [torch.norm(state, dim=-1).mean().item() for state in trace_states]
        return trace_states, norms