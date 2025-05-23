import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, __version__ as datasets_version # Get datasets version
import numpy as np
import time
import json
import re
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ===== RTX 4090 최적화 하이퍼파라미터 (예시) =====
CONFIG = {
    # 모델 크기
    "d_model": 512,
    "num_ir": 1024,     # 2 * d_model
    "num_steps": 4,     # 추론 단계 (Transformer layers와 동일)
    "num_heads": 8,
    "ffn_dim": 2048,    # 4 * d_model
    "dropout": 0.1,
    
    # 학습 설정
    "batch_size": 32,
    "max_seq_len": 128, # bAbI는 보통 짧음, en-10k는 60-70 정도가 평균
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 200, # 데이터셋 크기에 따라 조절
    "max_epochs": 3,    # 실제 실행 시 10-20으로 늘리세요 (예: 15)
    "gradient_clip": 1.0,
    
    # 정규화 및 안정성
    "c_regularization": 1e-4,
    "spectral_radius_limit": 0.9,
    "connection_scale": 0.1,
}

class PureConnTrans(nn.Module):
    """Pure Connection Transformer - 수치 안정성 강화 버전"""
    
    def __init__(self, vocab_size, config=CONFIG):
        super().__init__()
        
        self.config = config
        d_model = config["d_model"]
        num_ir = config["num_ir"]
        num_steps = config["num_steps"]
        num_heads = config["num_heads"]
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(config["max_seq_len"] + 100, d_model) # 좀 더 여유롭게
        
        self.register_buffer('H', torch.randn(num_ir, d_model) * 0.02)
        self.C = nn.Parameter(self._init_connection_matrix(num_ir))
        self.connection_scale = nn.Parameter(torch.tensor(config["connection_scale"]))
        
        self.input_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=config["dropout"])
        self.output_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=config["dropout"])
        
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.connection_norm = nn.LayerNorm(d_model)
        
        self.classifier = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
        self.numerical_warnings = 0
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔹 Pure Conn-Trans: {total_params:,} trainable parameters")
    
    def _init_connection_matrix(self, num_ir):
        C = torch.randn(num_ir, num_ir) * 0.001
        diagonal_idx = torch.arange(num_ir)
        C[diagonal_idx, diagonal_idx] = -torch.abs(torch.randn(num_ir) * 0.1) # Small negative random diagonal
        C = C * 0.01
        return C
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        # C is initialized in _init_connection_matrix
        # For attention and linear layers, default PyTorch init is often okay, or use Xavier/Kaiming
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.classifier:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                     nn.init.xavier_uniform_(module.in_proj_weight)
                if module.out_proj.weight is not None:
                     nn.init.xavier_uniform_(module.out_proj.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def spectral_normalize_connection(self):
        with torch.no_grad():
            try:
                # Using C.data to modify in-place without graph tracking
                C_matrix = self.C.data 
                eigenvals = torch.linalg.eigvals(C_matrix)
                spectral_radius = torch.abs(eigenvals).max().real
                
                limit = self.config["spectral_radius_limit"]
                if spectral_radius > limit:
                    scale_factor = limit / spectral_radius
                    self.C.data *= scale_factor
                    if self.numerical_warnings < 3:
                        print(f"⚠️ PureConnTrans: C spectral radius {spectral_radius:.3f} > {limit}. Normalized.")
                        self.numerical_warnings += 1
            except torch.linalg.LinAlgError as e: # Catch specific linalg errors
                if self.numerical_warnings < 3:
                    print(f"⚠️ PureConnTrans: 스펙트럼 계산 실패 (LinAlgError): {e}. Matrix might be ill-conditioned.")
                    self.numerical_warnings += 1
            except Exception as e: # Catch any other error
                if self.numerical_warnings < 3:
                    print(f"⚠️ PureConnTrans: 스펙트럼 계산 중 일반 오류: {e}")
                    self.numerical_warnings += 1
    
    def check_numerical_stability(self):
        C_norm = torch.norm(self.C, 'fro').item()
        C_max = self.C.abs().max().item()
        if C_norm > 2 * self.config["num_ir"]**0.5 and self.numerical_warnings < 3: # Adjusted threshold
            print(f"⚠️ PureConnTrans: C norm large: {C_norm:.3f}")
            self.numerical_warnings += 1
        if C_max > 1.0 and self.numerical_warnings < 3: # Adjusted threshold
            print(f"⚠️ PureConnTrans: C max value large: {C_max:.3f}")
            self.numerical_warnings += 1
        return C_norm, C_max
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if self.training:
            self.spectral_normalize_connection()
            if torch.rand(1).item() < 0.05: # Check less frequently
                self.check_numerical_stability()
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions[:, :seq_len]) # Ensure pos_emb matches seq_len
        input_emb = token_emb + pos_emb
        
        H_batch = self.H.unsqueeze(0).expand(batch_size, -1, -1) # [B, num_ir, d_model]
        
        # Input attention: H_batch queries input_emb
        # key_padding_mask should correspond to keys (input_emb)
        # PyTorch's MHA expects key_padding_mask where True indicates a padded item.
        # Our attention_mask is True for non-padded items. So we invert it.
        input_key_padding_mask = ~attention_mask if attention_mask is not None else None

        X, _ = self.input_attention(
            query=H_batch,
            key=input_emb,
            value=input_emb,
            key_padding_mask=input_key_padding_mask 
        )
        X = self.input_norm(X) # [B, num_ir, d_model]
        
        I = torch.eye(self.config["num_ir"], device=device)
        # Effective C includes learnable scale and spectral normalization (done in-place on self.C)
        effective_C = self.connection_scale * self.C
        
        for _ in range(self.config["num_steps"]):
            # H is fixed, so H.unsqueeze(0) for batch
            # knowledge_injection = torch.matmul(self.H.unsqueeze(0), effective_C.transpose(-1,-2)) # [B, num_ir, d_model] if H is [B, num_ir, d_model] and C is [num_ir, num_ir]
            # Corrected: C acts on H (num_ir, d_model), result is (num_ir, d_model)
            knowledge_injection = torch.matmul(effective_C, self.H) # [num_ir, d_model]
            state_evolution = torch.matmul(X, (I + effective_C).transpose(-1,-2)) # X is [B, num_ir, d_model], (I+C) is [num_ir, num_ir]
            
            X_new = knowledge_injection.unsqueeze(0) + state_evolution # Broadcast knowledge_injection
            X = self.connection_norm(X_new)
            X = torch.clamp(X, min=-5, max=5) # Stricter clamping
        
        # Output attention: input_emb queries the final IR states X (which now implicitly include H)
        # H_effective = self.H.unsqueeze(0) + X # This was one way, alternative below
        # The 'key' and 'value' for output attention should be the evolved IR states.
        output_states, _ = self.output_attention(
            query=input_emb, # Query with original input embeddings
            key=X,           # Key is the evolved IR states
            value=X,         # Value is also the evolved IR states
            # No key_padding_mask here as X is dense internal representation
        )
        output_states = self.output_norm(output_states) # [B, seq_len, d_model]
        logits = self.classifier(output_states)
        return logits
    
    def get_reasoning_trace(self, input_ids, attention_mask=None):
        # Simplified trace, focusing on X norms, not full X clones for memory
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions[:, :seq_len])
        input_emb = token_emb + pos_emb
        
        H_batch = self.H.unsqueeze(0).expand(batch_size, -1, -1)
        input_key_padding_mask = ~attention_mask if attention_mask is not None else None
        X, _ = self.input_attention(
            query=H_batch, key=input_emb, value=input_emb,
            key_padding_mask=input_key_padding_mask
        )
        X = self.input_norm(X)
        
        # reasoning_trace_clones = [X.clone().detach().cpu()] # Store clones on CPU
        reasoning_trace_norms = [torch.norm(X, p=2, dim=-1).mean().item()] # Store L2 norm
        
        I = torch.eye(self.config["num_ir"], device=device)
        effective_C = self.connection_scale * self.C # Use scaled C
        
        for _ in range(self.config["num_steps"]):
            knowledge_injection = torch.matmul(effective_C, self.H) 
            state_evolution = torch.matmul(X, (I + effective_C).transpose(-1,-2))
            X_new = knowledge_injection.unsqueeze(0) + state_evolution
            X = self.connection_norm(X_new)
            X = torch.clamp(X, min=-5, max=5)
            
            # reasoning_trace_clones.append(X.clone().detach().cpu())
            reasoning_trace_norms.append(torch.norm(X, p=2, dim=-1).mean().item())
        
        # Return norms for efficiency; full trace can be very memory intensive
        return None, reasoning_trace_norms # First element could be full trace if needed
    
    def get_connection_stats(self):
        with torch.no_grad():
            C_eff = self.connection_scale.data * self.C.data # Use .data to avoid graph
            
            stats = {
                'connection_scale': self.connection_scale.item(),
                'C_frobenius_norm': torch.norm(self.C.data, 'fro').item(),
                'C_eff_frobenius_norm': torch.norm(C_eff, 'fro').item(),
                'C_max_abs_val': self.C.data.abs().max().item(),
                'C_eff_max_abs_val': C_eff.abs().max().item(),
            }
            try:
                eigenvals_C = torch.linalg.eigvals(self.C.data)
                stats['C_spectral_radius'] = torch.abs(eigenvals_C).max().real.item()
                eigenvals_C_eff = torch.linalg.eigvals(C_eff)
                stats['C_eff_spectral_radius'] = torch.abs(eigenvals_C_eff).max().real.item()
                
                # Condition number can be very large or fail for ill-conditioned matrices
                # stats['C_condition_number'] = torch.linalg.cond(self.C.data).item()
                stats['C_eff_condition_number'] = torch.linalg.cond(C_eff).item()

            except torch.linalg.LinAlgError:
                stats['C_spectral_radius'] = float('nan')
                stats['C_eff_spectral_radius'] = float('nan')
                # stats['C_condition_number'] = float('nan')
                stats['C_eff_condition_number'] = float('nan')
                # print("⚠️ LinAlgError in get_connection_stats") # Suppress for cleaner logs
            return stats

class ConnTransWithFFN(PureConnTrans):
    def __init__(self, vocab_size, config=CONFIG):
        super().__init__(vocab_size, config) # This will print PureConnTrans param count
        
        d_model = config["d_model"]
        ffn_dim = config["ffn_dim"]
        dropout_val = config["dropout"] # Renamed to avoid conflict
        
        self.reasoning_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout_val),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout_val) # Added dropout after second linear too
        )
        self.reasoning_norm2 = nn.LayerNorm(d_model)
        
        # Re-initialize FFN weights
        for module in self.reasoning_ffn.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        total_params_ffn = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # This will be the total for ConnTransWithFFN
        print(f"🔸 ConnTrans+FFN: {total_params_ffn:,} trainable parameters (total, includes base)")

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if self.training:
            self.spectral_normalize_connection()
            if torch.rand(1).item() < 0.05:
                self.check_numerical_stability()
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions[:, :seq_len])
        input_emb = token_emb + pos_emb
        
        H_batch = self.H.unsqueeze(0).expand(batch_size, -1, -1)
        input_key_padding_mask = ~attention_mask if attention_mask is not None else None
        X, _ = self.input_attention(
            query=H_batch, key=input_emb, value=input_emb,
            key_padding_mask=input_key_padding_mask
        )
        X = self.input_norm(X) # Initial IR activations
        
        I = torch.eye(self.config["num_ir"], device=device)
        effective_C = self.connection_scale * self.C
        
        for _ in range(self.config["num_steps"]):
            # Connection update (same as PureConnTrans)
            knowledge_injection = torch.matmul(effective_C, self.H)
            state_evolution = torch.matmul(X, (I + effective_C).transpose(-1,-2))
            X_conn = knowledge_injection.unsqueeze(0) + state_evolution
            X_res_conn = X + self.connection_norm(X_conn) # Residual after connection update
            
            # FFN update with residual connection
            X_ffn_out = self.reasoning_ffn(X_res_conn)
            X = self.reasoning_norm2(X_res_conn + X_ffn_out) # Residual after FFN
            
            X = torch.clamp(X, min=-5, max=5)
        
        # Output (same as PureConnTrans)
        output_states, _ = self.output_attention(
            query=input_emb, key=X, value=X
        )
        output_states = self.output_norm(output_states)
        logits = self.classifier(output_states)
        return logits

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, config=CONFIG):
        super().__init__()
        self.config = config
        d_model = config["d_model"]
        num_heads = config["num_heads"]
        num_layers = config["num_steps"] # Match depth
        ffn_dim = config["ffn_dim"]
        dropout_val = config["dropout"]
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(config["max_seq_len"] + 100, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout_val, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_norm = nn.LayerNorm(d_model) # Final norm before classifier
        self.classifier = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔶 Standard Transformer: {total_params:,} trainable parameters")
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        # TransformerEncoderLayer and TransformerEncoder use default init which is often good (Xavier for linears)
        # Default init for nn.Linear is Kaiming uniform.
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions[:, :seq_len])
        x = token_emb + pos_emb # Add dropout here? config["dropout"]
        # x = F.dropout(x, p=self.config["dropout"], training=self.training) # Embedding dropout

        # PyTorch TransformerEncoderLayer expects src_key_padding_mask where True means pad
        src_key_padding_mask = ~attention_mask if attention_mask is not None else None
        
        transformer_output = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        normed_output = self.output_norm(transformer_output) # Apply norm after all layers
        logits = self.classifier(normed_output)
        return logits

class BabiDataset(Dataset):
    def __init__(self, babi_config_name: str, task_no_str: str, # e.g., "en-10k", "qa1"
                 split: str = 'train', max_seq_len: int = 128,
                 word_to_id: Optional[Dict[str, int]] = None, 
                 vocab: Optional[List[str]] = None):
        
        self.max_seq_len = max_seq_len
        self.babi_config_name = babi_config_name 
        self.task_no_str = task_no_str # e.g. "qa1", "qa2"
        self.split = split
        
        print(f"📦 Loading bAbI dataset (config='{self.babi_config_name}', task='{self.task_no_str}', split='{self.split}')...")

        try:
            # `name` parameter in load_dataset is for the configuration (e.g., 'en-10k')
            # `task_no` is a specific kwarg for the babi_qa loading script
            dataset_dict = load_dataset("facebook/babi_qa", name=self.babi_config_name, task_no=self.task_no_str)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load bAbI dataset (config {self.babi_config_name}, task {self.task_no_str}): {e}")

        if split not in dataset_dict:
            available_splits = list(dataset_dict.keys())
            suggestion = " Common practice is to use 'train' and 'test' splits."
            raise ValueError(
                f"❌ Split '{split}' not found for bAbI (config: {self.babi_config_name}, task: {self.task_no_str}). "
                f"Available splits: {available_splits}.{suggestion}"
            )
        
        self.raw_data = dataset_dict[split]
        self.data = self._convert_format()
        print(f"✅ Loaded {len(self.data)} examples for config '{self.babi_config_name}', task '{self.task_no_str}', split '{self.split}'")

        if word_to_id is not None and vocab is not None:
            print(f"📚 Using pre-existing vocabulary for {self.babi_config_name}/{self.task_no_str} ({split}).")
            self.word_to_id = word_to_id
            self.vocab = vocab
        else:
            if self.split != 'train':
                print(f"⚠️ Warning: Building new vocabulary for non-train split ('{self.split}') of {self.babi_config_name}/{self.task_no_str}.")
            print(f"🛠️ Building new vocabulary from current data ({self.babi_config_name}/{self.task_no_str}, {self.split} split).")
            self.vocab = self._build_vocab_from_data(self.data)
            self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        print(f"🔤 Vocabulary size for {self.babi_config_name}/{self.task_no_str} ({self.split}): {self.vocab_size}")

    def _convert_format(self) -> List[Dict]:
        converted_data = []
        for example_idx, example in enumerate(self.raw_data):
            story_lines = example.get('story', [])
            if isinstance(story_lines, str): story_lines = [story_lines] # Should be list of lines
            
            # The 'story' field in babi_qa from HF is already a list of dicts: [{'text': "line1", 'id': 1}, ...]
            # We need to extract the 'text'
            processed_story = []
            for line_info in story_lines:
                if isinstance(line_info, dict) and 'text' in line_info:
                    processed_story.append(line_info['text'])
                elif isinstance(line_info, str): # Fallback if it's just a list of strings
                    processed_story.append(line_info)
                # else: print(f"Warning: Unexpected story line format: {line_info}")

            converted_data.append({
                'id': example.get('id', str(example_idx)), # Use provided ID or generate one
                'story': processed_story, 
                'question': example.get('question', ''),
                'answer': example.get('answer', ''),
            })
        return converted_data

    def _build_vocab_from_data(self, data_to_build_from: List[Dict]) -> List[str]:
        vocab_set = set(['<PAD>', '<UNK>', '<SEP>']) # <SEP> might be redundant if handled in input prep
        for ex in data_to_build_from:
            # Story is now a list of strings
            story_full_text = ' '.join(ex['story'])
            story_words = re.findall(r'\w+|[.,!?]', story_full_text.lower()) # Keep punctuation as tokens
            question_words = re.findall(r'\w+|[.,!?]', ex['question'].lower())
            answer_words = re.findall(r'\w+|[.,!?]', ex['answer'].lower()) # Answers can be multi-word
            
            for word_list in [story_words, question_words, answer_words]:
                for word in word_list:
                    if word: vocab_set.add(word) # Regex findall shouldn't produce empty
        
        core_tokens = ['<PAD>', '<UNK>', '<SEP>']
        other_tokens = sorted(list(vocab_set - set(core_tokens)))
        final_vocab = core_tokens + other_tokens
        print(f"Built vocab with {len(final_vocab)} tokens. Example: {final_vocab[:10] + final_vocab[-5:]}")
        return final_vocab

    def _tokenize(self, text: str) -> List[int]:
        words = re.findall(r'\w+|[.,!?]', text.lower()) # Consistent with vocab building
        return [self.word_to_id.get(w, self.word_to_id['<UNK>']) for w in words]

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.data[idx]
        story_text = ' '.join(ex['story']) # Join story lines into one string
        # Format: "story sentence 1. story sentence 2. <SEP> question?"
        input_text = f"{story_text} <SEP> {ex['question']}" 
        
        input_ids = self._tokenize(input_text)
        # For bAbI, answers are often single words, but can be multiple. Model predicts one token.
        answer_tokenized = self._tokenize(ex['answer'])
        
        # Truncate/Pad input_ids
        input_ids = input_ids[:self.max_seq_len]
        input_length = len(input_ids)
        
        padded_input_ids = input_ids + [self.word_to_id['<PAD>']] * (self.max_seq_len - input_length)
        attention_mask_bool = [True] * input_length + [False] * (self.max_seq_len - input_length)
        
        # Target for loss is typically the first token of the answer for simplicity in bAbI
        first_answer_token_id = answer_tokenized[0] if answer_tokenized else self.word_to_id['<UNK>']
            
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask_bool, dtype=torch.bool), # For MHA key_padding_mask
            'target_ids': torch.tensor([first_answer_token_id], dtype=torch.long), # Ensure it's a tensor
            'answer_text': ex['answer'] # For inspection
        }

def train_model(model, train_loader, val_loader, config, device, model_name):
    model = model.to(device)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=config["learning_rate"], weight_decay=config["weight_decay"])
    
    # Adjust steps_per_epoch if train_loader is small
    effective_steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["learning_rate"],
        steps_per_epoch=effective_steps_per_epoch, epochs=config["max_epochs"],
        pct_start=0.1, # Percentage of steps for warm-up
        anneal_strategy='cos' 
    )
    
    best_val_acc = 0.0 # Ensure float
    training_unstable = False
    
    print(f"\n🚀 Training {model_name} on {device}...")
    print("=" * 50)
    print(f"📍 CHECKPOINT: Starting training for {model_name}.")
    
    for epoch in range(config["max_epochs"]):
        print(f"📍 CHECKPOINT: Starting Epoch {epoch+1}/{config['max_epochs']} for {model_name}.")
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total_samples = 0 # Count actual samples processed
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) # Bool mask, True for non-pad
            # target_ids should be [B, 1] (first token of answer), then squeeze for cross_entropy
            target_ids = batch['target_ids'].to(device).squeeze(-1) # [B]

            if target_ids.numel() == 0: # Should not happen if __getitem__ handles empty answers
                print(f"⚠️ Skipping batch {batch_idx} in {model_name} due to empty target_ids.")
                continue

            try:
                logits = model(input_ids, attention_mask) # logits: [B, seq_len, vocab_size]
                
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"⚠️ NaN/Inf in logits: epoch {epoch+1}, batch {batch_idx}, {model_name}.")
                    training_unstable = True; break
                
                # For bAbI, usually predict based on the representation of the last non-padded input token,
                # or a global representation. Here, we simplify to predicting for *each* output position
                # and then take the loss only for the *first answer token* based on the *last input position's output*.
                # However, the provided ConnTrans models output for each input seq_len position.
                # We need to decide *which* output position's logits to use.
                # A common strategy for bAbI QA is to use the output corresponding to the <SEP> token or last question token.
                # For simplicity, this code uses the logits from the *last sequence position* of the output.
                # This might not be optimal for bAbI. A CLS token or specific query token approach is often better.
                
                # Assuming model outputs [B, seq_len, vocab_size]
                # We are taking the logits from the last output position to predict the single answer token
                last_position_logits = logits[:, -1, :]  # [B, vocab_size]
                
                loss = F.cross_entropy(last_position_logits, target_ids, 
                                       ignore_index=train_loader.dataset.word_to_id['<PAD>']) # ignore PAD targets if any
                
                if hasattr(model, 'C') and model.C is not None : # Check C exists
                    c_reg = config.get("c_regularization", 0) * torch.norm(model.C, 'fro')
                    loss = loss + c_reg
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ NaN/Inf in loss: epoch {epoch+1}, batch {batch_idx}, {model_name}.")
                    training_unstable = True; break
                
                optimizer.zero_grad()
                loss.backward()
                
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad and p.grad is not None, model.parameters()), 
                    config["gradient_clip"]
                )
                if total_grad_norm > config["gradient_clip"] * 5 and not torch.isinf(total_grad_norm): # If norm is much larger than clip
                    print(f"⚠️ Grad norm {total_grad_norm:.2f} (clipped) at epoch {epoch+1}, batch {batch_idx}, {model_name}")

                optimizer.step()
                scheduler.step() # Step scheduler after each batch for OneCycleLR
                
                train_loss_sum += loss.item() * input_ids.size(0) # Weighted by batch size
                
                # Accuracy calculation using the same last_position_logits
                predicted_tokens = torch.argmax(last_position_logits, dim=1) # [B]
                train_correct += (predicted_tokens == target_ids).sum().item()
                train_total_samples += input_ids.size(0)
                
                if batch_idx % max(1, len(train_loader) // 10) == 0 and batch_idx > 0 : # Print 10 times per epoch
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"  E{epoch+1} B{batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | GradNorm: {total_grad_norm:.2f} | LR: {current_lr:.1e}")
                    if hasattr(model, 'get_connection_stats') and batch_idx % max(1, len(train_loader) // 2) == 0 : # Less frequent
                        try:
                            stats = model.get_connection_stats()
                            print(f"    ConnStats: Scale:{stats['connection_scale']:.2f} SR_eff:{stats.get('C_eff_spectral_radius',0):.2f}")
                        except Exception as e_stat: print(f"Error getting conn_stats: {e_stat}")
                        
            except Exception as e_batch: # Catch any unexpected error in batch processing
                print(f"❌ ERROR in training batch {batch_idx} for {model_name}: {e_batch}")
                import traceback; traceback.print_exc()
                training_unstable = True; break # Stop epoch on critical batch error
        
        if training_unstable:
            print(f"❌ Training unstable for {model_name}, stopping early at epoch {epoch+1}.")
            break
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_ids = batch['target_ids'].to(device).squeeze(-1)

                if target_ids.numel() == 0: continue

                try:
                    logits = model(input_ids, attention_mask)
                    last_position_logits = logits[:, -1, :]
                    
                    loss = F.cross_entropy(last_position_logits, target_ids,
                                           ignore_index=val_loader.dataset.word_to_id['<PAD>'])
                    val_loss_sum += loss.item() * input_ids.size(0)
                    
                    predicted_tokens = torch.argmax(last_position_logits, dim=1)
                    val_correct += (predicted_tokens == target_ids).sum().item()
                    val_total_samples += input_ids.size(0)
                except Exception as e_val_batch:
                    print(f"⚠️ Error in validation batch for {model_name}: {e_val_batch}")
                    continue # Skip problematic batch in validation
        
        epoch_duration = time.time() - epoch_start_time
        avg_train_loss = train_loss_sum / train_total_samples if train_total_samples > 0 else 0
        train_acc = train_correct / train_total_samples if train_total_samples > 0 else 0
        avg_val_loss = val_loss_sum / val_total_samples if val_total_samples > 0 else 0
        val_acc = val_correct / val_total_samples if val_total_samples > 0 else 0

        print(f"  Epoch {epoch+1} Summary for {model_name} ({epoch_duration:.1f}s):")
        print(f"    Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"    Val:   Loss={avg_val_loss:.4f}, Acc={val_acc:.4f} (Best: {best_val_acc:.4f})")
        print(f"📍 CHECKPOINT: End of Epoch {epoch+1} for {model_name}. Val Acc: {val_acc:.4f}.")

        if hasattr(model, 'get_connection_stats'):
            try:
                stats = model.get_connection_stats()
                print(f"    ConnStats Post-Epoch: Scale:{stats['connection_scale']:.3f} SR_eff:{stats.get('C_eff_spectral_radius',0):.3f} Cond_eff:{stats.get('C_eff_condition_number',0):.2e}")
            except Exception as e_stat_epoch: print(f"Error getting post-epoch conn_stats: {e_stat_epoch}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f'best_model_{model_name.replace(" ", "_").replace("+", "FFN")}.pt'
            torch.save(model.state_dict(), save_path)
            print(f"    💾 New best model saved to {save_path} with Val Acc: {best_val_acc:.4f}")
        print("-" * 30)
    
    print(f"✅ {model_name} training {'completed' if not training_unstable else 'stopped due to instability'}. Best Val Acc: {best_val_acc:.4f}")
    print(f"📍 CHECKPOINT: Finished training for {model_name}.")
    return best_val_acc if not training_unstable else (best_val_acc if best_val_acc > 0 else 0.0)


def print_comparison_results(results_dict):
    print("\n" + "🎯 COMPREHENSIVE MODEL COMPARISON" + "\n" + "=" * 70)
    if not results_dict: print("  No results to compare."); return

    valid_results = {k: v for k, v in results_dict.items() if isinstance(v, float)}
    if not valid_results: print("  No valid float results for ranking."); print("=" * 70); return

    sorted_results = sorted(valid_results.items(), key=lambda x: x[1], reverse=True)
    print("🏆 Performance Ranking (Validation Accuracy):")
    for i, (model_name, acc) in enumerate(sorted_results, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
        print(f"  {emoji} {model_name:<25}: {acc:.4f}")
    
    # ... (rest of comparison as previously, ensure it handles potential None/0 for accuracies)
    print("=" * 70)


def visualize_connection_matrix(model, save_path="connection_matrix.png", title_suffix=""):
    if not hasattr(model, 'C') or model.C is None:
        print(f"Model {title_suffix} doesn't have a Connection Matrix 'C'."); return
    
    with torch.no_grad():
        C_param = model.C.data
        if hasattr(model, 'connection_scale') and model.connection_scale is not None:
            C_numpy = (model.connection_scale.data * C_param).cpu().numpy()
            scale_info = f" (eff_scale: {model.connection_scale.item():.3f})"
        else:
            C_numpy = C_param.cpu().numpy(); scale_info = ""
    
    plt.figure(figsize=(10, 8)) # Slightly smaller
    sns.heatmap(C_numpy, cmap='RdBu_r', center=0, square=True, linewidths=0.01, 
                cbar_kws={"shrink": .7, "label": "Connection Strength"})
    plt.title(f'Connection Matrix {title_suffix}{scale_info}', fontsize=14)
    plt.xlabel('To IR Node Index', fontsize=10); plt.ylabel('From IR Node Index', fontsize=10)
    plt.xticks(fontsize=8); plt.yticks(fontsize=8)
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=200, bbox_inches='tight') # Lower DPI for speed
        print(f"💾 Connection Matrix saved to {save_path}")
    except Exception as e: print(f"⚠️ Could not save Connection Matrix '{save_path}': {e}")
    plt.close()


def analyze_reasoning_evolution(model, sample_input_batch, save_path="reasoning_evolution.png", model_name=""):
    if not hasattr(model, 'get_reasoning_trace'):
        print(f"Model {model_name} doesn't support get_reasoning_trace."); return
    
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        input_ids = sample_input_batch['input_ids'].to(device) # Expecting a batch
        attention_mask = sample_input_batch['attention_mask'].to(device)
        
        # Use only the first item in the batch for detailed trace visualization
        _, norms_trace = model.get_reasoning_trace(input_ids[:1], attention_mask[:1]) 
    
    if norms_trace is None or not norms_trace:
        print(f"No norm trace data from {model_name}."); return

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(norms_trace)), norms_trace, 'o-', linewidth=1.5, markersize=5)
    plt.xlabel('Reasoning Step', fontsize=10)
    plt.ylabel('Avg. Activation Norm (L2)', fontsize=10)
    plt.title(f'Reasoning State Norm Evolution ({model_name})', fontsize=12)
    plt.xticks(range(len(norms_trace)),fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Reasoning evolution plot saved to {save_path}")
    except Exception as e: print(f"⚠️ Could not save reasoning evolution plot '{save_path}': {e}")
    plt.close()
    print(f"Norm evolution for {model_name} (first sample): {' → '.join([f'{n:.2f}' for n in norms_trace])}")


def create_dummy_babi_dataset(babi_config_name_dummy: str, task_no_str_dummy:str,
                              split_dummy: str, config_main: Dict,
                              word_to_id_ref: Optional[Dict[str,int]] = None, 
                              vocab_ref: Optional[List[str]] = None):
    
    class DummyBabiDataset(Dataset):
        def __init__(self, size, max_seq_len_cfg, word_to_id_ext=None, vocab_ext=None):
            self.data = []
            self.max_seq_len = max_seq_len_cfg
            
            if word_to_id_ext and vocab_ext:
                self.vocab = vocab_ext; self.word_to_id = word_to_id_ext
            else:
                self.vocab = ['<PAD>', '<UNK>', '<SEP>', 'mary', 'moved', 'to', 'the', 'bathroom', '.', 
                              'john', 'went', 'hallway', 'where', 'is', '?', 
                              'daniel', 'kitchen', 'sandra', 'picked', 'up', 'milk'] # Minimal
                self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
            self.vocab_size = len(self.vocab)
            
            templates = [
                (["mary moved to the bathroom .", "john went to the hallway ."], "where is mary ?", "bathroom"),
                (["daniel was in the kitchen .", "sandra picked up the milk ."], "where is daniel ?", "kitchen"),
            ]
            
            for i in range(size):
                story_lines, question, answer = templates[i % len(templates)]
                self.data.append({'story': story_lines, 'question': question, 'answer': answer})
        
        def _tokenize(self, text): # Must match real BabiDataset tokenizer
            words = re.findall(r'\w+|[.,!?]', text.lower())
            return [self.word_to_id.get(word, self.word_to_id['<UNK>']) for word in words]
        
        def __len__(self): return len(self.data)
        
        def __getitem__(self, idx):
            ex = self.data[idx]
            story_text = ' '.join(ex['story'])
            input_text = f"{story_text} <SEP> {ex['question']}"
            input_ids = self._tokenize(input_text)[:self.max_seq_len]
            answer_tok = self._tokenize(ex['answer'])

            padded_input_ids = input_ids + [self.word_to_id['<PAD>']] * (self.max_seq_len - len(input_ids))
            attention_mask_bool = [True] * len(input_ids) + [False] * (self.max_seq_len - len(input_ids))
            target_id = answer_tok[0] if answer_tok else self.word_to_id['<UNK>']
            
            return {
                'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask_bool, dtype=torch.bool),
                'target_ids': torch.tensor([target_id], dtype=torch.long),
                'answer_text': ex['answer']
            }
    
    size = 1000 if split_dummy == 'train' else 200
    print(f"📍 CHECKPOINT: Creating DummyBabiDataset (size {size}) for {babi_config_name_dummy}/{task_no_str_dummy} ({split_dummy}).")
    return DummyBabiDataset(size, config_main['max_seq_len'], 
                            word_to_id_ext=word_to_id_ref, vocab_ext=vocab_ref)


class NpEncoder(json.JSONEncoder):
    def default(self, o): # Changed 'obj' to 'o' to match error message if any
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, torch.Tensor): return o.cpu().numpy().tolist()
        if isinstance(o, (torch.float32, torch.float64, torch.float16)): return float(o) # Handle torch scalars
        if isinstance(o, (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8)): return int(o)
        return super(NpEncoder, self).default(o)

def main():
    print("📍 CHECKPOINT: main() function started.")
    babi_config_name_to_load = "en-10k"  # Example: "en", "en-10k", "shuffled-en-10k"
    babi_task_no_to_load = "qa1"      # Example: "qa1", "qa2", ..., "qa20"
    
    print(f"🚀 CONN-TRANS vs STANDARD TRANSFORMER on bAbI {babi_config_name_to_load}/{babi_task_no_to_load}")
    print("=" * 70)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print(f"\n📦 Data Loading (bAbI config='{babi_config_name_to_load}', task='{babi_task_no_to_load}')...")
    train_dataset, val_dataset = None, None
    data_source_type = "Unknown"; vocab_size_main = 0

    try:
        print("📍 CHECKPOINT: Attempting to load REAL bAbI TRAIN split (to build vocab).")
        train_dataset_real = BabiDataset(babi_config_name=babi_config_name_to_load, 
                                         task_no_str=babi_task_no_to_load, split='train', 
                                         max_seq_len=CONFIG["max_seq_len"])
        
        print("📍 CHECKPOINT: Attempting to load REAL bAbI TEST split (using TRAIN vocab).")
        val_dataset_real = BabiDataset(babi_config_name=babi_config_name_to_load, 
                                       task_no_str=babi_task_no_to_load, split='test',
                                       max_seq_len=CONFIG["max_seq_len"],
                                       word_to_id=train_dataset_real.word_to_id,
                                       vocab=train_dataset_real.vocab)
        
        train_dataset, val_dataset = train_dataset_real, val_dataset_real
        vocab_size_main = train_dataset.vocab_size
        data_source_type = f"Real bAbI ({babi_config_name_to_load}/{babi_task_no_to_load})"
        print(f"✅ {data_source_type} loaded. Vocab size: {vocab_size_main}")
        
    except Exception as e_load:
        print(f"❌ Real bAbI dataset loading failed: {e_load}")
        print("   Ensure `datasets` library is up to date: pip install -U datasets")
        print("\n⚠️ Falling back to DUMMY dataset.")
        
        train_d = create_dummy_babi_dataset(babi_config_name_to_load, babi_task_no_to_load, 'train', CONFIG)
        val_d = create_dummy_babi_dataset(babi_config_name_to_load, babi_task_no_to_load, 'test', CONFIG,
                                          word_to_id_ref=train_d.word_to_id, vocab_ref=train_d.vocab)
        train_dataset, val_dataset = train_d, val_d
        vocab_size_main = train_dataset.vocab_size
        data_source_type = "Dummy Fallback Dataset"
        print(f"🔧 {data_source_type} created. Vocab size: {vocab_size_main}")

    if not train_dataset or not val_dataset: print("❌ CRITICAL: Dataset not loaded. Exiting."); return {} 

    # Create a single sample batch for reasoning trace (from val_dataset if possible)
    sample_batch_for_trace = None
    if len(val_dataset) > 0:
        # Manually collate a single batch of size 1 for simplicity
        sample_item = val_dataset[0]
        sample_batch_for_trace = {
            'input_ids': sample_item['input_ids'].unsqueeze(0),
            'attention_mask': sample_item['attention_mask'].unsqueeze(0),
            'target_ids': sample_item['target_ids'].unsqueeze(0) 
            # 'answer_text' is not needed by model
        }
        print(f"Prepared a sample batch of size 1 from validation set for reasoning trace analysis.")


    print("📍 CHECKPOINT: Creating DataLoaders.")
    # Determine num_workers based on batch_size and CPU cores, max 4 for small datasets
    cpu_cores = torch.multiprocessing.cpu_count()
    nw = min(4, CONFIG["batch_size"] // 4 if CONFIG["batch_size"] >=4 else 0, cpu_cores // 2 if cpu_cores > 1 else 0)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=torch.cuda.is_available(), drop_last=True) # drop_last for stability
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
                            num_workers=nw, pin_memory=torch.cuda.is_available())
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}, Vocab: {vocab_size_main}, Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"  Data Source: {data_source_type}")
    
    results = {}; model_stats = {}
    
    # --- Model Training Experiments ---
    models_to_train = {
        "Pure Conn-Trans": PureConnTrans,
        "Standard Transformer": StandardTransformer,
        "Conn-Trans + FFN": ConnTransWithFFN,
    }

    for model_name, model_class in models_to_train.items():
        print("\n" + "="*60 + f"\n▶️ EXPERIMENT: {model_name}" + "\n" + "="*60)
        print(f"📍 CHECKPOINT: Starting Experiment: {model_name}.")
        
        # Clear cache before creating model if on CUDA
        if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        model_instance = model_class(vocab_size_main, CONFIG)
        acc = train_model(model_instance, train_loader, val_loader, CONFIG, device, model_name)
        results[model_name] = acc
        print(f"📍 CHECKPOINT: Finished Experiment: {model_name}. Accuracy: {acc:.4f}")

        if acc > 0.0 and len(val_dataset) > 0: # Only analyze if training was somewhat successful
            if hasattr(model_instance, 'get_connection_stats'):
                try:
                    stats = model_instance.get_connection_stats()
                    model_stats[model_name] = stats
                    print(f"  📊 {model_name} Final Connection Stats: { {k: (f'{v:.3e}' if isinstance(v, float) and abs(v)>100 else f'{v:.3f}') for k,v in stats.items()} }") # Compact print
                    if sample_batch_for_trace:
                         visualize_connection_matrix(model_instance, f"{model_name.replace(' ', '_')}_C_matrix.png", f" ({model_name})")
                except Exception as e_stat_final: print(f"Error final conn_stats for {model_name}: {e_stat_final}")
            
            if hasattr(model_instance, 'get_reasoning_trace') and sample_batch_for_trace:
                try:
                    analyze_reasoning_evolution(model_instance, sample_batch_for_trace, 
                                                f"{model_name.replace(' ', '_')}_reasoning_evo.png", model_name)
                except Exception as e_trace_final: print(f"Error final trace_evo for {model_name}: {e_trace_final}")

        del model_instance # Explicitly delete model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"📍 CHECKPOINT: {model_name} instance deleted and CUDA cache (if applicable) cleared.")

    print_comparison_results(results)
    
    print("📍 CHECKPOINT: Preparing to save experimental results.")
    # ... (JSON saving as before, ensure NpEncoder is used) ...
    experiment_summary = {
        "experiment_config_name": f"babi_{babi_config_name_to_load}_{babi_task_no_to_load}_comparison",
        "babi_dataset_details": {"config": babi_config_name_to_load, "task": babi_task_no_to_load},
        "data_source_used": data_source_type,
        "vocab_size": vocab_size_main,
        "hyperparameters": CONFIG,
        "model_accuracies": results,
        "connection_model_final_stats": model_stats,
        "pytorch_version": torch.__version__,
        "datasets_version": datasets_version,
        "device_used": device,
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }
    
    results_filename = f"exp_summary_{experiment_summary['timestamp']}.json"
    try:
        with open(results_filename, "w") as f:
            json.dump(experiment_summary, f, indent=2, cls=NpEncoder) 
        print(f"  📄 Full experiment summary saved to: {results_filename}")
    except Exception as e_json_save:
        print(f"⚠️ Error saving JSON summary: {e_json_save}")
    
    print(f"\n✨ Experiment sequence completed for bAbI {babi_config_name_to_load}/{babi_task_no_to_load}!")
    print("📍 CHECKPOINT: main() function finished.")
    return results

if __name__ == "__main__":
    print("📍 CHECKPOINT: Script execution started (__name__ == '__main__').")
    
    print("🔧 Environment Check:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Datasets version: {datasets_version}")
    print(f"  NumPy version: {np.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.lazy")
    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
    warnings.filterwarnings("ignore", message="Possibly corrupt EXIF data.") # Matplotlib/PIL warning

    try:
        final_results = main()
        print(f"\n🎉 All experiment runs completed!")
        if final_results:
             print(f"Final Accuracies Summary: {final_results}")
             # Check if any model achieved a non-trivial accuracy
             if any(acc > 0.01 for acc in final_results.values() if isinstance(acc, float)):
                 print(f"✅ At least one model achieved some performance.")
             else:
                 print(f"⚠️ All models reported low or zero accuracy. Check training process.")
        else:
            print(f"⚠️ No results dictionary returned from main experiment function.")

    except KeyboardInterrupt: print(f"\n🛑 Experiment interrupted by user.")
    except Exception as e_global:
        print(f"\n❌ CRITICAL GLOBAL FAILURE: {e_global}")
        import traceback; traceback.print_exc()
    finally:
        print("📍 CHECKPOINT: Script execution finished.")