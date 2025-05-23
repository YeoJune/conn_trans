import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset # Keep this
import numpy as np
import time
import json
import re
from typing import Dict, List, Tuple, Optional # Added Optional
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ===== RTX 4090 최적화 하이퍼파라미터 =====
CONFIG = {
    # 모델 크기 (4090 24GB 최적화)
    "d_model": 512,
    "num_ir": 1024,     # 2 * d_model
    "num_steps": 4,     # 추론 단계 (Transformer layers와 동일)
    "num_heads": 8,
    "ffn_dim": 2048,    # 4 * d_model
    "dropout": 0.1,
    
    # 학습 설정
    "batch_size": 32,   # 4090 고성능 활용
    "max_seq_len": 128,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "max_epochs": 3, # Reduced for quicker example, original was 15
    "gradient_clip": 1.0,
    
    # 정규화 및 안정성
    "c_regularization": 1e-4,
    "spectral_radius_limit": 0.9,  # 안정성을 위한 스펙트럼 제한
    "connection_scale": 0.1,       # Connection 스케일링
}

# ... (PureConnTrans, ConnTransWithFFN, StandardTransformer classes remain the same) ...
# (I'll include them if you need the full script again, but they are unchanged from your last version)
class PureConnTrans(nn.Module):
    """Pure Connection Transformer - 수치 안정성 강화 버전"""
    
    def __init__(self, vocab_size, config=CONFIG):
        super().__init__()
        
        self.config = config
        d_model = config["d_model"]
        num_ir = config["num_ir"]
        num_steps = config["num_steps"]
        num_heads = config["num_heads"]
        
        # 임베딩
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1000, d_model) # Max sequence length of 1000 for positions
        
        # 고정 IR 노드 (학습되지 않음)
        self.register_buffer('H', torch.randn(num_ir, d_model) * 0.02)
        
        # 연결 행렬 (핵심 학습 파라미터!) - 안전한 초기화
        self.C = nn.Parameter(self._init_connection_matrix(num_ir))
        
        # Connection 스케일링 (학습 가능)
        self.connection_scale = nn.Parameter(torch.tensor(config["connection_scale"]))
        
        # 어텐션
        self.input_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.output_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # 정규화
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.connection_norm = nn.LayerNorm(d_model)  # Connection 후 정규화
        
        # 분류기
        self.classifier = nn.Linear(d_model, vocab_size)
        
        # 초기화
        self._init_weights()
        
        # 수치 안정성 모니터링
        self.numerical_warnings = 0
        
        # 파라미터 수 출력
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🔹 Pure Conn-Trans: {total_params:,} parameters")
    
    def _init_connection_matrix(self, num_ir):
        """안전한 Connection Matrix 초기화"""
        C = torch.randn(num_ir, num_ir) * 0.001
        diagonal_idx = torch.arange(num_ir)
        C[diagonal_idx, diagonal_idx] = -0.1
        C = C * 0.01
        return C
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
    
    def spectral_normalize_connection(self):
        """Connection Matrix의 스펙트럼 정규화"""
        with torch.no_grad():
            try:
                eigenvals = torch.linalg.eigvals(self.C)
                spectral_radius = torch.abs(eigenvals).max().real
                
                if spectral_radius > self.config["spectral_radius_limit"]:
                    scale_factor = self.config["spectral_radius_limit"] / spectral_radius
                    self.C.data *= scale_factor
                    if self.numerical_warnings < 3:
                        print(f"⚠️ Connection Matrix 정규화: spectral_radius={spectral_radius:.3f}")
                        self.numerical_warnings += 1
            except Exception as e:
                if self.numerical_warnings < 3:
                    print(f"⚠️ 스펙트럼 계산 실패: {e}")
                    self.numerical_warnings += 1
    
    def check_numerical_stability(self):
        """수치 안정성 체크"""
        C_norm = torch.norm(self.C, 'fro').item()
        C_max = self.C.abs().max().item()
        
        if C_norm > 10 and self.numerical_warnings < 3:
            print(f"⚠️ Warning: C norm large: {C_norm:.3f}")
            self.numerical_warnings += 1
        if C_max > 5 and self.numerical_warnings < 3:
            print(f"⚠️ Warning: C max value large: {C_max:.3f}")
            self.numerical_warnings += 1
        return C_norm, C_max
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if self.training:
            self.spectral_normalize_connection()
            if torch.rand(1).item() < 0.1:
                self.check_numerical_stability()
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions)
        input_emb = token_emb + pos_emb
        
        H_batch = self.H.unsqueeze(0).expand(batch_size, -1, -1)
        X, _ = self.input_attention(
            query=H_batch,
            key=input_emb,
            value=input_emb,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        X = self.input_norm(X)
        
        I = torch.eye(self.config["num_ir"], device=device)
        scaled_C = self.connection_scale * self.C
        
        for step in range(self.config["num_steps"]):
            knowledge_injection = torch.matmul(scaled_C, self.H)
            state_evolution = torch.matmul(I + scaled_C, X)
            X_new = knowledge_injection.unsqueeze(0) + state_evolution
            X = self.connection_norm(X_new)
            X = torch.clamp(X, min=-10, max=10)
        
        H_effective = self.H.unsqueeze(0) + X
        output_states, _ = self.output_attention(
            query=input_emb,
            key=H_effective,
            value=H_effective
        )
        output_states = self.output_norm(output_states)
        logits = self.classifier(output_states)
        return logits
    
    def get_reasoning_trace(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions)
        input_emb = token_emb + pos_emb
        
        H_batch = self.H.unsqueeze(0).expand(batch_size, -1, -1)
        X, _ = self.input_attention(
            query=H_batch, key=input_emb, value=input_emb,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        X = self.input_norm(X)
        
        reasoning_trace = [X.clone()]
        norms = [torch.norm(X, dim=-1).mean().item()]
        
        I = torch.eye(self.config["num_ir"], device=device)
        scaled_C = self.connection_scale * self.C
        
        for step in range(self.config["num_steps"]):
            knowledge_injection = torch.matmul(scaled_C, self.H)
            state_evolution = torch.matmul(I + scaled_C, X)
            X_new = knowledge_injection.unsqueeze(0) + state_evolution
            X = self.connection_norm(X_new)
            X = torch.clamp(X, min=-10, max=10)
            reasoning_trace.append(X.clone())
            norms.append(torch.norm(X, dim=-1).mean().item())
        
        return reasoning_trace, norms
    
    def get_connection_stats(self):
        with torch.no_grad():
            C_scaled = self.connection_scale * self.C
            eigenvals = torch.linalg.eigvals(C_scaled)
            
            return {
                'connection_scale': self.connection_scale.item(),
                'frobenius_norm': torch.norm(C_scaled, 'fro').item(),
                'spectral_radius': torch.abs(eigenvals).max().real.item(),
                'max_eigenval_real': eigenvals.real.max().item(),
                'min_eigenval_real': eigenvals.real.min().item(),
                'condition_number': torch.linalg.cond(C_scaled).item() if C_scaled.shape[0] == C_scaled.shape[1] else float('inf') # Cond only for square
            }

class ConnTransWithFFN(PureConnTrans):
    """Connection Transformer with FFN - 수치 안정성 강화"""
    
    def __init__(self, vocab_size, config=CONFIG):
        super().__init__(vocab_size, config)
        
        d_model = config["d_model"]
        ffn_dim = config["ffn_dim"]
        dropout = config["dropout"]
        
        self.reasoning_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        self.reasoning_norm2 = nn.LayerNorm(d_model)
        
        # Recalculate total_params specifically for this class after adding FFN
        # The super().__init__() already printed its own params. This print will be for the FFN version.
        total_params_ffn = sum(p.numel() for p in self.parameters())
        # The superclass already prints its name and params. This print clarifies the *additional* or *total* for this subclass.
        # To avoid double printing or confusion, let's adjust the superclass print or this one.
        # For now, let's make this print clearly about THIS class.
        # The `sum(p.numel() for p in self.parameters())` will include superclass parameters.
        print(f"🔸 Conn-Trans + FFN: {total_params_ffn:,} parameters (includes PureConnTrans base)")


    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if self.training:
            self.spectral_normalize_connection()
            if torch.rand(1).item() < 0.1:
                self.check_numerical_stability()
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions)
        input_emb = token_emb + pos_emb
        
        H_batch = self.H.unsqueeze(0).expand(batch_size, -1, -1)
        X, _ = self.input_attention(
            query=H_batch, key=input_emb, value=input_emb,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        X = self.input_norm(X)
        
        I = torch.eye(self.config["num_ir"], device=device)
        scaled_C = self.connection_scale * self.C
        
        for step in range(self.config["num_steps"]):
            knowledge_injection = torch.matmul(scaled_C, self.H)
            state_evolution = torch.matmul(I + scaled_C, X)
            X_conn = knowledge_injection.unsqueeze(0) + state_evolution
            X_conn = self.connection_norm(X_conn)
            
            X_ffn = X_conn + self.reasoning_ffn(X_conn) # Residual connection for FFN part
            X = self.reasoning_norm2(X_ffn)
            X = torch.clamp(X, min=-10, max=10)
        
        H_effective = self.H.unsqueeze(0) + X
        output_states, _ = self.output_attention(
            query=input_emb, key=H_effective, value=H_effective
        )
        output_states = self.output_norm(output_states)
        logits = self.classifier(output_states)
        return logits

class StandardTransformer(nn.Module):
    """Standard Transformer - 공정한 비교를 위한 베이스라인"""
    
    def __init__(self, vocab_size, config=CONFIG):
        super().__init__()
        
        self.config = config
        d_model = config["d_model"]
        num_heads = config["num_heads"]
        num_layers = config["num_steps"]
        ffn_dim = config["ffn_dim"]
        dropout = config["dropout"]
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1000, d_model) # Max sequence length of 1000 for positions
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, vocab_size)
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🔶 Standard Transformer: {total_params:,} parameters")
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions)
        x = token_emb + pos_emb
        
        src_key_padding_mask = ~attention_mask if attention_mask is not None else None
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        logits = self.classifier(x)
        return logits

class BabiDataset(Dataset):
    """bAbI Task Dataset - HuggingFace 방식 및 어휘 공유 개선"""
    
    def __init__(self, task_id: int = 16, split: str = 'train', 
                 max_seq_len: int = 128, type: str = 'en',
                 word_to_id: Optional[Dict[str, int]] = None, 
                 vocab: Optional[List[str]] = None):
        self.max_seq_len = max_seq_len
        self.task_id = task_id
        self.split = split
        self.type = type  # This 'type' corresponds to 'name' in load_dataset (e.g., 'en', 'en-10k')
        
        print(f"📦 Loading bAbI task {task_id} (config='{type}', split='{split}')...")

        task_name = f"qa{task_id}"
        try:
            # `name` parameter in load_dataset is for the configuration (e.g., 'en')
            # `task_no` is a specific kwarg for the babi_qa loading script
            dataset_dict = load_dataset("facebook/babi_qa", name=self.type, task_no=task_name)
        except Exception as e:
            # Fallback attempt, though the above is standard for babi_qa
            try:
                print(f"⚠️ Initial load failed, trying with type as positional: {e}")
                dataset_dict = load_dataset("facebook/babi_qa", self.type, task_no=task_name)
            except Exception as e_inner:
                raise RuntimeError(f"❌ Failed to load bAbI dataset (task {task_name}, config {self.type}): {e_inner}")

        if split not in dataset_dict:
            available_splits = list(dataset_dict.keys())
            suggestion = " Consider using 'test' for validation." if split == 'validation' else ""
            raise ValueError(
                f"❌ Split '{split}' not found for bAbI task {task_name} (config: {self.type}). "
                f"Available splits: {available_splits}.{suggestion}"
            )
        
        self.raw_data = dataset_dict[split]
        self.data = self._convert_format()
        print(f"✅ Loaded {len(self.data)} examples for task {task_id}, config '{self.type}', split '{self.split}'")

        # Vocabulary Handling
        if word_to_id is not None and vocab is not None:
            print(f"📚 Using pre-existing vocabulary for task {task_id} ({split}) provided by training set.")
            self.word_to_id = word_to_id
            self.vocab = vocab
            self.vocab_size = len(self.vocab)
        else:
            if self.split != 'train':
                # This case should ideally not happen if pipeline is correct (val/test should get vocab from train)
                print(f"⚠️ Warning: Building new vocabulary for a non-train split ('{self.split}'). "
                      "Ensure this is intended or provide vocab from training split.")
            print(f"🛠️ Building new vocabulary from current data ({self.split} split).")
            self.vocab = self._build_vocab_from_data(self.data)
            self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
            self.vocab_size = len(self.vocab)
        
        print(f"🔤 Vocabulary size for task {task_id} ({self.split}): {self.vocab_size}")

    def _convert_format(self) -> List[Dict]:
        converted_data = []
        for example in self.raw_data:
            # Ensure story is a list of strings
            story_lines = example.get('story', [])
            if isinstance(story_lines, str): # Sometimes story might be a single string
                story_lines = [story_lines]

            converted_data.append({
                'story': story_lines, 
                'question': example.get('question', ''),
                'answer': example.get('answer', ''),
                'task': self.task_id
            })
        return converted_data

    def _build_vocab_from_data(self, data_to_build_from: List[Dict]) -> List[str]:
        """Builds vocabulary from the provided data (list of dicts)."""
        vocab_set = set(['<PAD>', '<UNK>', '<SEP>'])
        for ex in data_to_build_from:
            story_words = ' '.join(ex['story']).lower().split()
            question_words = ex['question'].lower().split()
            answer_words = ex['answer'].lower().split()
            for word in story_words + question_words + answer_words:
                clean = re.sub(r'[^\w]', '', word) # Keep only alphanumeric
                if clean: # Ensure word is not empty after cleaning
                    vocab_set.add(clean)
        # Ensure <PAD>, <UNK>, <SEP> are first, then sorted rest
        core_tokens = ['<PAD>', '<UNK>', '<SEP>']
        other_tokens = sorted(list(vocab_set - set(core_tokens)))
        return core_tokens + other_tokens


    def _tokenize(self, text: str) -> List[int]:
        # Simple regex tokenizer, ensures consistency with vocab building
        words = re.findall(r'\w+', text.lower()) 
        return [self.word_to_id.get(w, self.word_to_id['<UNK>']) for w in words]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor or str]:
        ex = self.data[idx]
        story_text = ' '.join(ex['story'])
        input_text = f"{story_text} <SEP> {ex['question']}"
        
        input_ids = self._tokenize(input_text)
        answer_ids = self._tokenize(ex['answer'])

        # Truncate/Pad input_ids
        input_ids = input_ids[:self.max_seq_len] # Simple truncation
        input_length = len(input_ids)
        
        padded_input_ids = input_ids + [self.word_to_id['<PAD>']] * (self.max_seq_len - input_length)
        attention_mask = [1] * input_length + [0] * (self.max_seq_len - input_length)
        
        if not answer_ids: # Handle cases where tokenization results in empty list (e.g. answer is just punctuation)
            answer_ids = [self.word_to_id['<PAD>']]
            
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
            'answer_ids': torch.tensor(answer_ids, dtype=torch.long), 
            'answer_text': ex['answer']
        }

# ... (train_model, print_comparison_results, visualize_connection_matrix, analyze_reasoning_evolution)
# These functions remain largely the same as your last version with checkpoints.

# Dummy dataset creator (mostly same, ensure it uses CONFIG for max_seq_len)
def create_dummy_babi_dataset(size, task_id, config, 
                              word_to_id_ref: Optional[Dict[str,int]] = None, 
                              vocab_ref: Optional[List[str]] = None):
    """bAbI 데이터 로딩 실패시 더미 데이터셋 생성. CONFIG에서 max_seq_len 사용."""
    class DummyBabiDataset(Dataset):
        def __init__(self, size, task_id, max_seq_len_cfg, 
                     word_to_id_ext=None, vocab_ext=None):
            self.data = []
            self.max_seq_len = max_seq_len_cfg # Use from outer scope config
            
            if word_to_id_ext and vocab_ext:
                print("📚 DummyDataset using provided vocabulary reference.")
                self.vocab = vocab_ext
                self.word_to_id = word_to_id_ext
            else:
                print("🛠️ DummyDataset building its own simple vocabulary.")
                self.vocab = ['<PAD>', '<UNK>', '<SEP>', 'if', 'then', 'is', 'what', 'where', 
                             'john', 'mary', 'kitchen', 'garden', 'green', 'frog', 'color', 'yes', 'no',
                             'apple', 'football', 'bedroom', 'office', 'journeyed', 'travelled', 'moved',
                             'got', 'took', 'discarded', 'put', 'down', 'picked', 'up', 'left', 'the', 'a',
                             'to', 'in', 'red', 'blue', 'yellow', 'bill', 'sandra', 'daniel', 'julie']
                self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
            self.vocab_size = len(self.vocab)
            
            templates = [
                ("mary moved to the bathroom", "john went to the hallway", "where is mary", "bathroom"),
                ("daniel was in the kitchen", "sandra picked up the milk", "where is daniel", "kitchen"),
            ] # Simplified for brevity
            
            for i in range(size):
                template = templates[i % len(templates)]
                self.data.append({
                    'story': [template[0], template[1]],
                    'question': template[2],
                    'answer': template[3],
                    'task': task_id
                })
        
        def _tokenize(self, text):
            words = re.findall(r'\w+', text.lower()) # Consistent with BabiDataset
            return [self.word_to_id.get(word, self.word_to_id['<UNK>']) for word in words]
        
        def __len__(self): return len(self.data)
        
        def __getitem__(self, idx):
            example = self.data[idx]
            story_text = ' '.join(example['story'])
            input_text = f"{story_text} <SEP> {example['question']}"
            input_ids = self._tokenize(input_text)
            answer_ids = self._tokenize(example['answer'])
            
            input_ids = input_ids[:self.max_seq_len]
            input_length = len(input_ids)
            padded_input_ids = input_ids + [self.word_to_id['<PAD>']] * (self.max_seq_len - input_length)
            attention_mask = [1] * input_length + [0] * (self.max_seq_len - input_length)
            if not answer_ids: answer_ids = [self.word_to_id['<PAD>']]
            
            return {
                'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
                'answer_ids': torch.tensor(answer_ids, dtype=torch.long),
                'answer_text': example['answer']
            }
    
    print(f"📍 CHECKPOINT: Creating DummyBabiDataset with size {size}, task {task_id}, max_seq_len {config['max_seq_len']}.")
    return DummyBabiDataset(size, task_id, config['max_seq_len'], word_to_id_ext=word_to_id_ref, vocab_ext=vocab_ref)

def main():
    """메인 실험 - 수치 안정성 강화 및 데이터 로딩 최신화 버전"""
    print("📍 CHECKPOINT: main() function started.")
    # ... (initial prints from your previous version) ...
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    print("\n📦 Data Loading (bAbI Task 16, 'en' config)...")
    train_dataset, val_dataset = None, None
    data_source_type = "Unknown"
    vocab_size_main = 0 # To store the actual vocab size used by models

    try:
        print("📍 CHECKPOINT: Attempting to load REAL bAbI dataset - TRAIN split (to build vocab).")
        # Load train dataset first to build/get vocabulary
        train_dataset_real = BabiDataset(task_id=16, split='train', 
                                         max_seq_len=CONFIG["max_seq_len"], type='en')
        
        # Use the vocabulary from the training set for the validation (test) set
        print("📍 CHECKPOINT: Attempting to load REAL bAbI dataset - TEST split (using TRAIN vocab).")
        val_dataset_real = BabiDataset(task_id=16, split='test',  # Using 'test' split for validation
                                       max_seq_len=CONFIG["max_seq_len"], type='en',
                                       word_to_id=train_dataset_real.word_to_id,  # Pass training vocab
                                       vocab=train_dataset_real.vocab)          # Pass training vocab
        
        train_dataset = train_dataset_real
        val_dataset = val_dataset_real
        vocab_size_main = train_dataset.vocab_size # Get vocab size from training set
        print(f"✅ Real bAbI dataset loaded successfully. Using 'test' split for validation. Vocab size: {vocab_size_main}")
        data_source_type = "Real bAbI Dataset (Train vocab, Test for val)"
        print("📍 CHECKPOINT: Successfully loaded real bAbI dataset with shared vocabulary.")
        
    except Exception as e:
        print(f"❌ Real bAbI dataset loading failed: {e}")
        print("   Common issues: internet connection, HuggingFace cache, or dataset availability.")
        print("   Ensure `datasets` library is up to date: pip install -U datasets")
        print("\n⚠️ Falling back to DUMMY dataset for architecture testing.")
        print("📍 CHECKPOINT: Real dataset loading failed. Falling back to dummy dataset.")
        
        # Create dummy train dataset (it will build its own vocab)
        train_dataset_dummy = create_dummy_babi_dataset(1000, 16, CONFIG)
        
        # Create dummy val dataset using the vocab from dummy train dataset
        val_dataset_dummy = create_dummy_babi_dataset(200, 16, CONFIG,
                                                      word_to_id_ref=train_dataset_dummy.word_to_id,
                                                      vocab_ref=train_dataset_dummy.vocab)
        train_dataset = train_dataset_dummy
        val_dataset = val_dataset_dummy
        vocab_size_main = train_dataset.vocab_size # Get vocab size from dummy training set
        print(f"🔧 Dummy dataset created and being used. Vocab size: {vocab_size_main}")
        data_source_type = "Dummy Fallback Dataset (shared vocab)"
        print("📍 CHECKPOINT: Dummy dataset created with shared vocabulary.")

    if train_dataset is None or val_dataset is None:
        print("❌ CRITICAL: Dataset not loaded. Exiting.")
        return {} 

    print("📍 CHECKPOINT: Creating DataLoaders.")
    # ... (DataLoader creation as in your previous version) ...
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=min(4, CONFIG["batch_size"] // 8 if CONFIG["batch_size"] > 1 else 0), # Adjusted num_workers
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=min(4, CONFIG["batch_size"] // 8 if CONFIG["batch_size"] > 1 else 0),
        pin_memory=torch.cuda.is_available()
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"  ✅ Device: {device}")
    print(f"  📚 Final Vocabulary Size for models: {vocab_size_main:,} tokens")
    print(f"  🔢 Train samples: {len(train_dataset):,}")
    print(f"  🔢 Val samples: {len(val_dataset):,}")
    print(f"  📦 Batch size: {CONFIG['batch_size']}")
    print(f"  📊 Data Source: {data_source_type}")
    print("📍 CHECKPOINT: Data loading and setup complete.")
    
    results = {}
    model_stats = {} # This was already there
    
    # --- Model Training Experiments ---
    # Each model will now be initialized with `vocab_size_main`

    # 1. Pure Conn-Trans 실험
    print("\n" + "="*60 + "\n🔹 EXPERIMENT 1: Pure Connection Transformer" + "\n" + "="*60)
    print("📍 CHECKPOINT: Starting Experiment 1: Pure Conn-Trans.")
    pure_model = PureConnTrans(vocab_size_main, CONFIG) # Use vocab_size_main
    pure_acc = train_model(pure_model, train_loader, val_loader, CONFIG, device, "Pure Conn-Trans")
    results['Pure Conn-Trans'] = pure_acc
    print("📍 CHECKPOINT: Finished Experiment 1: Pure Conn-Trans.")
    # ... (analysis for Pure Conn-Trans as before, using val_dataset[0] for sample_data)
    if pure_acc is not None and pure_acc > 0 and len(val_dataset) > 0 :
        try:
            pure_stats = pure_model.get_connection_stats()
            model_stats['Pure Conn-Trans'] = pure_stats
            visualize_connection_matrix(pure_model, "pure_connection_matrix.png", " (Pure)")
            sample_data = val_dataset[0]
            analyze_reasoning_evolution(pure_model, sample_data, "pure_reasoning_evolution.png")
        except Exception as e_viz: print(f"⚠️ Error during Pure Conn-Trans visualization/analysis: {e_viz}")
    del pure_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 2. Standard Transformer 실험  
    print("\n" + "="*60 + "\n🔶 EXPERIMENT 2: Standard Transformer" + "\n" + "="*60)
    print("📍 CHECKPOINT: Starting Experiment 2: Standard Transformer.")
    standard_model = StandardTransformer(vocab_size_main, CONFIG) # Use vocab_size_main
    standard_acc = train_model(standard_model, train_loader, val_loader, CONFIG, device, "Standard Transformer")
    results['Standard Transformer'] = standard_acc
    print("📍 CHECKPOINT: Finished Experiment 2: Standard Transformer.")
    del standard_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 3. Conn-Trans with FFN 실험
    print("\n" + "="*60 + "\n🔸 EXPERIMENT 3: Connection Transformer + FFN" + "\n" + "="*60)
    print("📍 CHECKPOINT: Starting Experiment 3: Conn-Trans + FFN.")
    ffn_model = ConnTransWithFFN(vocab_size_main, CONFIG) # Use vocab_size_main
    ffn_acc = train_model(ffn_model, train_loader, val_loader, CONFIG, device, "Conn-Trans + FFN")
    results['Conn-Trans + FFN'] = ffn_acc
    print("📍 CHECKPOINT: Finished Experiment 3: Conn-Trans + FFN.")
    # ... (analysis for FFN model as before, using val_dataset[0] for sample_data)
    if ffn_acc is not None and ffn_acc > 0 and len(val_dataset) > 0:
        try:
            ffn_stats = ffn_model.get_connection_stats()
            model_stats['Conn-Trans + FFN'] = ffn_stats
            visualize_connection_matrix(ffn_model, "ffn_connection_matrix.png", " (FFN)")
            sample_data = val_dataset[0] 
            analyze_reasoning_evolution(ffn_model, sample_data, "ffn_reasoning_evolution.png")
        except Exception as e_viz_ffn: print(f"⚠️ Error during Conn-Trans + FFN visualization/analysis: {e_viz_ffn}")
    del ffn_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # ... (Rest of the main function: print_comparison_results, saving JSON, conclusions)
    # Ensure NpEncoder class is defined as in your previous version if saving JSON.
    print("📍 CHECKPOINT: Starting comprehensive analysis and results comparison.")
    print_comparison_results(results) # Assuming this function is defined elsewhere as per your full script
    print("📍 CHECKPOINT: Finished comprehensive analysis.")

    print("📍 CHECKPOINT: Preparing to save experimental results.")
    experiment_results = {
        "experiment_type": "comprehensive_comparison_stable_2024_vocab_corrected",
        "task": "babi_task16_basic_induction",
        "babi_config_type": "en", # Explicitly state babi config
        "hardware_target": "RTX_4090_24GB",
        "actual_device": device,
        "data_source": data_source_type,
        "vocab_size_used": vocab_size_main,
        "config_hyperparameters": CONFIG, # Renamed for clarity
        "model_accuracies": results, # Renamed for clarity
        "connection_model_stats": model_stats, # Renamed for clarity
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }
    
    results_filename = f"babi16_comparison_{experiment_results['timestamp']}.json"
    try:
        with open(results_filename, "w") as f:
            # Assuming NpEncoder is defined (as in your full script)
            json.dump(experiment_results, f, indent=2, cls=NpEncoder) 
        print(f"  📄 Results successfully saved to: {results_filename}")
    except Exception as e_json:
        print(f"⚠️ Error saving JSON results: {e_json}")
    print("📍 CHECKPOINT: Finished saving experimental results.")
    
    print("📍 CHECKPOINT: Printing final conclusions and future work.")
    # ... (final conclusions and future work prints from your full script)
    
    print(f"\n✨ Experiment completed!")
    print(f"   All models (attempted) trained with numerical stability measures.")
    print(f"   Data loading system validated (used {data_source_type}). Vocab sharing implemented.")
    print(f"   Results and analysis (if successful) saved for future reference.")
    print("📍 CHECKPOINT: main() function finished.")
    return results

# Make sure NpEncoder is defined if you run this standalone part
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, torch.Tensor): return obj.cpu().numpy().tolist()
        return super(NpEncoder, self).default(obj)

# The train_model, print_comparison_results, visualize_connection_matrix, analyze_reasoning_evolution
# functions would be here from your original script.

if __name__ == "__main__":
    print("📍 CHECKPOINT: Script execution started (__name__ == '__main__').")
    # ... (Environment check, warnings filter from your previous version)
    print("🔧 Environment Check:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Datasets version: {load_dataset.__version__ if hasattr(load_dataset, '__version__') else 'unknown'}") # Check datasets version
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.lazy")
    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated") # Common in newer PyTorch/HF

    try:
        final_results = main() # Call the main experiment function
        print(f"\n🎉 All experiments sequence completed!")
        # ... (final status prints from your previous version)
        if final_results:
             print(f"Final Results Summary: {final_results}")
             if all(acc is not None and (isinstance(acc, float) and acc > 0.01) for acc in final_results.values()):
                 print(f"✅ All reported models achieved some minimal performance.")
             else:
                 print(f"⚠️ Some models may have had training issues or yielded low/no accuracy.")
        else:
            print(f"⚠️ No results returned from main experiment function.")

    except KeyboardInterrupt:
        print(f"\n🛑 Experiment interrupted by user.")
    except Exception as e:
        print(f"\n❌ CRITICAL EXPERIMENT FAILURE: {e}")
        import traceback
        traceback.print_exc()
    print("📍 CHECKPOINT: Script execution finished.")