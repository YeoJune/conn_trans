# datasets/squad_dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import re
import math
from typing import Dict, List, Optional  # Optional 추가


class SQuADDataset(Dataset):
    """SQuAD 1.1 Dataset Class using Hugging Face Tokenizer"""

    tokenizer = None  # 클래스 변수로 토크나이저 공유
    vocab_size = 0

    # 특수 토큰 ID들은 tokenizer 객체에서 직접 접근 가능 (예: self.tokenizer.pad_token_id)

    def __init__(self, split='train', max_seq_len=384,
                 doc_stride=128, max_query_length=64, tokenizer_name="bert-base-uncased"):
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.split = split

        if SQuADDataset.tokenizer is None or SQuADDataset.tokenizer.name_or_path != tokenizer_name:
            print(f"🔄 Initializing tokenizer: {tokenizer_name}")
            SQuADDataset.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            SQuADDataset.vocab_size = SQuADDataset.tokenizer.vocab_size
            print(f"📚 Tokenizer vocabulary size: {SQuADDataset.vocab_size}")
        self.tokenizer = SQuADDataset.tokenizer

        print(f"📦 Loading SQuAD 1.1 dataset ({split} split)...")
        try:
            self.raw_dataset = load_dataset("squad", split=split)
            print(f"✅ Successfully loaded {len(self.raw_dataset)} examples for SQuAD {split} split.")
        except Exception as e:
            print(f"❌ SQuAD loading failed: {e}")
            raise RuntimeError("Failed to load SQuAD dataset.")

        print(f"⚙️ Preprocessing SQuAD examples for {split} split...")
        self.examples = self._preprocess_squad_examples()
        print(f"📝 Created {len(self.examples)} SQuAD features for {split} split.")

    def _preprocess_squad_examples(self):
        features = []
        for example_idx, example in enumerate(self.raw_dataset):
            question_text = example["question"].strip()
            context_text = example["context"]
            answers = example["answers"]

            tokenized_examples = self.tokenizer(
                question_text,
                context_text,
                truncation="only_second",
                max_length=self.max_seq_len,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length"
            )

            offset_mapping_batch = tokenized_examples.pop("offset_mapping")

            for i, offsets in enumerate(offset_mapping_batch):
                input_ids = tokenized_examples["input_ids"][i]
                attention_mask = tokenized_examples["attention_mask"][i]
                token_type_ids = tokenized_examples.get("token_type_ids")[i] if tokenized_examples.get(
                    "token_type_ids") else [0] * len(input_ids)

                cls_index = input_ids.index(self.tokenizer.cls_token_id)
                sequence_ids = tokenized_examples.sequence_ids(i)

                answer_start_char = -1
                answer_end_char = -1
                answer_text_feature = ""
                is_impossible = len(answers["answer_start"]) == 0  # SQuAD 1.1은 항상 답변 있음

                if not is_impossible:
                    answer_start_char = answers["answer_start"][0]
                    answer_text_feature = answers["text"][0]
                    answer_end_char = answer_start_char + len(answer_text_feature)

                token_start_index_context = 0
                while sequence_ids[token_start_index_context] != 1:
                    token_start_index_context += 1
                    if token_start_index_context >= len(sequence_ids): break

                token_end_index_context = len(input_ids) - 1
                while sequence_ids[token_end_index_context] != 1:
                    token_end_index_context -= 1
                    if token_end_index_context < 0: break

                start_position = cls_index
                end_position = cls_index

                if not is_impossible and answer_start_char != -1 and \
                        token_start_index_context < len(offsets) and token_end_index_context >= 0 and \
                        offsets[token_start_index_context][0] <= answer_start_char and \
                        offsets[token_end_index_context][1] >= answer_end_char:

                    current_tok_idx = token_start_index_context
                    while current_tok_idx <= token_end_index_context and sequence_ids[current_tok_idx] == 1:
                        if offsets[current_tok_idx][0] <= answer_start_char and offsets[current_tok_idx][
                            1] > answer_start_char:
                            start_position = current_tok_idx
                            break
                        current_tok_idx += 1

                    current_tok_idx = token_end_index_context
                    while current_tok_idx >= token_start_index_context and sequence_ids[current_tok_idx] == 1:
                        if offsets[current_tok_idx][0] < answer_end_char and offsets[current_tok_idx][
                            1] >= answer_end_char:
                            end_position = current_tok_idx
                            break
                        current_tok_idx -= 1

                    if start_position != cls_index and end_position == cls_index:
                        if offsets[start_position][0] <= answer_start_char and offsets[start_position][
                            1] >= answer_end_char:
                            end_position = start_position

                    if start_position > end_position or start_position == cls_index or end_position == cls_index:
                        start_position = cls_index
                        end_position = cls_index

                features.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'start_positions': start_position,
                    'end_positions': end_position,
                    'example_id': example['id'],
                    'answer_text': answer_text_feature
                })
        return features

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feature = self.examples[idx]
        return {
            'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.bool),
            'token_type_ids': torch.tensor(feature['token_type_ids'], dtype=torch.long),
            'start_positions': torch.tensor(feature['start_positions'], dtype=torch.long),
            'end_positions': torch.tensor(feature['end_positions'], dtype=torch.long),
            'answer_text': feature['answer_text'],
            'example_id': feature['example_id']
        }