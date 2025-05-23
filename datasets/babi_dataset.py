# datasets/babi_dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import re
from typing import Dict, List  # typing 추가


class BabiDataset(Dataset):
    """bAbI Task Dataset - (사용자 제공 원본 코드 기반, load_dataset 수정)"""

    def __init__(self, task_id=1, babi_hf_config_name_prefix="en-10k-qa",  # config에서 prefix 받고 task_id 결합
                 split='train', max_seq_len=128):
        self.max_seq_len = max_seq_len
        self.task_id = task_id

        # task_id에 따라 hf_name_param, hf_task_no_param 결정
        # "en-10k-qa1" 같은 형식을 name으로 사용
        hf_name_param = f"{babi_hf_config_name_prefix}{self.task_id}"
        # 이 경우 task_no는 명시적으로 필요 없을 수 있으나, babi_qa 로더가 요구할 수 있음
        # 가장 안전한 것은 name에 모든 정보를 담거나, name="en-10k", task_no="qaX" 형태 사용
        # 여기서는 name에 모든 정보를 담는 것으로 가정 (사용자 요청)
        # 만약 오류 시, name="en-10k", task_no=f"qa{self.task_id}" 시도

        print(f"Loading bAbI task (name: {hf_name_param}, split: {split})...")

        try:
            dataset = load_dataset("facebook/babi_qa", name=hf_name_param)

            split_mapping = {
                'train': 'train',
                'validation': 'test',
                'test': 'test'
            }
            actual_split = split_mapping.get(split, 'train')

            if actual_split not in dataset:
                available_splits_msg = list(dataset.keys()) if isinstance(dataset, dict) else "N/A"
                raise ValueError(
                    f"❌ Split '{actual_split}' not found in dataset for {hf_name_param}. Available: {available_splits_msg}.")
            self.raw_data = dataset[actual_split]

        except Exception as e:
            print(f"❌ HuggingFace 로딩 실패 (name: {hf_name_param}): {e}")
            print("🔄 대체 방법 시도 중...")
            try:
                fallback_dataset_name = "habanoz/babi_qa_en_valid_10k_qa1"  # task_id 1에 대한 예시
                if self.task_id != 1:
                    print(
                        f"⚠️ Fallback dataset {fallback_dataset_name} might not match requested task_id {self.task_id}.")
                dataset_fallback = load_dataset(fallback_dataset_name)
                actual_split_fallback = split_mapping.get(split, 'train')
                self.raw_data = dataset_fallback[
                    actual_split_fallback] if actual_split_fallback in dataset_fallback else dataset_fallback['train']
                print("✅ 대체 데이터셋 로딩 성공")
            except:
                print("❌ 모든 온라인 소스 실패")
                print("💡 해결방법: ... (생략, 원본 코드 참조)")
                raise Exception("bAbI 데이터셋 로딩 실패.")

        self.data = self._convert_format()
        print(f"Loaded {len(self.data)} examples")

        self.vocab = self._build_vocab()
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {self.vocab_size}")

    def _convert_format(self):
        converted_data = []
        for example in self.raw_data:
            story_from_raw = example.get('story', [])
            processed_story_lines = []
            if isinstance(story_from_raw, list):
                for item in story_from_raw:
                    if isinstance(item, dict) and 'text' in item:
                        processed_story_lines.append(item['text'])
                    elif isinstance(item, str):
                        processed_story_lines.append(item)
            elif isinstance(story_from_raw, str):
                processed_story_lines.append(story_from_raw)

            converted_example = {
                'story': processed_story_lines,
                'question': example.get('question', ''),
                'answer': example.get('answer', ''),
                'task': self.task_id
            }
            converted_data.append(converted_example)
        return converted_data

    def _build_vocab(self):
        vocab = set(['<PAD>', '<UNK>', '<SEP>'])
        for example in self.data:
            story_words = ' '.join(example['story']).lower().split()
            question_words = example['question'].lower().split()
            answer_words = example['answer'].lower().split()
            for word in story_words + question_words + answer_words:
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word:
                    vocab.add(clean_word)
        return ['<PAD>', '<UNK>', '<SEP>'] + sorted(list(vocab - {'<PAD>', '<UNK>', '<SEP>'}))

    def _tokenize(self, text):
        words = re.findall(r'\w+', text.lower())
        token_ids = []
        for word in words:
            token_ids.append(self.word_to_id.get(word, self.word_to_id['<UNK>']))
        return token_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        story_text = ' '.join(example['story'])
        question_text = example['question']
        input_text = f"{story_text} <SEP> {question_text}"
        answer_text = example['answer']

        input_ids = self._tokenize(input_text)
        tokenized_answer = self._tokenize(answer_text)
        if not tokenized_answer:
            tokenized_answer = [self.word_to_id['<UNK>']]
        answer_ids = tokenized_answer

        if len(input_ids) > self.max_seq_len - 1:
            input_ids = input_ids[:self.max_seq_len - 1]

        input_length = len(input_ids)
        input_ids += [self.word_to_id['<PAD>']] * (self.max_seq_len - input_length)
        attention_mask = [1] * input_length + [0] * (self.max_seq_len - input_length)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
            'answer_ids': torch.tensor(answer_ids, dtype=torch.long),  # train_model에서 첫번째 토큰 사용
            'answer_text': answer_text
        }