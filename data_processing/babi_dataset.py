# data_processing/babi_dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import re
from typing import Dict, List  # typing 추가


class BabiDataset(Dataset):
    """bAbI Task Dataset (Hugging Face Dataset Card 기준)"""

    def __init__(self, task_id=1, babi_type_prefix="en-10k",  # type의 prefix (예: "en-10k", "en")
                 split='train', max_seq_len=128):
        self.max_seq_len = max_seq_len
        self.task_id_num = task_id
        # babi_type_prefix와 task_id를 결합하여 최종 name 파라미터 생성
        # 예: babi_type_prefix="en-10k", task_id=1 -> hf_name_param="en-10k-qa1"
        # 예: babi_type_prefix="en", task_id=16 -> hf_name_param="en-qa16"
        hf_name_param = f"{babi_type_prefix}-qa{self.task_id_num}"
        self.split = split

        print(f"Loading bAbI dataset (name: '{hf_name_param}', split: '{self.split}')...")

        try:
            # name 파라미터에 task_id 정보 포함, task_no는 사용 안 함 또는 None
            dataset_dict = load_dataset("facebook/babi_qa", name=hf_name_param)  # task_no 제거

            actual_split = self.split
            if self.split == 'validation':
                # en-valid-* 타입이 아니면 validation 스플릿이 없을 수 있음
                if 'validation' not in dataset_dict and 'test' in dataset_dict:
                    print(f"Warning: 'validation' split not found for {hf_name_param}. Using 'test' split instead.")
                    actual_split = 'test'
                elif 'validation' not in dataset_dict:
                    raise ValueError(f"'validation' split (and no 'test' fallback) not found for {hf_name_param}.")

            if actual_split not in dataset_dict:
                available_splits = list(dataset_dict.keys())
                raise ValueError(
                    f"❌ Split '{actual_split}' not found for bAbI (name: {hf_name_param}). "
                    f"Available splits: {available_splits}."
                )
            self.raw_data_iterable = dataset_dict[actual_split]
            print(f"✅ Successfully loaded from facebook/babi_qa.")

        except Exception as e:
            print(f"❌ HuggingFace 로딩 실패 (name: {hf_name_param}): {e}")
            print("🔄 대체 방법 시도 중...")
            try:
                # 대체 로딩 시에도 task_id에 맞는 데이터셋을 찾도록 로직 개선 필요
                # 현재는 고정된 대체 데이터셋 사용
                fallback_dataset_name = "habanoz/babi_qa_en_valid_10k_qa1"
                if not (babi_type_prefix == "en-valid-10k" and self.task_id_num == 1):  # 대체 데이터셋과 일치하지 않으면 경고
                    print(
                        f"⚠️ Fallback dataset {fallback_dataset_name} might not match requested name {hf_name_param}.")

                dataset_fallback = load_dataset(fallback_dataset_name)
                actual_split_fallback = self.split
                if self.split == 'validation' and 'validation' not in dataset_fallback and 'test' in dataset_fallback:
                    actual_split_fallback = 'test'

                self.raw_data_iterable = dataset_fallback[
                    actual_split_fallback] if actual_split_fallback in dataset_fallback else dataset_fallback['train']
                print("✅ 대체 데이터셋 로딩 성공")
            except Exception as e_fallback:
                print(f"❌ 모든 온라인 소스 실패: {e_fallback}")
                raise Exception("bAbI 데이터셋 로딩 실패.")

        self.data = self._convert_format()
        print(f"Processed {len(self.data)} QA pairs from the dataset.")

        self.vocab = self._build_vocab()
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {self.vocab_size}")

    def _convert_format(self):
        converted_qa_pairs = []
        # self.raw_data_iterable은 Dataset 객체, 각 요소는 하나의 bAbI 문제 세트(story 딕셔너리 포함)
        for example_set in self.raw_data_iterable:
            story_dict = example_set['story']  # story 자체가 딕셔너리

            # story_dict 안의 'text', 'type', 'answer'는 모두 같은 길이의 리스트
            num_lines = len(story_dict['text'])

            current_story_context_lines = []
            for i in range(num_lines):
                line_text = story_dict['text'][i]
                line_type = story_dict['type'][i]  # 0: context, 1: question

                if line_type == 0:  # 문맥
                    current_story_context_lines.append(line_text)
                elif line_type == 1:  # 질문
                    question_text = line_text
                    # 해당 질문 라인의 답변을 가져옴
                    answer_text = story_dict['answer'][i]

                    # supporting_ids도 필요하면 여기서 추출 가능
                    # supporting_fact_indices = [int(sid) -1 for sid in story_dict['supporting_ids'][i] if sid] # 0-based index
                    # supporting_facts = [current_story_context_lines[s_idx] for s_idx in supporting_fact_indices if s_idx < len(current_story_context_lines)]

                    if answer_text:  # 답변이 있는 질문만 QA 쌍으로 구성
                        converted_qa_pairs.append({
                            'story': list(current_story_context_lines),  # 현재까지의 문맥
                            'question': question_text,
                            'answer': answer_text,
                            'task_id_num': self.task_id_num  # 숫자 task_id
                        })
                    # bAbI는 각 질문 후 컨텍스트가 리셋되지 않고 이어질 수 있음.
                    # 하지만 일반적인 QA 쌍으로 만들려면, 질문이 나올 때마다 이전 컨텍스트를 사용.
                    # 만약 태스크가 "하나의 스토리 블록 내 여러 QA"라면, current_story_context_lines를 초기화하지 않음.
                    # 여기서는 각 질문을 독립적인 QA 쌍으로 간주 (이전 컨텍스트 사용)
        return converted_qa_pairs

    def _build_vocab(self):  # 이전과 거의 동일, self.data 구조 변경에 따른 접근만 수정
        vocab = set(['<PAD>', '<UNK>', '<SEP>'])
        for qa_pair in self.data:  # self.data는 이제 QA 쌍 리스트
            story_words = ' '.join(qa_pair['story']).lower().split()
            question_words = qa_pair['question'].lower().split()
            answer_words = qa_pair['answer'].lower().split()
            for word in story_words + question_words + answer_words:
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word:
                    vocab.add(clean_word)
        return ['<PAD>', '<UNK>', '<SEP>'] + sorted(list(vocab - {'<PAD>', '<UNK>', '<SEP>'}))

    def _tokenize(self, text):  # 이전과 동일
        words = re.findall(r'\w+', text.lower())
        token_ids = []
        for word in words:
            token_ids.append(self.word_to_id.get(word, self.word_to_id['<UNK>']))
        return token_ids

    def __len__(self):
        return len(self.data)  # self.data는 이제 QA 쌍의 리스트

    def __getitem__(self, idx):  # 이전과 거의 동일, self.data 구조 변경에 따른 접근만 수정
        qa_pair = self.data[idx]

        story_text = ' '.join(qa_pair['story'])
        question_text = qa_pair['question']
        input_text = f"{story_text} <SEP> {question_text}"
        answer_text = qa_pair['answer']

        input_ids = self._tokenize(input_text)
        tokenized_answer = self._tokenize(answer_text)
        if not tokenized_answer:
            tokenized_answer = [self.word_to_id['<UNK>']]
        answer_ids_list = tokenized_answer  # 변수명 변경

        if len(input_ids) > self.max_seq_len - 1:
            input_ids = input_ids[:self.max_seq_len - 1]

        input_length = len(input_ids)
        padded_input_ids = input_ids + [self.word_to_id['<PAD>']] * (self.max_seq_len - input_length)  # 패딩된 input_ids
        attention_mask_bool = [True] * input_length + [False] * (self.max_seq_len - input_length)

        # answer_ids_list는 토큰 ID의 리스트. DataLoader가 배치 만들 때 패딩해줌.
        # train_model에서는 answer_ids[:, 0]을 사용.
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask_bool, dtype=torch.bool),
            'answer_ids': torch.tensor(answer_ids_list, dtype=torch.long),
            'answer_text': answer_text
        }