import torch
from torch.utils.data import Dataset


class KeywordDataset(Dataset):
    """
    Custom PyTorch Dataset for NER token classification.
    Handles sub-word token alignment for BERT-like models.
    """

    def __init__(self, data, tokenizer, label2id, max_len=128):
        """
        Args:
            data: List of dicts with "tokens" and "labels" keys.
            tokenizer: Hugging Face tokenizer instance.
            label2id: Dict mapping label strings to integer IDs.
            max_len: Maximum sequence length for padding/truncation.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        labels = item["labels"]

        # Tokenize pre-split words
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        # Align labels to sub-word tokens
        word_ids = tokenized_inputs.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special tokens ([CLS], [SEP], [PAD]) get -100
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                # First sub-token of a word gets the real label
                aligned_labels.append(self.label2id[labels[word_id]])
            else:
                # Subsequent sub-tokens of the same word get -100
                aligned_labels.append(-100)
            previous_word_id = word_id

        # Squeeze to remove batch dimension: (1, seq_len) -> (seq_len)
        return {
            "input_ids": tokenized_inputs["input_ids"].squeeze(0),
            "attention_mask": tokenized_inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }

