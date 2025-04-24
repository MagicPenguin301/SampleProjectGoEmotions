from typing import Iterable
import torch
from functools import reduce
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datasets import Dataset


class GoEmotionsDataset(Dataset):
    @staticmethod
    def preprocess(
    data,
    num_classes,
    tokenize_fn=word_tokenize,
    lemmatize_fn=WordNetLemmatizer().lemmatize,
):
        def id_to_onehot_label(labels: Iterable[Iterable[int]], num_classes):
            one_hot = torch.zeros(len(labels), num_classes)
            for i, multi_label in enumerate(labels):
                for single_label in multi_label:
                    one_hot[i, single_label-1] = 1
            return one_hot

        def build_vocab(tokens: Iterable[str]):
            vocab = {}
            vocab["[PAD]"] = 0
            vocab["[UNK]"] = 1
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
            return vocab

        # tokenize and lemmatize
        sents = [list(map(lemmatize_fn, tokenize_fn(item["text"]))) for item in data]
        # flatten
        flattened_tokens = reduce(lambda x, y: x + y, sents, [])
        # build a vocab
        vocab = build_vocab(flattened_tokens)
        # padding
        max_length = max(len(sent) for sent in sents)
        for sent in sents:
            if len(sent) < max_length:
                sent += ["[PAD]"] * (max_length - len(sent))
        # convert input tokens to ids
        unk_idx = vocab["[UNK]"]
        input_ids = [[vocab.get(token, unk_idx) for token in sent] for sent in sents]

        # convert labels into one-hot
        multi_labels = [item["labels"] for item in data]
        onehot_labels = id_to_onehot_label(multi_labels, num_classes)

        return torch.tensor(input_ids, dtype=torch.long), onehot_labels, vocab

    def __init__(
        self,
        data,
        num_classes,
        tokenize_fn=word_tokenize,
        lemmatize_fn=WordNetLemmatizer().lemmatize,
    ):
        self.input_ids, self.labels, self.word2id = GoEmotionsDataset.preprocess(
            data, num_classes, tokenize_fn, lemmatize_fn
        )
        self.num_classes = num_classes
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}

if __name__ == "__main__":
    from datasets import load_dataset
    train_data = load_dataset(
    "google-research-datasets/go_emotions", "simplified", split="train[:100]"
)
    dataset = GoEmotionsDataset(train_data, 27)
    print(dataset[15])