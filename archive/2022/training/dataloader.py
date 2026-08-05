import os
from tokenizers import Tokenizer, models
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence


def load_sequences(base_path, train_embedding=False):
    sequences = []
    labels = []
    for folder in os.listdir(base_path):
        if folder == 'Human' and train_embedding:
            continue
        if os.path.isdir(f'{base_path}/{folder}'):
            for file_name in os.listdir(f'{base_path}/{folder}'):
                fname = f'{base_path}/{folder}/{file_name}'
                with open(fname) as f:
                    for line in f:
                        if line.startswith('>'):
                            continue
                        sequences.append(line)
                        labels.append(folder)
    return sequences, labels


def sample_data(sequences, labels):
    idx = np.random.choice(len(labels), int(len(labels) * 0.5))

    sequences = [sequences[i] for i in idx]
    labels = [labels[i] for i in idx]

    return sequences, labels


def get_3_splits(sequences, labels):
    (train_sequence, test_sequence, train_label,
     test_label) = train_test_split(sequences, labels, test_size=0.2, stratify=labels)

    (valid_sequence, test_sequence, valid_label,
     test_label) = train_test_split(test_sequence,
                                    test_label,
                                    test_size=0.5,
                                    stratify=test_label)

    return (train_sequence, valid_sequence, test_sequence, train_label, valid_label,
            test_label)


class SequenceDataset(Dataset):

    def __init__(self,
                 sequence,
                 labels,
                 tokenizer_file='gene_tokenizer.json',
                 label_dict=None):
        """sequence: List of Str

        ["ACTG...", "GTCA...", ...]
        """
        self.sequence = sequence
        if label_dict is None:
            self.label_dict = self.get_label_dict(labels)
        else:
            self.label_dict = label_dict
        self.labels = self.encode_labels(labels)

        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer = self.tokenizer.from_file(tokenizer_file)
        self.tokenizer.enable_padding()

    def get_label_dict(self, labels):
        label_set = np.unique(labels)
        label_dict = {}
        for i, x in enumerate(label_set):
            label_dict[x] = i

        return label_dict

    def encode_labels(self, labels):
        encoded_label = []
        for y in labels:
            encoded_label.append(self.label_dict[y])

        return encoded_label

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        seq = self.sequence[idx].strip()
        label = self.labels[idx]
        encoded_seq = self.tokenizer.encode(seq)
        return torch.LongTensor(encoded_seq.ids), label

    def collate_fn(self, batch):
        """batch: list of (torch.LongTensor, int)"""
        sequences = []
        labels = []
        for item in batch:
            sequences.append(item[0])
            labels.append(item[1])

        sequence = pad_sequence(sequences,
                                batch_first=True,
                                padding_value=self.tokenizer.padding['pad_id'])
        labels = torch.LongTensor(labels)

        return sequence, labels
