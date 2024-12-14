"""
The dataset class and dataloaders for pretraining/finetuning with Canadian invertebrates 1.5M dataset (deWaard et al. 2019) as preprocessed in BarcodeBERT (Millan Arias et al., 2023).
"""

from itertools import product
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.char_tokenizer import coin_flip, string_reverse_complement


def get_tokenizer(tokenizer_name, tokenizer_config):
    if tokenizer_name == "char":
        from utils.char_tokenizer import CharacterTokenizer

        return CharacterTokenizer(
            characters=tokenizer_config.characters,
            model_max_length=tokenizer_config.model_max_length,
            add_special_tokens=tokenizer_config.add_special_tokens,
            padding_side=tokenizer_config.padding_side,
        )
    elif tokenizer_name == "k_mer":
        from utils.k_mer_tokenizer import KmerTokenizer
        from torchtext.vocab import build_vocab_from_iterator

        letters = "ACGT"
        specials = ["[MASK]", "[SEP]"]
        if tokenizer_config.use_unk_token:
            # Encode all kmers which contain at least one N as <UNK>
            UNK_TOKEN = "[UNK]"
            specials.append(UNK_TOKEN)
        else:
            # Encode kmers which contain N differently depending on where it is
            letters += "N"
        kmer_iter = (
            ["".join(kmer)] for kmer in product(letters, repeat=tokenizer_config.k_mer)
        )
        vocab = build_vocab_from_iterator(kmer_iter, specials=specials)
        if tokenizer_config.use_unk_token:
            vocab.set_default_index(vocab.lookup_indices([UNK_TOKEN])[0])
        return KmerTokenizer(
            k=tokenizer_config.k_mer,
            padding_side=tokenizer_config.padding_side,
            vocabulary_mapper=vocab,
            stride=tokenizer_config.stride,
            padding=tokenizer_config.padding,
            max_len=tokenizer_config.max_len,
        )


class TokenizedDNADataset(Dataset):
    def __init__(self, data_path, dataset_config, tokenizer_config):
        self.data_path = data_path
        self.phase = dataset_config.phase
        assert self.phase in ["pretrain", "finetune"]
        self.max_len = dataset_config.max_len
        self.use_padding = dataset_config.use_padding
        self.add_eos = dataset_config.add_eos
        self.rc_aug = dataset_config.rc_aug
        self.randomize_offset = dataset_config.randomize_offset
        self.tokenizer_name = tokenizer_config.name
        assert self.tokenizer_name in ["char", "k_mer"]
        if self.phase == "pretrain":
            self.pretrain_method = dataset_config.pretrain_method
            assert self.pretrain_method in ["mlm", "ntp"]
            if self.pretrain_method == "mlm":
                assert dataset_config.mask_ratio != None
                self.mask_ratio = dataset_config.mask_ratio
        if self.randomize_offset:
            self.offset_range = (
                1 if self.tokenizer_name == "char" else tokenizer_config.k_mer
            )
        self.tokenizer = get_tokenizer(
            tokenizer_name=self.tokenizer_name, tokenizer_config=tokenizer_config
        )
        train_csv = pd.read_csv(self.data_path, sep=",")
        self.barcodes = train_csv["nucleotides"].to_list()
        self.labels = None
        if dataset_config.classify_level == "species":
            self.labels = train_csv["species_name"].to_list()
        elif dataset_config.classify_level == "genus":
            self.labels = train_csv["genus_name"].to_list()
        if self.phase == "finetune":
            self.label_set = sorted(set(self.labels))

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        if self.randomize_offset:
            offset = torch.randint(self.offset_range, (1,)).item()
        else:
            offset = 0
        seq = self.barcodes[idx]
        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)
        if self.tokenizer_name == "char":
            self.tokenizer.pad_token = "N"
            seq = self.tokenizer(
                seq,
                add_special_tokens=False,
                padding="max_length" if self.use_padding else "do_not_pad",
                max_length=self.max_len,
                truncation=True,
            )
            att_mask = seq["attention_mask"]
            seq = seq["input_ids"]
            if self.add_eos:
                seq.append(self.tokenizer.sep_token_id)
        elif self.tokenizer_name == "k_mer":
            seq, att_mask = self.tokenizer(seq, offset=offset)
            if self.add_eos:
                seq.append(
                    self.tokenizer.vocabulary_mapper.get_stoi().get("[SEP]", None)
                )
        processed_barcode = torch.tensor(seq, dtype=torch.int64)
        if self.phase == "pretrain":
            if self.pretrain_method == "ntp":
                data = processed_barcode[:-1].clone()  # remove eos
                target = processed_barcode[1:].clone()  # offset by 1, includes eos
                return data, target, att_mask
            elif self.pretrain_method == "mlm":
                masked_input = processed_barcode.clone()
                random_mask = torch.rand(masked_input.shape)
                input_maskout = random_mask < self.mask_ratio
                masked_input[input_maskout] = 0
                return masked_input, processed_barcode, att_mask
        elif self.phase == "finetune":
            processed_barcode = torch.tensor(seq, dtype=torch.int64)
            label = torch.tensor(
                [self.label_set.index(self.labels[idx])], dtype=torch.int64
            )
            return processed_barcode, label, att_mask


def get_dataloader(config, phase="train"):
    assert phase in ["train", "val", "test"]
    if phase == "train":
        if config.dataset.phase == "pretrain":
            data_path = os.path.join(config.dataset.input_path, "pre_training.csv")
        elif config.dataset.phase == "finetune":
            data_path = os.path.join(config.dataset.input_path, "supervised_train.csv")
    elif phase == "val":
        if config.dataset.phase == "pretrain":
            data_path = os.path.join(config.dataset.input_path, "supervised_train.csv")
        elif config.dataset.phase == "finetune":
            data_path = os.path.join(config.dataset.input_path, "supervised_val.csv")
    elif phase == "test":
        if config.dataset.phase == "pretrain":
            data_path = os.path.join(config.dataset.input_path, "unseen.csv")
        elif config.dataset.phase == "finetune":
            data_path = os.path.join(config.dataset.input_path, "supervised_test.csv")
    dataset = TokenizedDNADataset(data_path, config.dataset, config.tokenizer)
    return DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        pin_memory=False,
        num_workers=config.dataset.num_workers,
        drop_last=False,
        shuffle=False,
    )


def get_probe_dataframe(input_path, phase="linear", split="train"):
    assert phase in ["linear", "knn"] and split in ["train", "test"]
    if split == "train":
        data_path = os.path.join(input_path, "supervised_train.csv")
    else:
        if phase == "linear":
            data_path = os.path.join(input_path, "supervised_test.csv")
        else:
            data_path = os.path.join(input_path, "unseen.csv")
    return pd.read_csv(data_path)


class TokenizedDNADatasetForBaselines(Dataset):

    def __init__(self, data_path, dataset_config, tokenizer):
        self.max_len = dataset_config.max_len
        self.data_path = data_path
        self.tokenizer = tokenizer
        train_csv = pd.read_csv(self.data_path, sep=",")
        self.barcodes = train_csv["nucleotides"].to_list()
        self.labels = None
        if dataset_config.classify_level == "species":
            self.labels = train_csv["species_name"].to_list()
        elif dataset_config.classify_level == "genus":
            self.labels = train_csv["genus_name"].to_list()
        self.label_set = sorted(set(self.labels))

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        seq = self.barcodes[idx]
        if len(seq) > self.max_len:
            seq = seq[: self.max_len]
        else:
            seq = "N" * (self.max_len - len(seq)) + seq
        seq = self.tokenizer(seq)["input_ids"]
        processed_barcode = torch.tensor(seq, dtype=torch.int64)
        label = torch.tensor(
            [self.label_set.index(self.labels[idx])], dtype=torch.int64
        )
        return processed_barcode, label
