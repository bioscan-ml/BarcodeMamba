"""
K-mer tokenizer from BarcodeBERT
"""

import torch


# Adapted from BarcodeBERT and added padding side
class KmerTokenizer(object):
    def __init__(
        self,
        k,
        vocabulary_mapper,
        stride=1,
        padding=False,
        max_len=660,
        padding_side="right",
    ):
        self.k = k
        self.stride = stride
        self.padding = padding
        self.max_len = max_len
        self.padding_side = padding_side
        assert self.padding_side in ["left", "right"]
        self.vocabulary_mapper = vocabulary_mapper

    def __call__(self, dna_sequence, offset=0) -> (list, list):
        tokens = []
        att_mask = [1] * (self.max_len // self.stride)
        x = dna_sequence[offset:]
        if self.padding:
            if len(x) > self.max_len:
                x = x[: self.max_len]
            else:
                assert self.padding_side in ["left", "right"]
                if self.padding_side == "right":
                    x = x + "N" * (self.max_len - len(x))
                else:
                    x = "N" * (self.max_len - len(x)) + x
                att_mask[len(x) // self.stride :] = [0] * (
                    len(att_mask) - len(x) // self.stride
                )

        for i in range(0, len(x) - self.k + 1, self.stride):
            k_mer = x[i : i + self.k]
            tokens.append(k_mer)

        # debug, returns <list> instead of tensor
        tokens = self.vocabulary_mapper(tokens)
        att_mask = torch.tensor(att_mask, dtype=torch.int32)

        return tokens, att_mask
