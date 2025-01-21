import json
from pprint import pprint
from collections import Counter
import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from functools import partial
from transformers import AutoTokenizer


class Lang:
    def __init__(self, words, slots, pad_token, cutoff=0):
        self.pad_token = pad_token
        self.slot2id = self.lab2id(slots)
        self.id2slot = {v: k for k, v in self.slot2id.items()}

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab["pad"] = self.pad_token
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab


class BERTSet(data.Dataset):

    def __init__(self, dataset, lang, model_name, unk="unk"):

        self.unk = unk
        self.lang = lang
        # TODO: model name should be a parameter
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.PAD_TOKEN_ID = self.tokenizer.pad_token_id
        self.SLOT_PAD = 0
        self.utterances = []
        self.slots = []

        for x in dataset:
            self.utterances.append(x["utterance"])
            self.slots.append(x["slots"])
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        # idx = 574
        sentence = self.utterances[idx]
        slots = self.slot_ids[idx]

        words = sentence.split()

        # sentence = "what flights leaving pittsburgh arrive in denver and leave after say 6 o'clock at night"

        encoded = self.tokenizer(sentence, return_tensors="pt")
        utt = encoded["input_ids"][0]
        typ = encoded["token_type_ids"][0]
        att = encoded["attention_mask"][0]
        tokens = encoded.tokens()
        word_ids = encoded.word_ids()

        wprev = None
        for i, wid in enumerate(word_ids[1:-1]):
            wstart = encoded.word_to_chars(wid).start
            if sentence[wstart - 1] != " " and wprev != wid and i > 0:
                # print(f"BET {wid} -> {wprev}")
                word_ids[i + 1] = wprev
                encoded.word_to_tokens
            else:
                wprev = wid

        slots_align = []
        wprev = None
        slots_idx = 0
        for i, wid in enumerate(word_ids[1:-1]):
            if wid != wprev:
                slots_align.append(slots[slots_idx])
                slots_idx += 1
                wprev = wid
            else:
                slots_align.append(self.SLOT_PAD)

        slots_align = [self.SLOT_PAD] + slots_align + [self.SLOT_PAD]
        gt_slots = [self.lang.id2slot[elem] for elem in slots_align]
        slots_align = torch.LongTensor(slots_align)
        # for tkn, wid, slot in zip(tokens, word_ids, gt_slots):
        #     print(f"{tkn} \t{wid} \t{slot}")
        # print("\n\n")
        # for tkn, wid, slot in zip(tokens, word_ids, gt_slots):
        #     print(f"{wid} \t{slot} \t{tkn} \t")

        # breakpoint()
        # breakpoint()
        sample = {
            "utt": utt,
            "att": att,
            "words": words,
            "slots": slots_align,
        }

        return sample

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper):  # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res


def collate_fn(data, pad_token, slot_pad, device):
    def merge(sequences, padding):
        """
        merge from batch * sent_len to batch * max_len
        """
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)

        # max_len = max(1, max(lengths))
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(padding)
        # breakpoint()
        for i, seq in enumerate(sequences):
            # breakpoint()
            # seq = torch.LongTensor(seq)
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = (
            padded_seqs.detach()
        )  # We remove these tensors from the computational graph
        # breakpoint()
        return padded_seqs, lengths

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x["utt"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    pad_utt, len_utt = merge(new_item["utt"], pad_token)
    pad_slots, len_slots = merge(new_item["slots"], slot_pad)
    len_slots = torch.LongTensor(len_slots)
    pad_att, len_att = merge(new_item["att"], 0)

    pad_utt = pad_utt.to(device)
    pad_slots = pad_slots.to(device)
    pad_att = pad_att.to(device)
    len_slots = len_slots.to(device)

    new_item["utt"] = pad_utt
    new_item["slots"] = pad_slots
    new_item["att"] = pad_att
    new_item["len_slots"] = len_slots

    return new_item


def get_dataloaders(data_path, pad_token, device, model_name, lang=None, portion=0.10):

    tmp_train_raw = load_data(os.path.join(data_path, "laptop14_train.txt"))
    test_raw = load_data(os.path.join(data_path, "laptop14_test.txt"))
    print("Train samples:", len(tmp_train_raw))
    print("Test samples:", len(test_raw))

    mini_train = []

    X_train, X_dev = train_test_split(
        tmp_train_raw,
        test_size=portion,
        random_state=42,
        shuffle=True,
    )

    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    if lang is None:

        words = sum(
            [x["utterance"].split() for x in train_raw], []
        )  # No set() since we want to compute
        # the cutoff
        corpus = train_raw + dev_raw + test_raw  # We do not wat unk labels,
        # however this depends on the research purpose
        slots = set(sum([line["slots"].split() for line in corpus], []))

        lang = Lang(words, slots, pad_token, cutoff=0)

    # Create our datasets
    train_dataset = BERTSet(train_raw, lang, model_name)
    dev_dataset = BERTSet(dev_raw, lang, model_name)
    test_dataset = BERTSet(test_raw, lang, model_name)

    # Dataloader instantiations
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        collate_fn=partial(
            collate_fn,
            pad_token=train_dataset.PAD_TOKEN_ID,
            slot_pad=train_dataset.SLOT_PAD,
            device=device,
        ),
        shuffle=True,
        # num_workers=8,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=64,
        collate_fn=partial(
            collate_fn,
            pad_token=train_dataset.PAD_TOKEN_ID,
            slot_pad=train_dataset.SLOT_PAD,
            device=device,
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        collate_fn=partial(
            collate_fn,
            pad_token=train_dataset.PAD_TOKEN_ID,
            slot_pad=train_dataset.SLOT_PAD,
            device=device,
        ),
    )

    return train_loader, dev_loader, test_loader, lang


def load_data(path):
    """
    input: path/to/data
    output: json
    """
    dataset = []
    with open(path) as f:
        dataset = f.read().splitlines()

    x = []
    for entry in dataset:
        _, slo = entry.split("####")

        utterance = []
        slot = []
        for sl in slo.split(" "):
            splt = sl.split("=")

            if len(splt) == 2:
                word, label = splt
            else:
                word = "="
                label = splt[-1]

            if label != "O":
                label = "T"

            utterance.append(word)
            slot.append(label)
        x.append({"utterance": " ".join(utterance), "slots": " ".join(slot)})

        # for i, (u, s) in enumerate(zip(utterance, slot)):
        #     print(f"{s} \t{u}")
        # breakpoint()

    # for el in x:
    #     utt = el["utterance"]
    #     slots = el["slots"]

    #     for i, (u, s) in enumerate(zip(utt.split(), slots.split())):

    #         print(f"{s} \t{u}")

    #     print("-------------------------\n")

    # breakpoint()

    return x
