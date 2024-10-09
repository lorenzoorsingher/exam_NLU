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
    def __init__(self, words, intents, slots, pad_token, cutoff=0):
        self.pad_token = pad_token
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab["pad"] = self.pad_token
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab


class BERTSet(data.Dataset):

    def __init__(self, dataset, lang, unk="unk"):

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.PAD_TOKEN_ID = self.tokenizer.pad_token_id
        self.SLOT_PAD = 0
        self.utterances = []
        self.intents = []
        self.slots = []

        for x in dataset:
            self.utterances.append(x["utterance"])
            self.slots.append(x["slots"])
            self.intents.append(x["intent"])

        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):

        sentence = self.utterances[idx]
        slots = self.slot_ids[idx]
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        words = sentence.split()

        # sentence = "what flights leaving pittsburgh arrive in denver and leave after say 6 o'clock at night"

        encoded = self.tokenizer(sentence, return_tensors="pt")
        utt = encoded["input_ids"][0]
        typ = encoded["token_type_ids"][0]
        att = encoded["attention_mask"][0]
        tokens = encoded.tokens()
        word_ids = encoded.word_ids()

        mapping2 = []
        wprev = None
        for i, wid in enumerate(word_ids[1:-1]):
            wstart = encoded.word_to_chars(wid).start
            if sentence[wstart - 1] != " " and wprev != wid and i > 0:

                # print(f"BET {wid} -> {wprev}")
                word_ids[i + 1] = wprev
                encoded.word_to_tokens
            else:
                wprev = wid
            if word_ids[i + 1] != wprev:
                mapping2.append(i)

        mapping = []
        wprev = None
        for i, wid in enumerate(word_ids[1:-1]):
            if wid != wprev:
                mapping.append(i)
            wprev = wid
        mapping = torch.Tensor(mapping)
        # check
        if len(mapping) != len(slots):
            print("ERROR")
            exit()

        sample = {
            "utt": utt,
            "att": att,
            "map": mapping,
            "words": words,
            "slots": slots,
            "intent": intent,
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

    # We just need one length for packed pad seq, since len(utt) == len(slots)
    # src_utt, _ = merge(new_item["utterance"])
    # y_slots, y_lengths = merge(new_item["slots"])
    # intent = torch.LongTensor(new_item["intent"])
    pad_utt, len_utt = merge(new_item["utt"], pad_token)
    pad_slots, len_slots = merge(new_item["slots"], slot_pad)
    len_slots = torch.LongTensor(len_slots)
    pad_att, len_att = merge(new_item["att"], 0)
    pad_map, len_map = merge(new_item["map"], 0)
    intent = torch.LongTensor(new_item["intent"])

    pad_utt = pad_utt.to(device)
    pad_slots = pad_slots.to(device)
    pad_map = pad_map.to(device)
    pad_att = pad_att.to(device)
    intent = intent.to(device)
    len_slots = len_slots.to(device)

    new_item["utt"] = pad_utt
    new_item["slots"] = pad_slots
    new_item["map"] = pad_map
    new_item["att"] = pad_att
    new_item["intent"] = intent
    new_item["len_slots"] = len_slots

    # breakpoint()
    return new_item


def get_dataloaders(data_path, pad_token, device, portion=0.10):
    tmp_train_raw = load_data(os.path.join(data_path, "ATIS", "train.json"))
    test_raw = load_data(os.path.join(data_path, "ATIS", "test.json"))
    print("Train samples:", len(tmp_train_raw))
    print("Test samples:", len(test_raw))

    intents = [x["intent"] for x in tmp_train_raw]  # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:  # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    # Random Stratify
    X_train, X_dev, _, _ = train_test_split(
        inputs,
        labels,
        test_size=portion,
        random_state=42,
        shuffle=True,
        stratify=labels,
    )
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    words = sum(
        [x["utterance"].split() for x in train_raw], []
    )  # No set() since we want to compute
    # the cutoff
    corpus = train_raw + dev_raw + test_raw  # We do not wat unk labels,
    # however this depends on the research purpose
    slots = set(sum([line["slots"].split() for line in corpus], []))
    intents = set([line["intent"] for line in corpus])

    lang = Lang(words, intents, slots, pad_token, cutoff=0)

    # Create our datasets
    train_dataset = BERTSet(train_raw, lang)
    dev_dataset = BERTSet(dev_raw, lang)
    test_dataset = BERTSet(test_raw, lang)

    # Dataloader instantiations
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        collate_fn=partial(
            collate_fn,
            pad_token=train_dataset.PAD_TOKEN_ID,
            slot_pad=train_dataset.SLOT_PAD,
            device=device,
        ),
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=16,
        collate_fn=partial(
            collate_fn,
            pad_token=train_dataset.PAD_TOKEN_ID,
            slot_pad=train_dataset.SLOT_PAD,
            device=device,
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
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
        dataset = json.loads(f.read())
    return dataset
