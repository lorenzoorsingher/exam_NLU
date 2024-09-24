import matplotlib.pyplot as plt
from tqdm import tqdm

from functions import train_loop, eval_loop, init_weights

from utils import collate_fn, load_data, IntentsAndSlots, Lang
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader

from model import ModelIAS
import torch.optim as optim
import torch.nn as nn

device = (
    "cuda:0"  # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Used to report errors on CUDA side
PAD_TOKEN = 0

tmp_train_raw = load_data(os.path.join("NLU/part_1/dataset", "ATIS", "train.json"))
test_raw = load_data(os.path.join("NLU/part_1/dataset", "ATIS", "test.json"))
print("Train samples:", len(tmp_train_raw))
print("Test samples:", len(test_raw))


portion = 0.10

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
X_train, X_dev, y_train, y_dev = train_test_split(
    inputs, labels, test_size=portion, random_state=42, shuffle=True, stratify=labels
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

lang = Lang(words, intents, slots, cutoff=0)

# Create our datasets
train_dataset = IntentsAndSlots(train_raw, lang)
dev_dataset = IntentsAndSlots(dev_raw, lang)
test_dataset = IntentsAndSlots(test_raw, lang)

# Dataloader instantiations
train_loader = DataLoader(
    train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True
)
dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

hid_size = 200
emb_size = 300

lr = 0.0001  # learning rate
clip = 5  # Clip the gradient


out_slot = len(lang.slot2id)
out_int = len(lang.intent2id)
vocab_len = len(lang.word2id)

n_epochs = 200
runs = 5

slot_f1s, intent_acc = [], []
for x in tqdm(range(0, runs)):
    model = ModelIAS(
        hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN
    ).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    for x in range(1, n_epochs):
        loss = train_loop(
            train_loader, optimizer, criterion_slots, criterion_intents, model
        )
        if x % 5 == 0:
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(
                dev_loader, criterion_slots, criterion_intents, model, lang
            )
            losses_dev.append(np.asarray(loss_dev).mean())
            f1 = results_dev["total"]["f"]

            if f1 > best_f1:
                best_f1 = f1
            else:
                patience -= 1
            if patience <= 0:  # Early stopping with patient
                break  # Not nice but it keeps the code clean

    results_test, intent_test, _ = eval_loop(
        test_loader, criterion_slots, criterion_intents, model, lang
    )
    intent_acc.append(intent_test["accuracy"])
    slot_f1s.append(results_test["total"]["f"])
slot_f1s = np.asarray(slot_f1s)
intent_acc = np.asarray(intent_acc)
print("Slot F1", round(slot_f1s.mean(), 3), "+-", round(slot_f1s.std(), 3))
print("Intent Acc", round(intent_acc.mean(), 3), "+-", round(slot_f1s.std(), 3))
