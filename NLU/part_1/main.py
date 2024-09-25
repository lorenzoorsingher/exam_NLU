import matplotlib.pyplot as plt
from tqdm import tqdm

from functions import train_loop, eval_loop, init_weights

from utils import get_dataloaders
import os
import numpy as np


from model import ModelIAS
import torch.optim as optim
import torch.nn as nn

DEVICE = "cuda:0"
PAD_TOKEN = 0
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Used to report errors on CUDA side

train_loader, dev_loader, test_loader, lang = get_dataloaders(
    "NLU/part_1/dataset", PAD_TOKEN, DEVICE
)


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
    ).to(DEVICE)
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
