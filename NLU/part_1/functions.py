import os
import argparse
import json
import torch
import numpy as np

from torch import nn
from torch import optim
from model import ModelIAS
from tqdm import tqdm
from conll import evaluate
from sklearn.metrics import classification_report

from utils import get_dataloaders


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        slots, intent = model(sample["utterances"], sample["slots_len"])
        loss_intent = criterion_intents(intent, sample["intents"])
        loss_slot = criterion_slots(slots, sample["y_slots"])
        loss = loss_intent + loss_slot  # In joint training we sum the losses.
        # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample["utterances"], sample["slots_len"])
            loss_intent = criterion_intents(intents, sample["intents"])
            loss_slot = criterion_slots(slots, sample["y_slots"])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [
                lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()
            ]
            gt_intents = [lang.id2intent[x] for x in sample["intents"].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample["slots_len"].tolist()[id_seq]
                utt_ids = sample["utterance"][id_seq][:length].tolist()
                gt_ids = sample["y_slots"][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append(
                    [(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)]
                )
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total": {"f": 0}}

    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=False, output_dict=True
    )
    return results, report_intent, loss_array


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        nn.init.xavier_uniform_(param[idx * mul : (idx + 1) * mul])
                elif "weight_hh" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        nn.init.orthogonal_(param[idx * mul : (idx + 1) * mul])
                elif "bias" in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


def run_experiments():
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


def get_args():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="""Get the params""",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training",
        default=False,
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests",
        default=False,
    )

    parser.add_argument(
        "-j",
        "--json",
        type=str,
        help="Load params from json",
        default="",
    )

    parser.add_argument(
        "-NL",
        "--no-log",
        action="store_true",
        help="Do not log run",
    )

    parser.add_argument(
        "-trs",
        "--train-batch-size",
        type=int,
        help="Set the train batch size",
        default=256,
        metavar="",
    )

    parser.add_argument(
        "-tes",
        "--test-batch-size",
        type=int,
        help="Set the test batch size",
        default=1024,
        metavar="",
    )

    parser.add_argument(
        "-vas",
        "--val-batch-size",
        type=int,
        help="Set the val batch size",
        default=1024,
        metavar="",
    )

    parser.add_argument(
        "-w",
        "--wandb-secret",
        type=str,
        help="Set wandb secret",
        default="",
        metavar="",
    )

    args = vars(parser.parse_args())
    return args


def load_experiments(json_path):
    print("loading from json...")
    if os.path.exists(json_path):
        filename = json_path.split("/")[-1]
        print("loading from ", filename)
        defaults, experiments = json.load(open(json_path))
    else:
        print("json not found, exiting...")
        exit()
    return defaults, experiments
