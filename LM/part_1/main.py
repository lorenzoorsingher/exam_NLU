import math
import os
import copy

import torch
import wandb
import json
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv


from time import time, localtime, strftime
from torch import nn, optim
from functools import partial

from model import LM_LSTM_TWO
from utils import Lang, read_file, get_vocab
from utils import PennTreeBank, DataLoader, collate_fn
from functions import get_args, generate_id


def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:

        optimizer.zero_grad()  # Zeroing the gradient
        output = model(sample["source"])
        loss = criterion(output, sample["target"])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample["source"])
            loss = eval_criterion(output, sample["target"])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(
                            param[idx * mul : (idx + 1) * mul]
                        )
                elif "weight_hh" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul : (idx + 1) * mul])
                elif "bias" in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


load_dotenv()
glob_args = get_args()

if glob_args["json"] == "":
    FROM_JSON = False
else:
    FROM_JSON = True
    json_path = glob_args["json"]

TRAIN_BS = glob_args["train_batch_size"]
DEV_BS = glob_args["dev_batch_size"]
TEST_BS = glob_args["test_batch_size"]
LOG = not glob_args["no_log"]
WANDB_SECRET = glob_args["wandb_secret"]
SAVE_PATH = glob_args["save_path"]


if LOG:
    if WANDB_SECRET == "":
        WANDB_SECRET = os.getenv("WANDB_SECRET")
    wandb.login(key=WANDB_SECRET)


DEVICE = "cuda:0"  # it can be changed with 'cpu' if you do not have a gpu

DATASET_PATH = "LM/part_1/dataset/PennTreeBank/"

train_raw = read_file(DATASET_PATH + "ptb.train.txt")
dev_raw = read_file(DATASET_PATH + "ptb.valid.txt")
test_raw = read_file(DATASET_PATH + "ptb.test.txt")

# Vocab is computed only on training set
# We add two special tokens end of sentence and padding
vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

lang = Lang(train_raw, ["<pad>", "<eos>"])

train_dataset = PennTreeBank(train_raw, lang)
dev_dataset = PennTreeBank(dev_raw, lang)
test_dataset = PennTreeBank(test_raw, lang)

# Dataloader instantiation
print(f"Using Train BS: {TRAIN_BS}, Dev BS: {DEV_BS}, Test BS: {TEST_BS}")

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BS,
    collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=DEVICE),
    shuffle=True,
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=DEV_BS,
    collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=DEVICE),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=TEST_BS,
    collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=DEVICE),
)


if FROM_JSON:
    print("loading from json...")
    if os.path.exists(json_path):
        filename = json_path.split("/")[-1]
        print("loading from ", filename)
        defaults, experiments = json.load(open(json_path))
    else:
        print("json not found, exiting...")
        exit()
else:
    defaults = {
        "emb_size": 300,
        "hid_size": 300,
        "lr": 1.5,
        "clip": 5,
        "n_layers": 1,
        "emb_drop": 0.1,
        "out_drop": 0.1,
        "tying": False,
        "var_drop": False,
        "EPOCHS": 99,
        "OPT": "SGD",
        "tag": "general",
        "arch": "LSTM",
    }
    experiments = [
        {
            "emb_drop": 0.5,
            "out_drop": 0.5,
            "var_drop": True,
            "tag": "general",
        }
    ]


for exp in experiments:

    args = defaults | exp

    print(args)

    emb_size = args["emb_size"]
    hid_size = args["hid_size"]
    lr = args["lr"]
    clip = args["clip"]
    n_layers = args["n_layers"]
    emb_drop = args["emb_drop"]
    out_drop = args["out_drop"]
    tying = args["tying"]
    var_drop = args["var_drop"]
    OPT = args["OPT"]
    tag = args["tag"]
    arch = args["arch"]
    device = "cuda:0"

    vocab_len = len(lang.word2id)

    model = LM_LSTM_TWO(
        emb_size,
        hid_size,
        vocab_len,
        tie=tying,
        out_dropout=out_drop,
        emb_dropout=emb_drop,
        n_layers=n_layers,
        var_drop=var_drop,
        pad_index=lang.word2id["<pad>"],
        arch=arch,
    ).to(device)

    model.apply(init_weights)

    if OPT == "SGD" or OPT == "ASGD":
        print("Using ", OPT)
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        print("Using AdamW")
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )

    # build run folder

    run_name = f"{emb_size}_{hid_size}_{int(emb_drop*100)}_{int(out_drop*100)}_{str(lr).replace('.','-')}"

    if var_drop:
        run_name += "_VD"
    if tying:
        run_name += "_TIE"
    run_name += "_" + str(OPT)

    run_name += "_" + generate_id(5)
    run_path = SAVE_PATH + run_name + "/"

    if os.path.exists(run_path):
        while os.path.exists(run_path):
            run_name += "_" + generate_id(5)
            run_path = SAVE_PATH + run_name + "/"
    print("starting ", run_name)
    os.mkdir(run_path)

    # start a new wandb run to track this script
    if LOG:
        wandb.init(
            # set the wandb project where this run will be logged
            project="NLU_assignment",
            name=run_name,
            config={
                "model": str(type(model).__name__),
                "lr": lr,
                "optim": str(type(optimizer).__name__),
                "clip": clip,
                "hid_size": hid_size,
                "emb_size": emb_size,
                "layers": n_layers,
                "tie": tying,
                "var_drop": var_drop,
                "dropout": [emb_drop, out_drop],
                "tag": tag,
            },
        )

    EPOCHS = args["EPOCHS"]
    PAT = args["PAT"]
    SAVE_RATE = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    best_epoch = -1
    pbar = tqdm(range(1, EPOCHS))

    for epoch in pbar:
        ppl_train, loss = train_loop(
            train_loader, optimizer, criterion_train, model, clip
        )

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            if "t0" in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]["ax"].clone()

                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()

            else:
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())

            if ppl_dev < best_ppl:  # the lower, the better
                best_ppl = ppl_dev
                best_epoch = epoch
                patience = PAT
                best_model = copy.deepcopy(model).to("cpu")
            else:
                patience -= 1

            pbar.set_description(
                "PPL: "
                + str(round(ppl_dev, 2))
                + " best: "
                + str(round(best_ppl, 2))
                + " P: "
                + str(patience)
            )
            if LOG:
                wandb.log({"ppl": ppl_dev, "ppl_train": ppl_train, "loss": loss_dev})

        if epoch % SAVE_RATE == 0:
            checkpoint_path = run_path + "epoch_" + ("0000" + str(epoch))[-4:] + ".pt"
            torch.save(model.state_dict(), checkpoint_path)

        if patience <= 0:  # Early stopping with patience
            if OPT == "ASGD" and "t0" not in optimizer.param_groups[0]:
                print("AverageSGD triggered")
                optimizer = torch.optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0.0)
                patience = PAT
            else:
                break  # Not nice but it keeps the code clean

    checkpoint_path = run_path + "epoch_" + ("0000" + str(epoch))[-4:] + ".pt"
    torch.save(model.state_dict(), checkpoint_path)

    best_model.to(device)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)

    print("Best ppl: ", best_ppl)
    print("Test ppl: ", final_ppl)
    if LOG:
        wandb.finish()
