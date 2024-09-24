from torch import nn, optim
from functools import partial

from model import LM_MODEL
from utils import Lang, read_file, get_vocab
from utils import PennTreeBank, DataLoader, collate_fn
import argparse
import random

import math
from tqdm import tqdm
import wandb

import os
import torch
import numpy as np
import copy
from pathlib import Path


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


def run_experiments(defaults, experiments, glob_args):

    TRAIN_BS = glob_args["train_batch_size"]
    DEV_BS = glob_args["dev_batch_size"]
    TEST_BS = glob_args["test_batch_size"]
    LOG = not glob_args["no_log"]

    SAVE_PATH = glob_args["save_path"]
    DEVICE = "cuda:0"  # it can be changed with 'cpu' if you do not have a gpu

    DATASET_PATH = "LM/part_1/dataset/PennTreeBank/"

    train_raw = read_file(DATASET_PATH + "ptb.train.txt")
    dev_raw = read_file(DATASET_PATH + "ptb.valid.txt")
    test_raw = read_file(DATASET_PATH + "ptb.test.txt")

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Dataloader instantiation
    print(f"[TRAIN] Using Train BS: {TRAIN_BS}, Dev BS: {DEV_BS}, Test BS: {TEST_BS}")

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

        model = LM_MODEL(
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
            print("[MODEL] Using ", OPT)
            optimizer = optim.SGD(model.parameters(), lr=lr)
        else:
            print("[MODEL] Using AdamW")
            optimizer = optim.AdamW(model.parameters(), lr=lr)

        criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
        criterion_eval = nn.CrossEntropyLoss(
            ignore_index=lang.word2id["<pad>"], reduction="sum"
        )

        # build run folder
        run_name, run_path = build_run_name(
            arch, emb_drop, out_drop, lr, var_drop, tying, OPT, SAVE_PATH
        )
        os.mkdir(run_path)

        print("[TRAIN] Starting ", run_name)

        # start a new wandb run to track this script
        if LOG:
            wandb.init(
                # set the wandb project where this run will be logged
                project="NLU_assignment",
                name=run_name,
                config={
                    "model": str(type(model).__name__),
                    "arch": arch,
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
        SAVE_RATE = 10000000
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1, EPOCHS))

        for epoch in pbar:
            ppl_train, loss = train_loop(
                train_loader, optimizer, criterion_train, model, clip
            )

            if epoch % 1 == 0:

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

                if ppl_dev < best_ppl:  # the lower, the better
                    best_ppl = ppl_dev
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
                    wandb.log(
                        {"ppl_dev": ppl_dev, "ppl_train": ppl_train, "loss": loss_dev}
                    )

            if epoch % SAVE_RATE == 0:
                checkpoint_path = (
                    run_path + "epoch_" + ("0000" + str(epoch))[-4:] + ".pt"
                )
                torch.save(model.state_dict(), checkpoint_path)

            if patience <= 0:  # Early stopping with patience
                if OPT == "ASGD" and "t0" not in optimizer.param_groups[0]:
                    print("[TRAIN] AverageSGD triggered")
                    optimizer = torch.optim.ASGD(
                        model.parameters(), lr=lr, t0=0, lambd=0.0
                    )
                    patience = PAT
                else:
                    break  # Not nice but it keeps the code clean

        checkpoint_path = run_path + "epoch_" + ("0000" + str(epoch))[-4:] + ".pt"
        torch.save(model.state_dict(), checkpoint_path)

        best_model.to(device)
        test_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        checkpoint_path = run_path + "best.pt"
        torch.save(best_model.state_dict(), checkpoint_path)

        print("[TRAIN] Best ppl: ", best_ppl)
        print("[TRAIN] Test ppl: ", test_ppl)
        if LOG:
            wandb.log(
                {
                    "ppl": ppl_dev,
                    "ppl_train": ppl_train,
                    "loss": loss_dev,
                    "test_ppl": test_ppl,
                }
            )
            wandb.finish()


def run_tests(defaults, experiments, glob_args):

    TEST_BS = glob_args["test_batch_size"]

    SAVE_PATH = glob_args["save_path"]
    DEVICE = "cuda:0"  # it can be changed with 'cpu' if you do not have a gpu

    DATASET_PATH = "LM/part_1/dataset/PennTreeBank/"

    train_raw = read_file(DATASET_PATH + "ptb.train.txt")
    test_raw = read_file(DATASET_PATH + "ptb.test.txt")

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    test_dataset = PennTreeBank(test_raw, lang)

    # Dataloader instantiation
    print(f"Using Test BS: {TEST_BS}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BS,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=DEVICE),
    )

    results = []

    for exp in experiments:

        args = defaults | exp

        print("[TEST]", args)

        emb_size = args["emb_size"]
        hid_size = args["hid_size"]
        lr = args["lr"]
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

        model = LM_MODEL(
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

        criterion_eval = nn.CrossEntropyLoss(
            ignore_index=lang.word2id["<pad>"], reduction="sum"
        )

        # build run folder
        run_name, _ = build_run_name(
            arch, emb_drop, out_drop, lr, var_drop, tying, OPT, SAVE_PATH
        )

        run_name = run_name[:-6]
        print("[TEST] Loading model ", run_name)

        paths = sorted(Path(SAVE_PATH).iterdir(), key=os.path.getmtime)

        paths_list = [path.stem for path in paths]

        weights_path = ""
        for path in paths_list[::-1]:
            if run_name == path[:-6]:
                # weights_path = SAVE_PATH + path + "/best.pt"
                if os.path.isfile(SAVE_PATH + path + "/best.pt"):
                    weights_path = SAVE_PATH + path + "/best.pt"
                break

        if weights_path == "":
            print("[TEST] Model not found ", run_name)
            continue

        model.load_state_dict(torch.load(weights_path, weights_only=True))

        model.to(device)
        test_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        results.append([args, test_ppl])

    print("\n\n\nFinal Results: \n")
    print("PPL\tmodel\tdrop\tlr\tOPT\tVD\tWT")
    for args, ppl in results:

        emb_drop = int(args["emb_drop"] * 100)
        out_drop = int(args["out_drop"] * 100)
        print(
            f"{round(ppl, 2)}\t{args['arch']}\t{emb_drop}+{out_drop}\t{args['lr']}\t{args['OPT']}\t{args['var_drop']}\t{args['tying']}"
        )


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


def build_run_name(arch, emb_drop, out_drop, lr, var_drop, tying, OPT, SAVE_PATH):
    run_name = (
        f"{arch}_{int(emb_drop*100)}_{int(out_drop*100)}_{str(lr).replace('.','-')}"
    )

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
    return run_name, run_path


def generate_id(len=5):
    STR_KEY_GEN = "ABCDEFGHIJKLMNOPQRSTUVWXYzabcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(STR_KEY_GEN) for _ in range(len))


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
        "-des",
        "--dev-batch-size",
        type=int,
        help="Set the dev batch size",
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

    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        help="Set checkpoint save path",
        default="LM/part_1/bin/",
        metavar="",
    )

    args = vars(parser.parse_args())
    return args
