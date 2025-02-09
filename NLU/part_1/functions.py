from copy import deepcopy
import os
import argparse
import json
from pathlib import Path
import torch
import wandb
import random
import numpy as np

from torch import nn
from torch import optim
from model import ModelIAS
from tqdm import tqdm
from conll import evaluate
from sklearn.metrics import classification_report
from tabulate import tabulate

from utils import get_dataloaders


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    """
    Function to train the model

    Parameters:
    - data (DataLoader): DataLoader object
    - optimizer (optim): optimizer
    - criterion_slots (nn.CrossEntropyLoss): loss function for slots
    - criterion_intents (nn.CrossEntropyLoss): loss function for intents
    - model (nn.Module): model
    - clip (int): clipping value
    """

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

    avg_loss = np.array(loss_array).mean()
    return loss_array, avg_loss


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    """
    Function to evaluate the model

    Parameters:
    - data (DataLoader): DataLoader object
    - criterion_slots (torch.nn): loss function for slots
    - criterion_intents (torch.nn): loss function for intents
    - model (nn.Module): model
    - lang (Lang): language object

    Returns:
    - results (dict): results from the evaluation
    - report_intent (dict): intent classification report
    - loss_array (list): list with the losses
    - loss_avg (float): average loss
    """

    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:

            # compute output
            slots, intents = model(sample["utterances"], sample["slots_len"])

            # compute losses
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

            # For each sequence in the batch we need to decode the output
            # and the ground truth
            for id_seq, seq in enumerate(output_slots):
                length = sample["slots_len"].tolist()[id_seq]
                utt_ids = sample["utterance"][id_seq][:length].tolist()
                gt_ids = sample["y_slots"][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()

                # recover words and slots for GT e prediction
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
        print("Class is not in REF")
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        results = {"total": {"f": 0}}

    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=False, output_dict=True
    )

    loss_avg = np.array(loss_array).mean()

    return results, report_intent, loss_array, loss_avg


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


def run_experiments(defaults, experiments, glob_args):
    """
    Function to run the experiments, manage the training and the logging

    Parameters:
    - defaults (dict): default parameters
    - experiments (list): list of experiments
    - glob_args (dict): global arguments
    """

    LOG = glob_args["log"]
    SAVE_PATH = glob_args["save_path"]
    DEVICE = "cuda:0"  # it can be changed with 'cpu' if you do not have a gpu
    DATASET_PATH = "NLU/part_1/dataset"
    PAD_TOKEN = 0

    train_loader, dev_loader, test_loader, lang = get_dataloaders(
        DATASET_PATH, PAD_TOKEN, DEVICE
    )

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    for exp in experiments:

        # prepare experiment params
        args = defaults | exp

        print(args)

        emb_size = args["emb_size"]
        hid_size = args["hid_size"]
        lr = args["lr"]
        OPT = args["OPT"]
        drop = args["drop"]
        var_drop = args["var_drop"]
        bi = args["bi"]

        runs = args["runs"]
        tag = args["tag"]

        run_name, run_path = build_run_name(args, SAVE_PATH)

        os.mkdir(run_path)

        EPOCHS = args["EPOCHS"]
        PAT = args["PAT"]
        # SAVE_RATE = 10000000

        slot_f1s, intent_acc = [], []

        print("[TRAIN] Starting ", run_name)

        pbar_runs = tqdm(range(1, runs + 1))
        for run_n in pbar_runs:

            model = ModelIAS(
                hid_size,
                out_slot,
                out_int,
                emb_size,
                vocab_len,
                pad_index=PAD_TOKEN,
                var_drop=var_drop,
                dropout=drop,
                bi=bi,
            ).to(DEVICE)

            model.apply(init_weights)

            if OPT == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=lr)
            elif OPT == "AdamW":
                optimizer = optim.AdamW(model.parameters(), lr=lr)

            criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
            criterion_intents = nn.CrossEntropyLoss()

            if LOG:
                wandb.init(
                    project="NLU_assignment2",
                    name=run_name + "_" + str(run_n),
                    config={
                        "model": str(type(model).__name__),
                        "lr": lr,
                        "optim": str(type(optimizer).__name__),
                        "hid_size": hid_size,
                        "emb_size": emb_size,
                        "drop": drop,
                        "var_drop": var_drop,
                        "tag": tag,
                        "runset": run_name,
                        "run": run_n,
                    },
                )

            best_f1 = 0
            patience = PAT
            pbar_epochs = tqdm(range(0, EPOCHS))

            best_model = None
            for epoch in pbar_epochs:
                loss, avg_loss = train_loop(
                    train_loader, optimizer, criterion_slots, criterion_intents, model
                )
                results_dev, intent_res, loss_dev, avg_loss_dev = eval_loop(
                    dev_loader, criterion_slots, criterion_intents, model, lang
                )
                f1 = results_dev["total"]["f"]

                if f1 > best_f1:
                    best_f1 = f1
                    patience = PAT
                    best_model = deepcopy(model)
                else:
                    if epoch > 10:  # make sure early stopping
                        patience -= 1

                pbar_epochs.set_description(
                    f"F1 {round_sf(f1,2)} best F1 {round_sf(best_f1,2)} PAT {patience}"
                )

                if LOG:
                    wandb.log(
                        {
                            "loss": avg_loss,
                            "val loss": avg_loss_dev,
                            "F1": f1,
                            "acc": intent_res["accuracy"],
                        }
                    )

                if patience < 0:  # Early stopping with patient
                    break

            # load the best model
            model.load_state_dict(best_model.state_dict())

            # run the evaluation on the val set
            results_dev, intent_res, _, avg_loss_dev = eval_loop(
                dev_loader, criterion_slots, criterion_intents, model, lang
            )

            # run the evaluation on the test set
            results_test, intent_test, _, avg_loss_test = eval_loop(
                test_loader, criterion_slots, criterion_intents, model, lang
            )
            intent_acc.append(intent_test["accuracy"])
            slot_f1s.append(results_test["total"]["f"])

            if LOG:

                loss_gap = (avg_loss_test - avg_loss_dev) / avg_loss_test

                wandb.log(
                    {
                        "val loss": avg_loss_dev,
                        "F1": results_dev["total"]["f"],
                        "acc": intent_res["accuracy"],
                        "test F1": results_test["total"]["f"],
                        "test acc": intent_test["accuracy"],
                        "loss_gap": loss_gap,
                    }
                )

                # on last run log the mean and std of the results
                if run_n == runs - 1:

                    slot_f1s_mean, slot_f1s_std, intent_acc_mean, intent_acc_std = (
                        remove_outliers(slot_f1s, intent_acc)
                    )

                    wandb.log(
                        {
                            "slot_f1s_mean": slot_f1s_mean,
                            "slot_f1s_std": slot_f1s_std,
                            "intent_acc_mean": intent_acc_mean,
                            "intent_acc_std": intent_acc_std,
                        }
                    )

                wandb.finish()

            model_savefile = {
                "state_dict": model.state_dict(),
                "lang": lang,
            }

            torch.save(model_savefile, run_path + "best.pt")

        slot_f1s = np.asarray(slot_f1s)
        intent_acc = np.asarray(intent_acc)

        print("Slot F1", round(slot_f1s.mean(), 3), "+-", round(slot_f1s.std(), 3))
        print("Intent Acc", round(intent_acc.mean(), 3), "+-", round(slot_f1s.std(), 3))


def run_tests(defaults, experiments, glob_args):
    """
    Function to run the tests

    Parameters:
    - defaults (dict): default parameters
    - experiments (list): list of experiments
    - glob_args (dict): global arguments
    """

    SAVE_PATH = glob_args["save_path"]
    DEVICE = "cuda:0"  # it can be changed with 'cpu' if you do not have a gpu
    DATASET_PATH = "NLU/part_1/dataset"
    PAD_TOKEN = 0

    results = []

    for exp in experiments:

        # prepare experiment params
        args = defaults | exp

        emb_size = args["emb_size"]
        hid_size = args["hid_size"]
        drop = args["drop"]
        var_drop = args["var_drop"]
        bi = args["bi"]

        run_name, run_path = build_run_name(args, SAVE_PATH)
        run_name = run_name[:-5]

        # retrieving model weights
        print("[TEST] Loading model ", run_name)
        paths = sorted(Path(SAVE_PATH).iterdir(), key=os.path.getmtime)
        paths_list = [path.stem for path in paths]
        weights_path = ""
        for path in paths_list[::-1]:
            if run_name == path[:-5]:
                # weights_path = SAVE_PATH + path + "/best.pt"
                if os.path.isfile(SAVE_PATH + path + "/best.pt"):
                    weights_path = SAVE_PATH + path + "/best.pt"
                    print("[TEST] Loading ", weights_path)
                break

        if weights_path == "":
            print("[TEST] Model not found ", run_name)
            continue

        # load the model, lang and test set
        model_savefile = torch.load(weights_path, weights_only=False)
        state_dict = model_savefile["state_dict"]
        lang = model_savefile["lang"]

        _, _, test_loader, _ = get_dataloaders(
            DATASET_PATH, PAD_TOKEN, DEVICE, lang=lang
        )

        out_slot = len(lang.slot2id)
        out_int = len(lang.intent2id)
        vocab_len = len(lang.word2id)
        model = ModelIAS(
            hid_size,
            out_slot,
            out_int,
            emb_size,
            vocab_len,
            pad_index=PAD_TOKEN,
            var_drop=var_drop,
            dropout=drop,
            bi=bi,
        ).to(DEVICE)

        # load the weights
        model.load_state_dict(state_dict)
        model.to(DEVICE)

        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()

        results_test, intent_test, _, avg_loss_test = eval_loop(
            test_loader, criterion_slots, criterion_intents, model, lang
        )

        results.append(
            {
                "args": args,
                "f1": results_test["total"]["f"],
                "acc": intent_test["accuracy"],
                "name": run_name,
            }
        )

    # print the results in a table
    headers = ["Name", "LR", "Drop", "VD", "Bi", "Acc", "F1"]
    data = []
    for res in results:
        args = res["args"]
        f1 = res["f1"]
        acc = res["acc"]
        vd = "x" if args["var_drop"] else ""
        bi = "x" if args["bi"] else ""

        data.append(
            [
                res["name"],
                args["lr"],
                args["drop"],
                vd,
                bi,
                round(acc, 3),
                round(f1, 3),
            ]
        )

    print("\n\n")
    print(tabulate(data, headers=headers, tablefmt="orgtbl"))


############################################################################################################
# Functions to handle the experiments


def get_args():
    """
    Function to get the arguments from the command line

    Returns:
    - args (dict): arguments
    """
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
        "-L",
        "--log",
        type=bool,
        help="Log the run",
        default=False,
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
        default="NLU/part_1/bin/",
        metavar="",
    )

    args = vars(parser.parse_args())
    return args


def build_run_name(args, SAVE_PATH):
    """
    Function to build the run name and the path

    Parameters:
    - args (dict): arguments
    - SAVE_PATH (str): path to save the model

    Returns:
    - run_name (str): name of the run
    - run_path (str): path to save the model
    """

    run_name = "NLU"

    run_name += "_" + str(args["lr"])[2:] + "_" + str(round(args["drop"] * 100))

    if args["var_drop"]:
        run_name += "_VD"

    if args["bi"]:
        run_name += "_B"

    run_name += "_" + generate_id(4)
    run_path = SAVE_PATH + run_name + "/"

    if os.path.exists(run_path):
        while os.path.exists(run_path):
            run_name += "_" + generate_id(5)
            run_path = SAVE_PATH + run_name + "/"
    return run_name, run_path


def generate_id(len=5):
    """
    Function to generate a random id for the runs
    """
    STR_KEY_GEN = "ABCDEFGHIJKLMNOPQRSTUVWXYzabcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(STR_KEY_GEN) for _ in range(len))


def round_sf(number, significant=2):
    """
    Function to round a number to a certain number of significant figures

    Parameters:
    - number (float): number to round
    - significant (int): number of significant figures

    Returns:
    - rounded number (float)
    """
    return round(number, max(significant, significant - len(str(number))))


def load_experiments(json_path):
    """
    Function to load the experiments from a json file

    Parameters:
    - json_path (str): path to the json file

    Returns:
    - defaults (dict): default parameters
    - experiments (list): list of experiments
    """

    print("loading from json...")
    if os.path.exists(json_path):
        filename = json_path.split("/")[-1]
        print("loading from ", filename)
        defaults, experiments = json.load(open(json_path))
    else:
        print("json not found, exiting...")
        exit()
    return defaults, experiments


def remove_outliers(slot_f1s, intent_acc):
    """
    Some runs might end too early or too late, leading to outliers.
    This function removes the outliers from the results using the z-score

    Parameters:
    - slot_f1s (list): list of slot f1 scores
    - intent_acc (list): list of intent accuracies

    Returns:
    - clean_slot_f1s_mean (float): mean of the slot f1 scores
    - clean_slot_f1s_std (float): standard deviation of the slot f1 scores
    - clean_intent_acc_mean (float): mean of the intent accuracies
    - clean_intent_acc_std (float): standard deviation of the intent accuracies
    """
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)

    slot_f1s_mean = round(slot_f1s.mean(), 3)

    slot_f1s_std = round(slot_f1s.std(), 3)

    clean_slot_f1s = []
    clean_intent_acc = []
    for idx, (score, acc) in enumerate(zip(slot_f1s, intent_acc)):
        z = abs((score - slot_f1s_mean) / slot_f1s_std)
        if z < 3:
            clean_slot_f1s.append(score)
            clean_intent_acc.append(acc)
        else:
            print("Removing outlier ", idx + 1)
    clean_slot_f1s = np.asarray(clean_slot_f1s)
    clean_intent_acc = np.asarray(clean_intent_acc)

    clean_slot_f1s_mean = round(clean_slot_f1s.mean(), 3)
    clean_slot_f1s_std = round(clean_slot_f1s.std(), 3)

    clean_intent_acc_mean = round(clean_intent_acc.mean(), 3)
    clean_intent_acc_std = round(clean_intent_acc.std(), 3)

    return (
        clean_slot_f1s_mean,
        clean_slot_f1s_std,
        clean_intent_acc_mean,
        clean_intent_acc_std,
    )
