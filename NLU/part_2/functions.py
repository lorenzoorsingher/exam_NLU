from copy import deepcopy
import os
import argparse
import json
from pathlib import Path
from tabulate import tabulate
import torch
import wandb
import random
import numpy as np

from torch import nn
from torch import optim
from tqdm import tqdm
from conll import evaluate
from sklearn.metrics import classification_report

from utils import get_dataloaders

from model import MyBert


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

        utt_x = sample["utt"]
        slots_y = sample["slots"]
        att = sample["att"]
        intent_y = sample["intent"]

        # breakpoint()
        optimizer.zero_grad()  # Zeroing the gradient
        # breakpoint()q
        intent, slots = model(utt_x, att)

        loss_intent = criterion_intents(intent, intent_y)
        loss_slot = criterion_slots(slots, slots_y)
        loss = loss_intent + loss_slot  # In joint training we sum the losses.
        # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward()  # Compute the gradient, deleting the computational graph

        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()  # Update the weights

    avg_loss = np.array(loss_array).mean()
    return loss_array, avg_loss


def train_loop_sch(
    data, optimizer, scheduler, criterion_slots, criterion_intents, model, clip=5
):
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

        utt_x = sample["utt"]
        slots_y = sample["slots"]
        att = sample["att"]
        intent_y = sample["intent"]

        # breakpoint()
        optimizer.zero_grad()  # Zeroing the gradient
        intent, slots = model(utt_x, att)

        loss_intent = criterion_intents(intent, intent_y)
        loss_slot = criterion_slots(slots, slots_y)
        loss = loss_intent + loss_slot  # In joint training we sum the losses.
        # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward()  # Compute the gradient, deleting the computational graph

        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights

    if type(scheduler).__name__ == "ReduceLROnPlateau":
        scheduler.step(loss)
    else:
        scheduler.step()  # Update the optimizer

    avg_loss = np.array(loss_array).mean()
    return loss_array, avg_loss


def train_loop_sch_in(
    data, optimizer, scheduler, criterion_slots, criterion_intents, model, clip=5
):
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

        utt_x = sample["utt"]
        slots_y = sample["slots"]
        att = sample["att"]
        intent_y = sample["intent"]

        # breakpoint()
        optimizer.zero_grad()  # Zeroing the gradient
        intent, slots = model(utt_x, att)

        loss_intent = criterion_intents(intent, intent_y)
        loss_slot = criterion_slots(slots, slots_y)
        loss = loss_intent + loss_slot  # In joint training we sum the losses.
        # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward()  # Compute the gradient, deleting the computational graph

        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights

    if type(scheduler).__name__ == "ReduceLROnPlateau":
        scheduler.step(loss)
    else:
        scheduler.step()  # Update the optimizer

    avg_loss = np.array(loss_array).mean()
    return loss_array, avg_loss


def intent_inference(intent, intent_y, lang):
    """
    Prepare the intent inference for the evaluation

    Parameters:
    - intent (torch.tensor): output of the model
    - intent_y (torch.tensor): ground truth
    - lang (Lang): language object

    Returns:
    - gt_intents (list): ground truth intents
    - pred_intents (list): predicted intents
    """

    out_intents = torch.argmax(intent, dim=1).tolist()
    pred_intents = [lang.id2intent[x] for x in out_intents]
    gt_intents = [lang.id2intent[x] for x in intent_y.tolist()]
    return gt_intents, pred_intents


def slot_inference(out_slot, gt_ids, sentence, length, lang):
    """
    Prepare the slot inference for the evaluation

    Parameters:
    - out_slot (torch.tensor): output of the model
    - gt_ids (list): ground truth
    - sentence (list): sentence word list
    - length (int): length of the sentence
    - lang (Lang): language object

    Returns:
    - hyp_slot (list): predicted slots
    - ref_slot (list): ground truth slots
    """

    out_slot = out_slot[1 : length - 1]
    gt_ids = gt_ids[1 : length - 1]

    hyp_slot = []
    ref_slot = []
    slot_idx = 0
    for pred_id, gt_id in zip(out_slot, gt_ids):
        if gt_id != lang.pad_token:
            word = sentence[slot_idx]
            hyp_slot.append((word, lang.id2slot[pred_id.item()]))
            ref_slot.append((word, lang.id2slot[gt_id]))
            slot_idx += 1
    return hyp_slot, ref_slot


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

            utt_x = sample["utt"]
            words = sample["words"]
            slots_y = sample["slots"]
            att = sample["att"]
            intent_y = sample["intent"]
            lens = sample["len_slots"]

            intent, slots = model(utt_x, att)

            loss_intent = criterion_intents(intent, intent_y)
            loss_slot = criterion_slots(slots, slots_y)
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())

            # Intent inference
            gt_intents, pred_intents = intent_inference(intent, intent_y, lang)
            ref_intents.extend(gt_intents)
            hyp_intents.extend(pred_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)

            for i, out_slot in enumerate(output_slots):
                sent = words[i]
                length = int(lens.tolist()[i])
                gt_ids = slots_y[i].tolist()
                hyp_slot, ref_slot = slot_inference(
                    out_slot, gt_ids, sent, length, lang
                )
                hyp_slots.append(hyp_slot)
                ref_slots.append(ref_slot)
            # for a, b in zip(ref_slots, hyp_slots):
            #     print(f"REF: {a}\t HYP: {b}")

    try:
        # breakpoint()
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        # print("Warning:", ex)
        print("Class is not in REF")
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        # print(hyp_s.difference(ref_s))
        results = {"total": {"f": 0}}
    # breakpoint()
    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=False, output_dict=True
    )

    loss_avg = np.array(loss_array).mean()

    return results, report_intent, loss_array, loss_avg


def run_experiments(defaults, experiments, glob_args):
    """
    Function to run the experiments, manage the training and the logging

    Parameters:
    - defaults (dict): default parameters
    - experiments (list): list of experiments
    - glob_args (dict): global arguments
    """

    LOG = not glob_args["no_log"]
    SAVE_PATH = glob_args["save_path"]
    DEVICE = "cuda:0"  # it can be changed with 'cpu' if you do not have a gpu
    DATASET_PATH = "NLU/part_1/dataset"
    PAD_TOKEN = 0

    train_loader, dev_loader, test_loader, lang = get_dataloaders(
        DATASET_PATH, PAD_TOKEN, DEVICE
    )

    out_int = len(lang.intent2id)
    out_slot = len(lang.slot2id)

    for exp in experiments:

        # prepare experiment params
        args = defaults | exp

        print(args)

        lr = args["lr"]
        OPT = args["OPT"]
        SCH = args["SCH"]
        sch_in = args["sch_in"]
        drop = args["drop"]
        runs = args["runs"]
        tag = args["tag"]
        model_name = args["model_name"]
        pooler = args["pooler"]

        run_name, run_path = build_run_name(args, SAVE_PATH)

        os.mkdir(run_path)

        EPOCHS = args["EPOCHS"]
        PAT = args["PAT"]

        slot_f1s, intent_acc = [], []

        print("[TRAIN] Starting ", run_name)

        pbar_runs = tqdm(range(1, runs + 1))
        for run_n in pbar_runs:

            model = MyBert(
                out_slot,
                out_int,
                drop,
                model_name,
                pooler,
            ).to(DEVICE)

            # trainable_params = []
            # for name, param in model.named_parameters():
            #     # if "pooler" in name:
            #     #     param.requires_grad = args["custom_pooler"]

            #     if "attention" in name:
            #         print(name, param.requires_grad)
            #         trainable_params.append(param)

            # breakpoint()

            if OPT == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=lr)
            elif OPT == "AdamW":
                optimizer = optim.AdamW(model.parameters(), lr=lr)

            if not sch_in:
                if SCH == "cosine":
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=30  # EPOCHS
                    )
                if SCH == "plateau":
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, "min", factor=0.2, patience=PAT // 2
                    )

            criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
            criterion_intents = nn.CrossEntropyLoss()

            if LOG:
                wandb.init(
                    project="NLU_assignment2",
                    name=run_name + "_" + str(run_n),
                    config={
                        "model": str(type(model).__name__),
                        "model_name": model_name,
                        "lr": lr,
                        "optim": str(type(optimizer).__name__),
                        "drop": drop,
                        "pooler": pooler,
                        "scheduler": SCH,
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

                if sch_in:
                    if SCH == "cosine":
                        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=len(train_loader)  # EPOCHS
                        )
                    if SCH == "plateau":
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, "min", factor=0.2, patience=5
                        )
                results_dev, intent_res, loss_dev, avg_loss_dev = eval_loop(
                    dev_loader, criterion_slots, criterion_intents, model, lang
                )
                if SCH == "none":
                    loss, avg_loss = train_loop(
                        train_loader,
                        optimizer,
                        criterion_slots,
                        criterion_intents,
                        model,
                    )
                else:

                    if sch_in:
                        loss, avg_loss = train_loop_sch_in(
                            train_loader,
                            optimizer,
                            scheduler,
                            criterion_slots,
                            criterion_intents,
                            model,
                        )
                    else:
                        loss, avg_loss = train_loop_sch(
                            train_loader,
                            optimizer,
                            scheduler,
                            criterion_slots,
                            criterion_intents,
                            model,
                        )

                results_dev, intent_res, loss_dev, avg_loss_dev = eval_loop(
                    dev_loader, criterion_slots, criterion_intents, model, lang
                )
                f1 = results_dev["total"]["f"]
                acc = intent_res["accuracy"]
                if f1 > best_f1:
                    best_f1 = f1
                    patience = PAT
                    best_model = deepcopy(model)
                else:
                    patience -= 1

                pbar_epochs.set_description(
                    f"F1 {round_sf(f1,3)} acc {round_sf(acc,3)} loss {round_sf(avg_loss_dev,3)} PAT {patience}"
                )

                if LOG:
                    wandb.log(
                        {
                            "loss": avg_loss,
                            "val loss": avg_loss_dev,
                            "F1": f1,
                            "acc": acc,
                        }
                    )

                if patience < 0:  # Early stopping with patient
                    break

            model.load_state_dict(best_model.state_dict())
            results_dev, intent_res, _, avg_loss_dev = eval_loop(
                dev_loader, criterion_slots, criterion_intents, model, lang
            )
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
                if run_n == runs:

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
    DATASET_PATH = "NLU/part_2/dataset"
    PAD_TOKEN = 0

    results = []

    for exp in experiments:

        # prepare experiment params
        args = defaults | exp

        drop = args["drop"]
        model_name = args["model_name"]
        pooler = args["pooler"]

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

        model = MyBert(
            out_slot,
            out_int,
            drop,
            model_name,
            pooler,
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

    print(args)
    # print the results in a table
    headers = ["Name", "Model", "LR", "Drop", "Scheduler", "F1", "Acc"]
    data = []
    for res in results:
        args = res["args"]
        f1 = res["f1"]
        acc = res["acc"]

        data.append(
            [
                res["name"],
                args["model_name"],
                args["lr"],
                args["drop"],
                args["SCH"],
                round(f1, 3),
                round(acc, 3),
            ]
        )

    print(tabulate(data, headers=headers, tablefmt="orgtbl"))


############################################################################################################
# Functions to handle the experiments


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

    if args["model_name"].split("-")[0] == "bert":
        run_name = "BERT"
    else:
        run_name = "ROBERTA"

    run_name += "_" + str(args["lr"])[2:] + "_" + str(round(args["drop"] * 100))

    if args["pooler"]:
        run_name += "_P"

    if args["custom_pooler"]:
        run_name += "_CP"

    if args["SCH"] == "cosine":
        run_name += "_cos"
    elif args["SCH"] == "plateau":
        run_name += "_plat"

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

    if len(slot_f1s) <= 2:
        return (
            round(np.array(slot_f1s).mean(), 3),
            round(np.array(slot_f1s).std(), 3),
            round(np.array(intent_acc).mean(), 3),
            round(np.array(intent_acc).std(), 3),
        )

    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)

    slot_f1s_mean = round(slot_f1s.mean(), 3)

    slot_f1s_std = round(slot_f1s.std(), 3)

    clean_slot_f1s = []
    clean_intent_acc = []
    for idx, (score, acc) in enumerate(zip(slot_f1s, intent_acc)):
        z = abs((score - slot_f1s_mean) / (slot_f1s_std + 0.0001))
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
        default="NLU/part_2/bin/",
        metavar="",
    )

    args = vars(parser.parse_args())
    return args
