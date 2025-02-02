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

SMALL_POSITIVE_CONST = 1e-6


class MockScheduler:
    def __init__(self):
        pass

    def step(self):
        pass


def match_ts(gold_ts_sequence, pred_ts_sequence):
    """
    calculate the number of correctly predicted targeted sentiment
    :param gold_ts_sequence: gold standard targeted sentiment sequence
    :param pred_ts_sequence: predicted targeted sentiment sequence
    :return:
    """
    # positive, negative and neutral
    tag2tagid = {"T": 0}
    hit_count, gold_count, pred_count = np.zeros(1), np.zeros(1), np.zeros(1)
    for t in gold_ts_sequence:
        # print(t)
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        gold_count[tid] += 1
    for t in pred_ts_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        if t in gold_ts_sequence:
            hit_count[tid] += 1
        pred_count[tid] += 1
    return hit_count, gold_count, pred_count


def tag2ts(ts_tag_sequence):
    """
    transform ts tag sequence to targeted sentiment
    :param ts_tag_sequence: tag sequence for ts task
    :return:
    """
    n_tags = len(ts_tag_sequence)
    ts_sequence = []
    beg, end = -1, -1
    sentiment = None

    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        if ts_tag == "T":
            if beg == -1:
                beg = i
            end = i
            sentiment = "T"
        else:
            if beg != -1 and end != -1:
                ts_sequence.append((beg, end, sentiment))
                beg, end, sentiment = -1, -1, None

    if beg != -1 and end != -1:
        ts_sequence.append((beg, end, sentiment))

    return ts_sequence


# TODO Check if it's a problem that model predicts pad instead of O
def evaluate_ts(gold_ts, pred_ts):
    """
    evaluate the model performance for the ts task
    :param gold_ts: gold standard ts tags
    :param pred_ts: predicted ts tags
    :return:
    """
    assert len(gold_ts) == len(pred_ts)
    n_samples = len(gold_ts)
    # number of true postive, gold standard, predicted targeted sentiment
    n_tp_ts, n_gold_ts, n_pred_ts = 0, 0, 0
    ts_precision, ts_recall, ts_f1 = 0, 0, 0

    for i in range(n_samples):
        g_ts = gold_ts[i]
        p_ts = pred_ts[i]
        # breakpoint()
        g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=g_ts), tag2ts(
            ts_tag_sequence=p_ts
        )
        # breakpoint()
        hit_ts_count, gold_ts_count, pred_ts_count = match_ts(
            gold_ts_sequence=g_ts_sequence, pred_ts_sequence=p_ts_sequence
        )
        n_tp_ts += hit_ts_count[0]
        n_gold_ts += gold_ts_count[0]
        n_pred_ts += pred_ts_count[0]
    # breakpoint()
    # calculate macro-average scores for ts task
    # for i in range(3):
    #     n_ts = n_tp_ts[i]
    #     n_g_ts = n_gold_ts[i]
    #     n_p_ts = n_pred_ts[i]
    #     ts_precision[i] = float(n_ts) / float(n_p_ts + SMALL_POSITIVE_CONST)
    #     ts_recall[i] = float(n_ts) / float(n_g_ts + SMALL_POSITIVE_CONST)
    #     ts_f1[i] = (
    #         2
    #         * ts_precision[i]
    #         * ts_recall[i]
    #         / (ts_precision[i] + ts_recall[i] + SMALL_POSITIVE_CONST)
    #     )

    # ts_macro_f1 = ts_f1.mean()

    # calculate micro-average scores for ts task
    # total sum of TP and FN
    # total sum of TP and FP

    ts_micro_p = float(n_tp_ts) / (n_pred_ts + SMALL_POSITIVE_CONST)
    ts_micro_r = float(n_tp_ts) / (n_gold_ts + SMALL_POSITIVE_CONST)
    ts_micro_f1 = (
        2 * ts_micro_p * ts_micro_r / (ts_micro_p + ts_micro_r + SMALL_POSITIVE_CONST)
    )
    ts_scores = (0, ts_micro_p, ts_micro_r, ts_micro_f1)

    return ts_scores


def eval_loop(data, criterion_slots, model, lang):
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

    ref_slots = []
    hyp_slots = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:

            utt_x = sample["utt"]
            words = sample["words"]
            slots_y = sample["slots"]
            att = sample["att"]
            lens = sample["len_slots"]

            slots = model(utt_x, att)

            loss = criterion_slots(slots, slots_y)
            loss_array.append(loss.item())

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

        results = evaluate_ts(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        # print("Warning:", ex)
        print("Class is not in REF")
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        # print(hyp_s.difference(ref_s))
        results = 0, 0, 0, 0
    # report_intent = classification_report(
    #     ref_intents, hyp_intents, zero_division=False, output_dict=True
    # )

    loss_avg = np.array(loss_array).mean()
    return results, loss_array, loss_avg


def train_loop(data, optimizer, criterion_slots, model, lang, clip=5):
    """
    Function to train the model

    Parameters:
    - data (DataLoader): DataLoader object
    - optimizer (optim): optimizer
    - criterion_slots (nn.CrossEntropyLoss): loss function for slots
    - model (nn.Module): model
    - clip (int): clipping value
    """
    lang
    model.train()
    loss_array = []
    for sample in tqdm(data):

        utt_x = sample["utt"]
        slots_y = sample["slots"]
        att = sample["att"]

        optimizer.zero_grad()  # Zeroing the gradient
        slots = model(utt_x, att)

        loss = criterion_slots(slots, slots_y)

        # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward()  # Compute the gradient, deleting the computational graph

        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights

    avg_loss = np.array(loss_array).mean()
    return loss_array, avg_loss


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
            hyp_slot.append(lang.id2slot[pred_id.item()])
            ref_slot.append(lang.id2slot[gt_id])
            slot_idx += 1
    return hyp_slot, ref_slot


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
    DATASET_PATH = "SA/part_1/data"
    PAD_TOKEN = 0

    for exp in experiments:

        # prepare experiment params
        args = defaults | exp

        print(args)

        lr = args["lr"]
        SCH = args["SCH"]
        drop = args["drop"]
        runs = args["runs"]
        tag = args["tag"]
        model_name = args["model_name"]

        train_loader, dev_loader, test_loader, lang = get_dataloaders(
            DATASET_PATH, PAD_TOKEN, DEVICE, model_name=model_name
        )
        out_slot = len(lang.slot2id)

        run_name, run_path = build_run_name(args, SAVE_PATH)

        os.mkdir(run_path)

        EPOCHS = args["EPOCHS"]
        PAT = args["PAT"]

        slot_f1s = []

        print("[TRAIN] Starting ", run_name)

        pbar_runs = tqdm(range(1, runs + 1))
        for run_n in pbar_runs:

            model = MyBert(
                out_slot,
                drop,
                model_name,
            ).to(DEVICE)

            optimizer = optim.AdamW(model.parameters(), lr=lr)

            if SCH == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=30  # EPOCHS
                )
            elif SCH == "plateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "max", factor=0.2, patience=PAT // 2
                )
            else:
                scheduler = MockScheduler()

            criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

            if LOG:
                wandb.init(
                    project="NLU_assignment3",
                    name=run_name + "_" + str(run_n),
                    config=args,
                )
            best_f1 = 0
            best_res = None
            patience = PAT
            pbar_epochs = tqdm(range(0, EPOCHS))

            best_model = None
            for epoch in pbar_epochs:

                # results, _, loss_avg = eval_loop(
                #     test_loader, criterion_slots, model, lang
                # )
                loss, avg_loss = train_loop(
                    train_loader, optimizer, criterion_slots, model, lang
                )

                # Update the optimizer
                print("Epoch", epoch, "Loss", round(avg_loss, 3))

                results, _, loss_avg = eval_loop(
                    test_loader, criterion_slots, model, lang
                )
                _, micro_p, micro_r, micro_f1 = results

                if micro_f1 > best_f1:
                    best_f1 = micro_f1
                    best_res = results
                    patience = PAT
                    best_model = deepcopy(model)
                else:
                    patience -= 1

                if SCH == "plateau":
                    scheduler.step(micro_f1)
                else:
                    scheduler.step()
                # pbar_epochs.set_description(
                #     f"F1 {round_sf(macro_f1,3)} prec {round_sf(micro_p,3)} rec {round_sf(micro_r,3)} loss {round_sf(loss_avg,3)} PAT {patience}"
                # )

                print(
                    f"F1 {round_sf(micro_f1,3)} prec {round_sf(micro_p,3)} rec {round_sf(micro_r,3)} loss {round_sf(loss_avg,3)} PAT {patience}\n"
                )

                if LOG:
                    wandb.log(
                        {
                            "loss": avg_loss,
                            "micro_p": micro_p,
                            "micro_r": micro_r,
                            "micro_f1": micro_f1,
                        }
                    )

                if patience < 0:  # Early stopping with patient
                    break

            # model.load_state_dict(best_model.state_dict())
            # results_dev, intent_res, _, avg_loss_dev = eval_loop(
            #     dev_loader, criterion_slots, criterion_intents, model, lang
            # )
            # results_test, intent_test, _, avg_loss_test = eval_loop(
            #     test_loader, criterion_slots, criterion_intents, model, lang
            # )
            # intent_acc.append(intent_test["accuracy"])
            # slot_f1s.append(results_test["total"]["f"])

            macro_f1, micro_p, micro_r, micro_f1 = best_res
            print("BEST RESULTS:")
            print(
                f"F1 {round_sf(macro_f1,3)} prec {round_sf(micro_p,3)} rec {round_sf(micro_r,3)} loss {round_sf(loss_avg,3)}"
            )

            # if LOG:

            # loss_gap = (avg_loss_test - avg_loss_dev) / avg_loss_test

            # wandb.log(
            #     {
            #         "val loss": avg_loss_dev,
            #         "F1": results_dev["total"]["f"],
            #         "acc": intent_res["accuracy"],
            #         "test F1": results_test["total"]["f"],
            #         "test acc": intent_test["accuracy"],
            #         "loss_gap": loss_gap,
            #     }
            # )

            # # on last run log the mean and std of the results
            # if run_n == runs:

            #     slot_f1s_mean, slot_f1s_std, intent_acc_mean, intent_acc_std = (
            #         remove_outliers(slot_f1s, intent_acc)
            #     )

            #     wandb.log(
            #         {
            #             "slot_f1s_mean": slot_f1s_mean,
            #             "slot_f1s_std": slot_f1s_std,
            #             "intent_acc_mean": intent_acc_mean,
            #             "intent_acc_std": intent_acc_std,
            #         }
            #     )

            wandb.finish()

        #     model_savefile = {
        #         "state_dict": model.state_dict(),
        #         "lang": lang,
        #     }
        #     torch.save(model_savefile, run_path + "best.pt")

        # slot_f1s = np.asarray(slot_f1s)
        # intent_acc = np.asarray(intent_acc)

        # print("Slot F1", round(slot_f1s.mean(), 3), "+-", round(slot_f1s.std(), 3))
        # print("Intent Acc", round(intent_acc.mean(), 3), "+-", round(slot_f1s.std(), 3))


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
    DATASET_PATH = "SA/part_1/data"
    PAD_TOKEN = 0

    results = []

    for exp in experiments:

        # prepare experiment params
        args = defaults | exp

        drop = args["drop"]
        model_name = args["model_name"]

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
        # breakpoint()
        if weights_path == "":
            print("[TEST] Model not found ", run_name)
            continue

        # load the model, lang and test set
        model_savefile = torch.load(weights_path, weights_only=False)
        state_dict = model_savefile["state_dict"]
        lang = model_savefile["lang"]

        _, _, test_loader, _ = get_dataloaders(
            DATASET_PATH, PAD_TOKEN, DEVICE, model_name=model_name, lang=lang
        )

        out_slot = len(lang.slot2id)
        out_int = len(lang.intent2id)

        model = MyBert(
            out_slot,
            out_int,
            drop,
            model_name,
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
    headers = ["Name", "Model", "LR", "Drop", "Scheduler", "Pooler", "Acc", "F1"]
    data = []
    for res in results:
        args = res["args"]
        f1 = res["f1"]
        acc = res["acc"]

        if args["pooler"]:
            pooler = "P"
        else:
            pooler = " "

        data.append(
            [
                res["name"],
                args["model_name"],
                args["lr"],
                args["drop"],
                args["SCH"],
                pooler,
                round(acc, 3),
                round(f1, 3),
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
        default="SA/part_1/bin/",
        metavar="",
    )

    args = vars(parser.parse_args())
    return args
