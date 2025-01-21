import os
import wandb
from dotenv import load_dotenv

from functions import get_args, run_experiments, run_tests, load_experiments

if __name__ == "__main__":
    # load args
    load_dotenv()
    glob_args = get_args()

    TRAIN = glob_args["train"]
    TEST = True  # glob_args["test"]
    LOG = glob_args["log"]
    FROM_JSON = not glob_args["json"] == ""

    # setup logging
    if LOG:
        WANDB_SECRET = glob_args["wandb_secret"]
        if WANDB_SECRET == "":
            WANDB_SECRET = os.getenv("WANDB_SECRET")
        wandb.login(key=WANDB_SECRET)

    # parse and prepare experiments
    if FROM_JSON:
        json_path = glob_args["json"]
        defaults, experiments = load_experiments(json_path)
    else:
        defaults = {
            "emb_size": 300,
            "hid_size": 300,
            "lr": 1.5,
            "clip": 5,
            "n_layers": 1,
            "emb_drop": 0,
            "out_drop": 0,
            "tying": False,
            "var_drop": False,
            "EPOCHS": 120,
            "OPT": "SGD",
            "PAT": 5,
            "tag": "general",
            "arch": "LSTM",
        }
        experiments = [
            {
                "lr": 1.5,
                "tying": True,
                "arch": "LSTM",
                "EPOCHS": 250,
                "tag": "lastrun",
            },
            {
                "lr": 2,
                "tying": True,
                "arch": "LSTM",
                "EPOCHS": 250,
                "tag": "lastrun",
            },
            {
                "lr": 2.5,
                "tying": True,
                "arch": "LSTM",
                "EPOCHS": 250,
                "tag": "lastrun",
            },
            {
                "lr": 1.5,
                "var_drop": True,
                "arch": "LSTM",
                "emb_drop": 0.35,
                "out_drop": 0.35,
                "EPOCHS": 250,
                "tag": "lastrun",
            },
            {
                "lr": 2,
                "var_drop": True,
                "arch": "LSTM",
                "emb_drop": 0.35,
                "out_drop": 0.35,
                "EPOCHS": 250,
                "tag": "lastrun",
            },
            {
                "lr": 2.5,
                "var_drop": True,
                "arch": "LSTM",
                "emb_drop": 0.35,
                "out_drop": 0.35,
                "EPOCHS": 250,
                "tag": "lastrun",
            },
            {
                "lr": 1.5,
                "var_drop": True,
                "tying": True,
                "arch": "LSTM",
                "emb_drop": 0.35,
                "out_drop": 0.35,
                "EPOCHS": 250,
                "tag": "lastrun",
            },
            {
                "lr": 2,
                "var_drop": True,
                "tying": True,
                "arch": "LSTM",
                "emb_drop": 0.35,
                "out_drop": 0.35,
                "EPOCHS": 250,
                "tag": "lastrun",
            },
            {
                "lr": 2.5,
                "var_drop": True,
                "tying": True,
                "arch": "LSTM",
                "emb_drop": 0.35,
                "out_drop": 0.35,
                "EPOCHS": 250,
                "tag": "lastrun",
            },
            {
                "lr": 1.5,
                "var_drop": True,
                "tying": True,
                "arch": "LSTM",
                "emb_drop": 0.35,
                "out_drop": 0.35,
                "OPT": "ASGD",
                "EPOCHS": 250,
                "tag": "lastrun",
            },
            {
                "lr": 2,
                "var_drop": True,
                "tying": True,
                "arch": "LSTM",
                "emb_drop": 0.35,
                "out_drop": 0.35,
                "OPT": "ASGD",
                "EPOCHS": 250,
                "tag": "lastrun",
            },
            {
                "lr": 2.5,
                "var_drop": True,
                "tying": True,
                "arch": "LSTM",
                "emb_drop": 0.35,
                "out_drop": 0.35,
                "OPT": "ASGD",
                "EPOCHS": 250,
                "tag": "lastrun",
            },
        ]

    if TRAIN:
        run_experiments(defaults, experiments, glob_args)

    if TEST:
        run_tests(defaults, experiments, glob_args)
