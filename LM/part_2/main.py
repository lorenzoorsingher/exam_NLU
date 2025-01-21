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
            "emb_drop": 0.35,
            "out_drop": 0.35,
            "tying": False,
            "var_drop": False,
            "EPOCHS": 250,
            "OPT": "SGD",
            "PAT": 5,
            "tag": "general",
            "arch": "LSTM",
        }
        experiments = [
            {"lr": 1.5, "tying": True, "emb_drop": 0, "out_drop": 0},
            {"lr": 2, "tying": True, "emb_drop": 0, "out_drop": 0},
            {"lr": 2.5, "tying": True, "emb_drop": 0, "out_drop": 0},
            {"lr": 1.5, "var_drop": True},
            {"lr": 2, "var_drop": True},
            {"lr": 2.5, "var_drop": True},
            {"lr": 1.5, "var_drop": True, "tying": True},
            {"lr": 2, "var_drop": True, "tying": True},
            {"lr": 2.5, "var_drop": True, "tying": True},
            {"lr": 1.5, "var_drop": True, "tying": True, "OPT": "ASGD"},
            {"lr": 2, "var_drop": True, "tying": True, "OPT": "ASGD"},
            {"lr": 2.5, "var_drop": True, "tying": True, "OPT": "ASGD"},
        ]

    if TRAIN:
        run_experiments(defaults, experiments, glob_args)

    if TEST:
        run_tests(defaults, experiments, glob_args)
