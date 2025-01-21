import os
import wandb
from dotenv import load_dotenv

from functions import get_args, load_experiments, run_experiments, run_tests

if __name__ == "__main__":
    # load args
    load_dotenv()
    glob_args = get_args()

    TRAIN = glob_args["train"]
    TEST = glob_args["test"]
    LOG = not glob_args["no_log"]
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
            "lr": 0.001,
            "EPOCHS": 350,
            "runs": 5,
            "PAT": 5,
            "var_drop": False,
            "drop": 0,
            "OPT": "AdamW",
            "tag": "NLU1",
            "bi": False,
        }
        experiments = [
            {"lr": 0.001, "drop": 0, "bi": False},
            {"lr": 0.005, "drop": 0, "bi": False},
            {"lr": 0.001, "drop": 0.25, "bi": False},
            {"lr": 0.001, "drop": 0.5, "bi": False},
            {"lr": 0.001, "drop": 0.25, "var_drop": True, "bi": False},
            {"lr": 0.001, "drop": 0.5, "var_drop": True, "bi": False},
            {"lr": 0.001, "drop": 0, "bi": True},
            {"lr": 0.001, "drop": 0.25, "var_drop": True, "bi": True},
            {"lr": 0.001, "drop": 0.5, "var_drop": True, "bi": True},
        ]

    if TRAIN:
        run_experiments(defaults, experiments, glob_args)
        # run_tests(defaults, experiments, glob_args)

    if TEST:
        run_tests(defaults, experiments, glob_args)
