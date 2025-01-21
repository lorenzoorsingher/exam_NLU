import os
import wandb
from dotenv import load_dotenv

from functions import get_args, load_experiments, run_experiments, run_tests


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
            "model_name": "bert-base-uncased",
            "lr": 0.0001,
            "EPOCHS": 100,
            "runs": 4,
            "PAT": 5,
            "drop": 0,
            "pooler": False,
            "SCH": "none",
            "sch_in": False,
            "custom_pooler": False,
            "OPT": "AdamW",
            "tag": "bert",
        }

        experiments = [
            {"SCH": "none"},
            {"PAT": 7, "SCH": "plateau"},
            {"SCH": "cosine"},
            {"pooler": True},
            {"drop": 0.25},
            {"drop": 0.5},
            {"drop": 0.25, "lr": 0.0001},
            {"drop": 0.5, "lr": 0.0001},
            {"drop": 0.25, "lr": 0.0002},
            {"drop": 0.5, "lr": 0.0002},
            {"PAT": 7, "SCH": "plateau", "drop": 0.25, "lr": 0.0002},
            {"PAT": 7, "SCH": "plateau", "drop": 0.5, "lr": 0.0002},
            {"PAT": 7, "SCH": "cosine", "drop": 0.25, "lr": 0.0002},
            {"PAT": 7, "SCH": "cosine", "drop": 0.5, "lr": 0.0002},
        ]

    if TRAIN:
        run_experiments(defaults, experiments, glob_args)

    if TEST:
        run_tests(defaults, experiments, glob_args)
