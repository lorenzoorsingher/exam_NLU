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
            "PAT": 10,
            "drop": 0,
            "SCH": "none",
            "tag": "",
        }
        experiments = [
            {"lr": 0.0001},
            {"lr": 0.0001, "drop": 0.3},
            {"lr": 0.0002},
            {"lr": 0.0002, "drop": 0.3},
            {"model_name": "roberta-base"},
            {"model_name": "cardiffnlp/twitter-roberta-base-emotion"},
            {"model_name": "roberta-base", "drop": 0.3},
            {"model_name": "cardiffnlp/twitter-roberta-base-emotion", "drop": 0.3},
        ]

    if TRAIN:
        run_experiments(defaults, experiments, glob_args)

    if TEST:
        run_tests(defaults, experiments, glob_args)
