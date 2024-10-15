import os
import wandb
from dotenv import load_dotenv

from functions import get_args, load_experiments, run_experiments


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
        defaults = {}
        experiments = []

    if TRAIN:
        run_experiments(defaults, experiments, glob_args)

    if TEST:
        # run_tests(defaults, experiments, glob_args)
        pass
