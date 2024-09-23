import os
import wandb
import json
from dotenv import load_dotenv


from functions import get_args, run_experiments


# load args
load_dotenv()
glob_args = get_args()

# load experiments from json if provided
if glob_args["json"] == "":
    FROM_JSON = False
    WANDB_SECRET = ""
else:
    FROM_JSON = True
    json_path = glob_args["json"]
    WANDB_SECRET = glob_args["wandb_secret"]

# set up loggin if required
LOG = not glob_args["no_log"]
if LOG:
    if WANDB_SECRET == "":
        WANDB_SECRET = os.getenv("WANDB_SECRET")
    wandb.login(key=WANDB_SECRET)

# parse and prepare experiments
if FROM_JSON:
    print("loading from json...")
    if os.path.exists(json_path):
        filename = json_path.split("/")[-1]
        print("loading from ", filename)
        defaults, experiments = json.load(open(json_path))
    else:
        print("json not found, exiting...")
        exit()
else:
    defaults = {
        "emb_size": 300,
        "hid_size": 300,
        "lr": 1.5,
        "clip": 5,
        "n_layers": 1,
        "emb_drop": 0.1,
        "out_drop": 0.1,
        "tying": False,
        "var_drop": False,
        "EPOCHS": 99,
        "OPT": "SGD",
        "PAT": 5,
        "tag": "general",
        "arch": "LSTM",
    }
    experiments = [
        {
            "emb_drop": 0.5,
            "out_drop": 0.5,
            "var_drop": True,
            "tag": "general",
        }
    ]

run_experiments(defaults, experiments, glob_args)
