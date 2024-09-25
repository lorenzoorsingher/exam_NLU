import os
from functions import run_experiments

from dotenv import load_dotenv
from functions import get_args
import wandb
import json

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

TRAIN = glob_args["train"]
TEST = glob_args["test"]

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
    defaults = {}
    experiments = []

if TRAIN:
    run_experiments(defaults, experiments, glob_args)

if TEST:
    # run_tests(defaults, experiments, glob_args)
    pass
