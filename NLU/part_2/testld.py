import os
import argparse
import json
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


DEVICE = "cuda:0"  # it can be changed with 'cpu' if you do not have a gpu
DATASET_PATH = "NLU/part_1/dataset"
PAD_TOKEN = 0
train_loader, dev_loader, test_loader, lang = get_dataloaders(
    DATASET_PATH, PAD_TOKEN, DEVICE
)

for sample in train_loader:

    breakpoint()
    print(sample)
    pass
