import torch.optim as optim

import torch.nn as nn
import torch

from transformers import BertTokenizer, BertModel

from functions import prepare_input
from utils import get_dataloaders

PAD_TOKEN = 0
DATASET_PATH = "NLU/part_2/dataset"
DEVICE = "cuda:0"

train_loader, dev_loader, test_loader, lang = get_dataloaders(
    DATASET_PATH, PAD_TOKEN, DEVICE
)

out_slot = len(lang.slot2id)
out_int = len(lang.intent2id)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

from model import MyBert

# Create a model instance
model = MyBert(intent_num_labels=out_int, slot_num_labels=out_slot)

# Define loss functions for intent and slot
intent_loss_fn = nn.CrossEntropyLoss()
slot_loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=2e-5)

model.train()
# Dummy training loop
for epoch in range(3):  # Train for 3 epochs

    for sample in train_loader:

        txt = sample["txt"]
        utt = sample["utterances"]
        slen = sample["slots_len"]
        intents = sample["intents"]
        yslots = sample["y_slots"]

        inputs = tokenizer(txt, return_tensors="pt", padding=True, truncation=True)
        ids = inputs["input_ids"]
        att = inputs["attention_mask"]
        ids = inputs["token_type_ids"]
        breakpoint()

    input_ids, attention_mask, token_type_ids = prepare_input(
        "Book a flight to Paris", tokenizer
    )

    intent_labels = torch.tensor([1])  # Example intent label
    slot_labels = torch.tensor([[0, 0, 0, 0, 1, 0, 0]])  # Example slot labels

    # Forward pass
    intent_logits, slot_logits = model(input_ids, attention_mask, token_type_ids)

    # Compute losses
    intent_loss = intent_loss_fn(intent_logits, intent_labels)

    slot_loss = slot_loss_fn(
        slot_logits.view(-1, slot_logits.shape[-1]), slot_labels.view(-1)
    )

    # Total loss is a weighted sum (you can adjust weights as needed)
    total_loss = intent_loss + slot_loss

    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss.item()}")
