from transformers import AutoTokenizer
from utils_old import get_dataloaders

# BERT TOKENIZER
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "Hello from trento university"
encoding = tokenizer(example)
print(type(encoding))

DEVICE = "cuda:0"  # it can be changed with 'cpu' if you do not have a gpu
DATASET_PATH = "NLU/part_1/dataset"
PAD_TOKEN = 0

train_loader, dev_loader, test_loader, lang = get_dataloaders(
    DATASET_PATH, PAD_TOKEN, DEVICE
)
