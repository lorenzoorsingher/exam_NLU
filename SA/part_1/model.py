import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertTokenizer, BertModel
from transformers import AutoModel, AutoConfig


class MyBert(nn.Module):
    def __init__(
        self,
        slot_size,
        drop=0.1,
        model_name="bert-base-uncased",
    ):
        super(MyBert, self).__init__()

        # self.bert = BertModel.from_pretrained("bert-base-uncased")

        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # config = self.bert.config
        self.dropout = nn.Dropout(drop)

        # Slot filling classification head
        self.slot_classifier = nn.Linear(config.hidden_size, slot_size)

    def forward(self, input, attention_mask):

        # Pass inputs through BERT
        outputs = self.bert(input_ids=input, attention_mask=attention_mask)

        # Token-level embeddings for slot classification
        sequence_output = outputs.last_hidden_state
        slot_logits = self.slot_classifier(self.dropout(sequence_output))
        slot_logits = slot_logits.permute(0, 2, 1)
        # breakpoint()
        return slot_logits
