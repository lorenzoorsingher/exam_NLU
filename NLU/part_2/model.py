import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertTokenizer, BertModel
from transformers import AutoModel, AutoConfig


class MyBert(nn.Module):
    def __init__(
        self,
        slot_size,
        intent_size,
        drop=0.1,
        model_name="bert-base-uncased",
        pooler=False,
    ):
        super(MyBert, self).__init__()

        # self.bert = BertModel.from_pretrained("bert-base-uncased")

        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        self.pooler = pooler

        # config = self.bert.config
        self.dropout = nn.Dropout(drop)

        # Intent classification head
        self.intent_classifier = nn.Linear(config.hidden_size, intent_size)

        # Slot filling classification head
        self.slot_classifier = nn.Linear(config.hidden_size, slot_size)

    def forward(self, input, attention_mask):

        # Pass inputs through BERT
        outputs = self.bert(input_ids=input, attention_mask=attention_mask)

        # CLS token embedding for intent classification
        if self.pooler:
            cls_output = outputs.pooler_output
        else:
            cls_output = outputs.last_hidden_state[:, 0]
        intent_logits = self.intent_classifier(self.dropout(cls_output))

        # Token-level embeddings for slot classification
        sequence_output = outputs.last_hidden_state
        slot_logits = self.slot_classifier(self.dropout(sequence_output))
        slot_logits = slot_logits.permute(0, 2, 1)
        # breakpoint()
        return intent_logits, slot_logits
