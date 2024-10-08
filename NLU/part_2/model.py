import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertTokenizer, BertModel


class MyBert(nn.Module):
    def __init__(
        self,
        intent_num_labels,
        slot_num_labels,
    ):
        super(MyBert, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)

        # Intent classification head
        self.intent_classifier = nn.Linear(
            self.bert.config.hidden_size, intent_num_labels
        )

        # Slot filling classification head
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, slot_num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Pass inputs through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # CLS token embedding for intent classification
        cls_output = outputs.pooler_output
        intent_logits = self.intent_classifier(self.dropout(cls_output))

        # Token-level embeddings for slot classification
        sequence_output = outputs.last_hidden_state
        slot_logits = self.slot_classifier(self.dropout(sequence_output))

        return intent_logits, slot_logits
