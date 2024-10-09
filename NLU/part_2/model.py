import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertTokenizer, BertModel


class MyBert(nn.Module):
    def __init__(
        self,
        slot_num_labels,
        intent_num_labels,
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

    def forward(self, input, attention_mask, mapping, lens):
        # Pass inputs through BERT
        outputs = self.bert(input_ids=input, attention_mask=attention_mask)

        # sequence_output = outputs.last_hidden_state
        # final_output = outputs.last_hidden_state[:, 0]

        # CLS token embedding for intent classification
        cls_output = outputs.pooler_output
        intent_logits = self.intent_classifier(self.dropout(cls_output))

        # Token-level embeddings for slot classification
        sequence_output = outputs.last_hidden_state

        # Apply the mapping to the sequence_output
        remapped_output = []

        remapped_seq = torch.zeros(
            (mapping.shape[0], mapping.shape[1], sequence_output.shape[2])
        ).to(input.device)

        for i, (map, len) in enumerate(zip(mapping, lens)):

            for j, idx in enumerate(map):
                remapped_seq[i, j] = sequence_output[i, idx]

        slot_logits = self.slot_classifier(self.dropout(remapped_seq))

        slot_logits = slot_logits.permute(0, 2, 1)

        return intent_logits, slot_logits
