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
    ):
        super(MyBert, self).__init__()

        # self.bert = BertModel.from_pretrained("bert-base-uncased")

        model_name = "bert-base-uncased"
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        # config = self.bert.config
        self.dropout = nn.Dropout(0)

        # Intent classification head
        self.intent_classifier = nn.Linear(config.hidden_size, intent_size)

        # Slot filling classification head
        self.slot_classifier = nn.Linear(config.hidden_size, slot_size)

    def forward(self, input, attention_mask, mapping):

        # utt_x = sample["utt"]
        # slots_y = sample["slots"]
        # att = sample["att"]
        # intent_y = sample["intent"]
        # words = sample["words"]
        # Pass inputs through BERT
        outputs = self.bert(input_ids=input, attention_mask=attention_mask)

        # CLS token embedding for intent classification
        # cls_output = outputs.pooler_output
        cls_output = outputs.last_hidden_state[:, 0]
        intent_logits = self.intent_classifier(cls_output)

        # Token-level embeddings for slot classification
        sequence_output = outputs.last_hidden_state

        # Apply the mapping to the sequence_output
        remapped_seq = torch.zeros(
            (mapping.shape[0], mapping.shape[1], sequence_output.shape[2])
        ).to(input.device)

        for i, map in enumerate(mapping):
            for j, idx in enumerate(map):
                remapped_seq[i, j] = sequence_output[i, idx]

        # remapped_seq = torch.gather(
        #     sequence_output,
        #     1,
        #     mapping.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1)),
        # )
        # breakpoint()
        # Compute slot logits
        # remapped_seq = torch.stack(
        #     [outputs.last_hidden_state[i][idx] for i, idx in enumerate(mapping)]
        # ).to(input.device)
        # breakpoint()
        slot_logits = self.slot_classifier(remapped_seq)

        slot_logits = slot_logits.permute(0, 2, 1)
        # breakpoint()
        return intent_logits, slot_logits
