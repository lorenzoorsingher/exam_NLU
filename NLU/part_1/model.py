import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VarDropout(nn.Module):
    def __init__(self, p=0.1):
        super(VarDropout, self).__init__()
        self.p = p

    def forward(self, x):

        if not self.training:
            return x

        rand_mask = torch.rand((x.shape[::2]), requires_grad=True, device="cuda")
        expanded_mask = (rand_mask > self.p).int().unsqueeze(1)
        full_mask = expanded_mask.repeat(1, x.shape[1], 1)

        return (x * full_mask) * (1.0 / (1.0 - self.p))


class ModelIAS(nn.Module):

    def __init__(
        self,
        hid_size,
        out_slot,
        out_int,
        emb_size,
        vocab_len,
        n_layer=1,
        pad_index=0,
        var_drop=False,
        bi=False,
        dropout=0.1,
    ):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(
            emb_size, hid_size, n_layer, bidirectional=bi, batch_first=True
        )

        if bi:
            self.slot_out = nn.Linear(hid_size * 2, out_slot)
        else:
            self.slot_out = nn.Linear(hid_size, out_slot)

        self.intent_out = nn.Linear(hid_size, out_int)

        if var_drop:
            print("[MODEL] Using VarDrop")
            self.drop_emb = VarDropout(p=dropout)
            self.drop_enc = VarDropout(p=dropout)
        else:
            print("[MODEL] Using Drop")
            self.drop_emb = nn.Dropout(p=dropout)
            self.drop_enc = nn.Dropout(p=dropout)

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(
            utterance
        )  # utt_emb.size() = batch_size X seq_len X emb_size

        drop_emb = self.drop_emb(utt_emb)
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost

        packed_input = pack_padded_sequence(
            drop_emb, seq_lengths.cpu().numpy(), batch_first=True
        )
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        drop_enc = self.drop_enc(utt_encoded)

        # TODO dropout on last hidden?
        # Get the last hidden state
        last_hidden = last_hidden[-1, :, :]

        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]
        # breakpoint()
        # Compute slot logits
        slots = self.slot_out(drop_enc)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0, 2, 1)  # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len

        return slots, intent
