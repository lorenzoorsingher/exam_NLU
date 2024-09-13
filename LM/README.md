# Mandatory Exam Exercise

## Part 1 (4 points)

In this, you have to modify the baseline LM_RNN by adding a set of techniques that might improve the performance. In this, you have to add one modification at a time incrementally. If adding a modification decreases the performance, you can remove it and move forward with the others. However, in the report, you have to provide and comment on this unsuccessful experiment. For each of your experiments, you have to print the performance expressed with Perplexity (PPL).
<br>
One of the important tasks of training a neural network is hyperparameter optimization. Thus, you have to play with the hyperparameters to minimise the PPL and thus print the results achieved with the best configuration (in particular <b>the learning rate</b>).
These are two links to the state-of-the-art papers which use vanilla RNN [paper1](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5947611), [paper2](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf).

**Mandatory requirements**: For the following experiments the perplexity must be below 250 (**_PPL < 250_**).

1. Replace RNN with a Long-Short Term Memory (LSTM) network --> [link](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
2. Add two dropout layers: --> [link](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
   - one after the embedding layer,
   - one before the last linear layer
3. Replace SGD with AdamW --> [link](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

## Part 2 (11 points)

**Mandatory requirements**: For the following experiments the perplexity must be below 250 (**_PPL < 250_**) and it should be lower than the one achieved in Part 1.1 (i.e. base LSTM).

Starting from the `LM_RNN` in which you replaced the RNN with a LSTM model, apply the following regularisation techniques:

- Weight Tying
- Variational Dropout (no DropConnect)
- Non-monotonically Triggered AvSGD

These techniques are described in [this paper](https://openreview.net/pdf?id=SyyGPP0TZ).

---

## Notes

- [data spreadsheet](https://docs.google.com/spreadsheets/d/1VwQ7OH2VsRvNmzsrU7Hb7BSlrhgjF7YU90Jx3-wQ2v8/edit?gid=770483852#gid=770483852)
