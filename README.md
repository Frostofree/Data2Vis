# Data2Vis

## Train instructions
```bash
python run.py
```

## Contents

- Implemented the Data2Vis model for the task of generating visualizations for the Vega-Lite6 dataset in Pytorch.
- The model is based on the Encoder-Decoder architecture with an attention mechanism.
- The model is trained on the Vega-Lite dataset and generates visualizations for the given input data.

## Parameters

- Embedding dimension: 512.
- Encoder: Bidirectional LSTM, 2 layers, 512 hidden units.
- Decoder: LSTM, 2 layers, 512 hidden units.
- Droupout: probability 0.5 before every encoder/decoder layer, only on the
inputs, not the hidden/cell states.
- Attention type: Dot product.
- Attention vector dimension: 512.
- Max source sequence length: 500 (first 500).
- Max target sequence length: 500 (first 500).
- Max decode sequence length: 2000 (during inference, first 2000).
- Width for beam search: 15 (beam search is only used during inference).
- Optimizer: Adam, lr=1e-4
- Batch size: 32
- Number of steps: 20000 steps (minibatches)


## Features Implemented

- [x] Encoder Decoder Architecture
- [x] Attention Mechanism
- [x] Achieved Log Perplexity of 0.032
- [x] Beam Search for Decoding
- [] Visualizations

## Errors to Fix
- [] Visualizations, most likely a problem with the decoder, decrease in loss is minimal after 16000 batches.

## Download Dataset

- [Vega-Lite](https://github.com/victordibia/data2vis/blob/master/code/sourcedata)





