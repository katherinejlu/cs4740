# Project 3 — Sentiment Analysis with Neural Networks
**CS 4740: Natural Language Processing, Cornell University, Fall 2019**

5-class sentiment classification on Yelp reviews using a Feedforward Neural Network and a Recurrent Neural Network, built from scratch in PyTorch.

## Task
Given a Yelp review, predict the star rating (1–5). This is treated as a 5-class classification problem. Labels are shifted to 0–4 for zero-indexing.

## Models

### Feedforward Neural Network (FFNN)
- Bag-of-words input representation: each review is a fixed-length vector of word counts over the vocabulary
- Two linear layers with ReLU activation
- Output: log-softmax over 5 classes, trained with NLL loss
- Part 1 of the assignment involved debugging a provided buggy FFNN implementation — bugs included incorrect output layer dimensionality, wrong activation ordering, and missing eval mode during validation

### Recurrent Neural Network (RNN)
- Word-level input: each word in the review maps to a learned embedding
- Single-layer RNN with ReLU nonlinearity, trained with minibatch SGD
- Classification uses the final hidden state passed through a linear layer + log-softmax
- Early stopping based on validation accuracy trend across epochs

## Files
- `data_loader.py` — loads and preprocesses Yelp training/validation JSON data
- `ffnn.py` — original (buggy) FFNN implementation provided by course staff
- `ffnn1fix.py` — debugged and corrected FFNN implementation
- `rnn.py` — RNN implementation built from scratch
- `main.py` — entry point; toggle between FFNN and RNN via the `FLAG` variable; also runs FFNN across hidden dims (32, 64, 128) for ablation comparison
- `training.json` / `validation.json` — Yelp review datasets (not included; obtained via course)

## How to Run
```bash
# Toggle FLAG in main.py to 'RNN' or 'FFNN', then:
python main.py
```

Requires: PyTorch, numpy, tqdm

## Key Design Decisions
- **Minibatch SGD** (batch size 16) with momentum 0.9 for stable training
- **Learned embeddings** for the RNN rather than pretrained vectors, to keep the implementation self-contained
- **Early stopping** triggered when validation accuracy drops by more than 0.9% over two consecutive epochs
- FFNN hidden dimensionality compared across 32, 64, and 128 units to study capacity vs. generalization tradeoff

## Technologies
Python · PyTorch · NumPy
