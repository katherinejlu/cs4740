import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

from data_loader import fetch_data

UNK = '<UNK>'
MINIBATCH_SIZE = 16
LEARNING_RATE = 0.01

POS_TAGS = [
    "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
    "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR",
    "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP",
    "VBZ", "WDT", "WP", "WP$", "WRB"
]


class FFNN(nn.Module):
    """
    Feedforward Neural Network for 5-class sentiment classification.
    Input: bag-of-words vector over vocabulary
    Architecture: linear -> ReLU -> linear -> log-softmax
    Trained with negative log-likelihood loss.
    """
    def __init__(self, input_dim, hidden_dim):
        super(FFNN, self).__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.W2 = nn.Linear(hidden_dim, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_loss(self, predicted, gold_label):
        return self.loss(predicted, gold_label)

    def forward(self, input_vector):
        z1 = self.activation(self.W1(input_vector))
        z2 = self.W2(z1)
        return self.softmax(z2)


def make_vocab(data):
    """
    Build vocabulary from training data.
    Includes all words in the corpus plus a fixed set of POS tags
    and similarity markers used as additional features.
    """
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    for tag in POS_TAGS:
        vocab.add(tag)
    vocab.add('<similar>')
    vocab.add('<different>')
    return vocab


def make_indices(vocab):
    """
    Assign integer indices to vocabulary items.
    Returns updated vocab (with UNK), word-to-index, and index-to-word mappings.
    """
    vocab_list = sorted(vocab) + [UNK]
    word2index = {word: i for i, word in enumerate(vocab_list)}
    index2word = {i: word for i, word in enumerate(vocab_list)}
    vocab.add(UNK)
    return vocab, word2index, index2word


def vectorize(data, word2index):
    """
    Convert a list of (document, label) pairs into bag-of-words vectors.
    Unknown words are mapped to the UNK index.
    """
    vectorized = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[UNK])
            vector[index] += 1
        vectorized.append((vector, y))
    return vectorized


def main(train_data, valid_data, hidden_dim, number_of_epochs):
    """
    Train and evaluate the FFNN on sentiment classification.
    Returns a list of per-example correctness on the validation set (1=correct, 0=wrong),
    used for cross-model comparison in main.py.
    """
    vocab = make_vocab(train_data)
    vocab, word2index, _ = make_indices(vocab)

    train_data = vectorize(train_data, word2index)
    valid_data = vectorize(valid_data, word2index)

    model = FFNN(input_dim=len(vocab), hidden_dim=hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    for epoch in range(number_of_epochs):
        # --- Training ---
        model.train()
        correct, total = 0, 0
        start = time.time()
        N = len(train_data)

        for batch_start in range(min(128, N // MINIBATCH_SIZE)):
            optimizer.zero_grad()
            loss = None

            for i in range(MINIBATCH_SIZE):
                input_vector, gold_label = train_data[batch_start * MINIBATCH_SIZE + i]
                predicted = model(input_vector)
                predicted_label = torch.argmax(predicted)

                correct += int(predicted_label == gold_label)
                total += 1

                example_loss = model.compute_loss(predicted.view(1, -1), torch.tensor([gold_label]))
                loss = example_loss if loss is None else loss + example_loss

            (loss / MINIBATCH_SIZE).backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Train accuracy: {correct/total:.4f} | Time: {time.time()-start:.1f}s")

        # --- Validation ---
        model.eval()
        correct, total = 0, 0
        start = time.time()
        N = len(valid_data)
        results = []

        for batch_start in range(min(128, N // MINIBATCH_SIZE)):
            optimizer.zero_grad()
            loss = None

            for i in range(MINIBATCH_SIZE):
                input_vector, gold_label = valid_data[batch_start * MINIBATCH_SIZE + i]
                predicted = model(input_vector)
                predicted_label = torch.argmax(predicted)
                is_correct = int(predicted_label == gold_label)

                correct += is_correct
                total += 1
                results.append(is_correct)

                example_loss = model.compute_loss(predicted.view(1, -1), torch.tensor([gold_label]))
                loss = example_loss if loss is None else loss + example_loss

            (loss / MINIBATCH_SIZE).backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Val accuracy: {correct/total:.4f} | Time: {time.time()-start:.1f}s")

    return results
