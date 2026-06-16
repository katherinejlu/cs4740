import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from torch.autograd import Variable
from torch.nn import Embedding

from data_loader import fetch_data
from ffnn1fix import make_vocab, make_indices

UNK = '<UNK>'
HIDDEN_DIM = 64
NUM_EPOCHS = 10
MINIBATCH_SIZE = 16
LEARNING_RATE = 0.01
EARLY_STOP_THRESHOLD = 0.009


class RNN(nn.Module):
    """
    Single-layer RNN for 5-class sentiment classification on Yelp reviews.
    Input: sequence of word indices
    Architecture: learned embeddings -> RNN -> linear classifier -> log-softmax
    Classification uses the final hidden state as the sequence representation.
    """
    def __init__(self, hidden_dim, vocab_size):
        super(RNN, self).__init__()
        self.embedding = Embedding(vocab_size, hidden_dim)
        self.rnn = nn.RNN(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            nonlinearity='relu',
            batch_first=True
        )
        self.h0 = Variable(torch.zeros(1, 1, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_loss(self, predicted, gold_label):
        return self.loss(predicted, gold_label)

    def forward(self, sentence, word2index):
        # Convert words to indices, using UNK for out-of-vocabulary words
        indices = [word2index.get(word, word2index[UNK]) for word in sentence]
        embedded = self.embedding(torch.LongTensor([indices]))
        rnn_out, _ = self.rnn(embedded, self.h0)

        # Use final hidden state as sequence representation
        final_hidden = rnn_out[0][-1].unsqueeze(0)
        return self.softmax(self.classifier(final_hidden))


def train_epoch(model, optimizer, train_data, word2index, epoch):
    model.train()
    correct, total = 0, 0
    start = time.time()
    random.shuffle(train_data)

    for batch_start in range(0, len(train_data) - MINIBATCH_SIZE, MINIBATCH_SIZE):
        optimizer.zero_grad()
        loss = None

        for i in range(MINIBATCH_SIZE):
            sentence, gold_label = train_data[batch_start + i]
            predicted = model(sentence, word2index)

            z = np.zeros(5)
            for vec in predicted:
                z[torch.argmax(vec).item()] += 1
            predicted_label = torch.argmax(torch.Tensor(z)).item()

            correct += int(predicted_label == gold_label)
            total += 1

            example_loss = model.compute_loss(predicted.view(1, -1), torch.tensor([gold_label]))
            loss = example_loss if loss is None else loss + example_loss

        (loss / MINIBATCH_SIZE).backward()
        optimizer.step()

    print(f"Epoch {epoch} | Train accuracy: {correct/total:.4f} | Time: {time.time()-start:.1f}s")


def validate_epoch(model, optimizer, valid_data, word2index, epoch):
    model.eval()
    correct, total = 0, 0
    start = time.time()

    for batch_start in range(0, len(valid_data) - MINIBATCH_SIZE, MINIBATCH_SIZE):
        optimizer.zero_grad()
        loss = None

        for i in range(MINIBATCH_SIZE):
            sentence, gold_label = valid_data[batch_start + i]
            predicted = model(sentence, word2index)

            z = np.zeros(5)
            for vec in predicted:
                z[torch.argmax(vec).item()] += 1
            predicted_label = torch.argmax(torch.Tensor(z)).item()

            correct += int(predicted_label == gold_label)
            total += 1

            example_loss = model.compute_loss(predicted.view(1, -1), torch.tensor([gold_label]))
            loss = example_loss if loss is None else loss + example_loss

        (loss / MINIBATCH_SIZE).backward()
        optimizer.step()

    accuracy = correct / total
    print(f"Epoch {epoch} | Val accuracy: {accuracy:.4f} | Time: {time.time()-start:.1f}s")
    return accuracy


def main(hidden_dim=HIDDEN_DIM, number_of_epochs=NUM_EPOCHS):
    print(f"Training RNN | hidden_dim={hidden_dim} | lr={LEARNING_RATE}")

    train_data, valid_data = fetch_data()
    vocab = make_vocab(train_data)
    vocab, word2index, _ = make_indices(vocab)

    model = RNN(hidden_dim=hidden_dim, vocab_size=len(vocab))
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    prev_val = 0
    two_prev_val = 0

    for epoch in range(1, number_of_epochs + 1):
        # Early stopping: halt if validation accuracy drops consistently
        if epoch > 2 and (prev_val - two_prev_val) <= -EARLY_STOP_THRESHOLD * 2:
            print(f"Early stopping at epoch {epoch}")
            break

        train_epoch(model, optimizer, train_data, word2index, epoch)
        val_acc = validate_epoch(model, optimizer, valid_data, word2index, epoch)

        two_prev_val = prev_val
        prev_val = val_acc


if __name__ == '__main__':
    main()
    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)    

    def forward(self, sentence, word2index): 
        #begin code
        x = [] 
        for word in sentence: 
            x.append(word2index.get(word, word2index[unk]))
        y = self.embedding(torch.LongTensor([x]))
        z1,_ = self.rnn(y,self.h)
        z2 = self.W2(z1[len(z1)-1])
        predicted_vector = self.softmax(z2)
        # Remember to include the predicted unnormalized scores which should be normalized into a (log) probability distribution
        #end code
        return predicted_vector

# You may find the functions make_vocab() and make_indices from ffnn.py useful; you are free to copy them directly (or call those functions from this file)



def main(hidden_dim, number_of_epochs): # Add relevant parameters
    train_data, valid_data = fetch_data() # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    print("THIS IS RNN h = 128 lr = 0.01")
    print("Fetching data")
  
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    
    print("Fetched and indexed data")
#    train_data = convert_to_vector_representation(train_data, word2index)
#    valid_data = convert_to_vector_representation(valid_data, word2index)
    print("Vectorized data")


    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further. 
    # Option 3 will be the most time consuming, so we do not recommend starting with this
    minibatch_size = 16
    model = RNN(h = hidden_dim, input_dim = len(vocab)) # Fill in parameters
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
    
#    stopping_condition = False
#    
#    while not stopping_condition: # How will you decide to stop training and why
    twoprev_validscore = 0
    validationscore = 0
    prev_validscore = 0 
        # You will need further code to operationalize training, ffnn.py may be helpful
    for epoch in range(number_of_epochs):
        if (validationscore - twoprev_validscore) <= -0.009*2: 
            break
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
 
        N = len(train_data) 
        for minibatch_index in range(N // minibatch_size):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]

                predicted_vector = model(input_vector, word2index)
                z = np.zeros(5)
                for x in range(len(predicted_vector)): 
                    z[torch.argmax(predicted_vector[x]).item()] +=1
                predicted_label = torch.argmax(torch.Tensor(z)).item()
                    
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
            
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))
        model.eval() 
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        random.shuffle(valid_data) # Good practice to shuffle order of validation data
        minibatch_size = 16 
        N = len(valid_data) 
        for minibatch_index in range(N // minibatch_size):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector, word2index)
                z = np.zeros(5)
                for x in range(len(predicted_vector)): 
                    z[torch.argmax(predicted_vector[x]).item()] +=1
                predicted_label = torch.argmax(torch.Tensor(z)).item()
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size 
            loss.backward()
            optimizer.step()
            if epoch > 0 : 
                twoprev_validscore = prev_validscore
                prev_validscore = validationscore 
            validationscore = correct / total 
            
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

        # You may find it beneficial to keep track of training accuracy or training loss; 

        # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance

        # You will need to validate your model. All results for Part 3 should be reported on the validation set. 
        # Consider ffnn.py; making changes to validation if you find them necessary

