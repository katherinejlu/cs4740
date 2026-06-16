import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
from torch.autograd import Variable
from torch.nn import Embedding


import time
from tqdm import tqdm
from data_loader import fetch_data
from ffnn1fix import convert_to_vector_representation, make_vocab, make_indices

unk = '<UNK>'


class RNN(nn.Module):
    def __init__(self, h, input_dim): # Add relevant parameters
        super(RNN, self).__init__()
        # Fill in relevant parameters
        # Ensure parameters are initialized to small values, see PyTorch documentation for guidance
        self.embedding = Embedding(input_dim,h)
        self.outsize = 5 
        self.batch = 16
        self.h = Variable(torch.zeros(1,1,h))
        #self.W1 = nn.Linear(input_dim,h)
        self.W2 = nn.Linear(h,self.outsize)
        self.activation = nn.ReLU() 
        self.rnn = nn.RNN(input_size = h, hidden_size = h, nonlinearity = 'relu', batch_first = True )
            
        self.softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss()

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

