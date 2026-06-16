import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import nltk
from torch.autograd import Variable
from torch.nn import Embedding
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from ffnn1fix import make_vocab, make_indices

# Paths
TRAIN_PATH = 'train.csv'
VAL_PATH   = 'dev.csv'
TEST_PATH  = 'test.csv'
GLOVE_PATH = 'glove.6B.100d.txt'

# Hyperparameters
HIDDEN_DIM     = 100
NUM_EPOCHS     = 10
MINIBATCH_SIZE = 16
LEARNING_RATE  = 0.01

# Sentiment similarity threshold: stories with sentiment shift > this
# are marked as <different>, otherwise <similar>
SENTIMENT_THRESHOLD = 0.5

STORY_COLUMNS = [
    'InputStoryid', 'InputSentence1', 'InputSentence2',
    'InputSentence3', 'InputSentence4',
    'RandomFifthSentenceQuiz1', 'RandomFifthSentenceQuiz2', 'AnswerRightEnding'
]


def load_glove(glove_path):
    """
    Convert GloVe vectors to word2vec format and load with gensim.
    Returns the KeyedVectors model and a FloatTensor of the weight matrix.
    """
    tmp_file = get_tmpfile('temp_word2vec.txt')
    glove2word2vec(glove_path, tmp_file)
    gmodel = KeyedVectors.load_word2vec_format(tmp_file)
    weights = torch.FloatTensor(gmodel.vectors)
    return gmodel, weights


def build_story_pairs(df, gmodel):
    """
    Build (story_text, label) pairs from a story cloze dataframe.
    Each story has two candidate endings; the correct one is labeled 1,
    the incorrect one is labeled 0. Both are included as training examples.
    """
    data = []
    for i in range(len(df)):
        context = ' '.join([
            df.InputSentence1[i], df.InputSentence2[i],
            df.InputSentence3[i], df.InputSentence4[i]
        ])
        quiz1 = df.RandomFifthSentenceQuiz1[i]
        quiz2 = df.RandomFifthSentenceQuiz2[i]

        if int(df.AnswerRightEnding[i]) == 1:
            data.append((context + ' ' + quiz1, 1))
            data.append((context + ' ' + quiz2, 0))
        else:
            data.append((context + ' ' + quiz2, 1))
            data.append((context + ' ' + quiz1, 0))
    return data


def build_test_pairs(df):
    """
    Build candidate ending pairs for test stories (no labels).
    Returns flat list alternating ending1, ending2 for each story.
    """
    data = []
    for i in range(len(df)):
        context = ' '.join([
            df.InputSentence1[i], df.InputSentence2[i],
            df.InputSentence3[i], df.InputSentence4[i]
        ])
        data.append(context + ' ' + df.RandomFifthSentenceQuiz1[i])
        data.append(context + ' ' + df.RandomFifthSentenceQuiz2[i])
    return data


class RNN(nn.Module):
    """
    Bidirectional RNN with GloVe embeddings for binary story ending classification.

    Input: a full story sentence (context + one candidate ending)
    Features: GloVe word embeddings + POS tag embeddings + sentiment shift marker
    Architecture: pretrained GloVe embeddings (fine-tuned) -> biRNN -> linear -> log-softmax

    The sentiment shift marker (<similar> or <different>) is appended after the
    4th sentence boundary, capturing the emotional arc of the story as a feature.
    """
    def __init__(self, hidden_dim, vocab_size, weights):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.outsize = 2

        # Initialize embeddings from GloVe, allow fine-tuning
        embed = Embedding(vocab_size, hidden_dim)
        embed.from_pretrained(weights, freeze=False)
        self.embed = embed
        self.weight = nn.Parameter(embed.weight)

        # Bidirectional RNN: output dim is hidden_dim * 2
        self.rnn = nn.RNN(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            nonlinearity='relu',
            batch_first=True,
            bidirectional=True
        )
        self.h0 = Variable(torch.zeros(2, 1, hidden_dim))
        self.classifier = nn.Linear(hidden_dim * 2, self.outsize)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_loss(self, predicted, gold_label):
        return self.loss(predicted, gold_label)

    def forward(self, sentence, gmodel):
        tokens = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)
        sid = SentimentIntensityAnalyzer()

        indices = []
        sentence_count = 0

        for k, token in enumerate(tokens):
            # Handle OOV words and POS tags
            if token not in gmodel.vocab:
                gmodel.vocab[token] = gmodel.vocab['random']
            pos = pos_tags[k][1]
            if pos not in gmodel.vocab:
                gmodel.vocab[pos] = gmodel.vocab['at']

            indices.append(gmodel.vocab[token].index)
            indices.append(gmodel.vocab[pos].index)

            # Append sentiment shift marker after 4th sentence boundary
            if token == '.':
                sentence_count += 1
            if sentence_count == 4:
                sent1 = ' '.join(tokens[:k])
                sent2 = ' '.join(tokens[k+1:])
                shift = abs(
                    sid.polarity_scores(sent1)['compound'] -
                    sid.polarity_scores(sent2)['compound']
                )
                marker = '<similar>' if shift < SENTIMENT_THRESHOLD else '<different>'
                if marker not in gmodel.vocab:
                    gmodel.vocab[marker] = gmodel.vocab['the']
                indices.append(gmodel.vocab[marker].index)
                sentence_count += 1  # prevent re-triggering

        embedded = self.embed(torch.LongTensor([indices]))
        rnn_out, _ = self.rnn(embedded, self.h0)
        final = rnn_out[0][-1].unsqueeze(0)
        return self.softmax(self.classifier(final))


def predict(model, sentences, gmodel):
    """Run model inference on a list of sentences. Returns predictions and scores."""
    predictions, scores = [], []
    for sentence in sentences:
        predicted = model(sentence, gmodel)
        z = np.zeros(2)
        for vec in predicted:
            z[torch.argmax(vec).item()] += 1
        predictions.append(torch.argmax(torch.Tensor(z)).item())
        scores.append(torch.max(torch.Tensor(z)).item())
    return predictions, scores


def write_predictions(filename, predictions, scores, story_ids):
    """
    Write test predictions to CSV.
    Each story has two candidate endings (even/odd index pairs).
    If models disagree, use the predicted label directly.
    If both agree, use the confidence score to break the tie.
    """
    with open(filename, 'w') as f:
        f.write('Id,Prediction\n')
        for i in range(len(story_ids)):
            p1, p2 = predictions[2*i], predictions[2*i+1]
            s1, s2 = scores[2*i], scores[2*i+1]

            if p1 != p2:
                answer = 1 if p1 == 1 else 2
            elif p1 == 1:
                answer = 1 if s1 > s2 else 2
            else:
                answer = 2 if s1 > s2 else 1

            f.write(f"1,{story_ids[i]}\n" if answer == 1 else f"2,{story_ids[i]}\n")


def main(hidden_dim=HIDDEN_DIM, number_of_epochs=NUM_EPOCHS):
    # Load GloVe embeddings
    gmodel, weights = load_glove(GLOVE_PATH)

    # Load datasets
    train_df = pd.read_csv(TRAIN_PATH, encoding='latin1', names=STORY_COLUMNS, skiprows=1)
    val_df   = pd.read_csv(VAL_PATH,   encoding='latin1', names=STORY_COLUMNS, skiprows=1)
    test_df  = pd.read_csv(TEST_PATH,  encoding='latin1', names=STORY_COLUMNS, skiprows=1)

    train_data = build_story_pairs(train_df, gmodel)
    valid_data = build_story_pairs(val_df, gmodel)
    test_data  = build_test_pairs(test_df)

    model = RNN(hidden_dim=hidden_dim, vocab_size=len(gmodel.vocab), weights=weights)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    for epoch in range(number_of_epochs):
        # --- Training ---
        model.train()
        correct, total = 0, 0
        start = time.time()

        for batch_start in range(len(train_data) // MINIBATCH_SIZE):
            optimizer.zero_grad()
            loss = None

            for i in range(MINIBATCH_SIZE):
                sentence, gold_label = train_data[batch_start * MINIBATCH_SIZE + i]
                predicted = model(sentence, gmodel)

                z = np.zeros(2)
                for vec in predicted:
                    z[torch.argmax(vec).item()] += 1
                predicted_label = torch.argmax(torch.Tensor(z)).item()

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
        random.shuffle(valid_data)

        for batch_start in range(len(valid_data) // MINIBATCH_SIZE):
            optimizer.zero_grad()
            loss = None

            for i in range(MINIBATCH_SIZE):
                sentence, gold_label = valid_data[batch_start * MINIBATCH_SIZE + i]
                predicted = model(sentence, gmodel)

                z = np.zeros(2)
                for vec in predicted:
                    z[torch.argmax(vec).item()] += 1
                predicted_label = torch.argmax(torch.Tensor(z)).item()

                correct += int(predicted_label == gold_label)
                total += 1

                example_loss = model.compute_loss(predicted.view(1, -1), torch.tensor([gold_label]))
                loss = example_loss if loss is None else loss + example_loss

            (loss / MINIBATCH_SIZE).backward()
            optimizer.step()

        val_acc = correct / total
        print(f"Epoch {epoch+1} | Val accuracy: {val_acc:.4f} | Time: {time.time()-start:.1f}s")

        # --- Test predictions ---
        predictions, scores = predict(model, test_data, gmodel)
        filename = f"predict_epoch{epoch+1}_val{val_acc:.4f}.csv"
        write_predictions(filename, predictions, scores, test_df.InputStoryid.tolist())
        print(f"Predictions written to {filename}")


if __name__ == '__main__':
    main()
pos_list = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]


#glove = torchtext.vocab.GloVe(name='6B', dim = 50)
#print(glove.shape())
#pickle.dump(glove, open("save.p", "wb"))
path = 'train.csv'
vpath = 'dev.csv'
test_path = 'test.csv'
data = pd.read_csv(path, encoding = 'latin1', names = ['InputStoryid','InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4', 'RandomFifthSentenceQuiz1', 'RandomFifthSentenceQuiz2',  'AnswerRightEnding'], skiprows = 1 )
vdata = pd.read_csv(vpath, encoding = 'latin1', names = ['InputStoryid','InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4', 'RandomFifthSentenceQuiz1', 'RandomFifthSentenceQuiz2',  'AnswerRightEnding'], skiprows = 1 )
test = pd.read_csv(test_path, encoding = 'latin1', names = ['InputStoryid','InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4', 'RandomFifthSentenceQuiz1', 'RandomFifthSentenceQuiz2',  'AnswerRightEnding'], skiprows = 1 )

unk = '<UNK>'
#f = open('glove.6B.100d.txt')
#contents = f.readlines()
#glove = torchtext.data.Field(lower=True, batch_first=True)
#glove.build_vocab(vectors = GloVe(name='6B', dim=100))
#weights = glove.vocab.vectors

#model = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.100d.txt')
#model = gensim.models.Word2Vec.load('glove.6B.100d.txt')
tmp_file = get_tmpfile('temp_word2vec.txt')
glove2word2vec('glove.6B.100d.txt', tmp_file)
gmodel = KeyedVectors.load_word2vec_format(tmp_file)
weights = torch.FloatTensor(gmodel.vectors)
print(weights)
#vocab = make_vocab(contents)
#vocab, word2index, index2word = make_indices(vocab)

class RNN(nn.Module):
    def __init__(self, h, input_dim): # Add relevant parameters
        super(RNN, self).__init__()
        # Fill in relevant parameters
        # Ensure parameters are initialized to small values, see PyTorch documentation for guidance
#        self.embedding = Embedding(input_dim,h)

        embed = Embedding(input_dim, h)

        embed.from_pretrained(weights, freeze=False)
        self.embed=embed
        self.weight = nn.Parameter(embed.weight)
        self.outsize = 2
        self.batch = 16
        self.h = Variable(torch.zeros(2,1,h))
        #self.W1 = nn.Linear(input_dim,h)
        self.W2 = nn.Linear(h*2,self.outsize)
        print(self.W2)
        self.activation = nn.ReLU() 
        self.rnn = nn.RNN(input_size = h, hidden_size = h, nonlinearity = 'relu', batch_first = True, bidirectional = True )
        
        self.softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss()
        

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)    

    def forward(self, sentence): 
        #begin code
        x = [] 
        
        count = 0
        text = nltk.word_tokenize(sentence)

        pos = nltk.pos_tag(text)
        count2=0
        gmodel.vocab['<similar>']=gmodel.vocab['the']
        gmodel.vocab['<different>']=gmodel.vocab['the']
        
        for i in text: 
            
            if i not in gmodel.vocab.keys():
                gmodel.vocab[i]=gmodel.vocab["random"]
            if pos[count2][1] not in gmodel.vocab.keys():
                gmodel.vocab[pos[count2][1]]=gmodel.vocab["at"]


            x.append(gmodel.vocab[i].index)
            #x.append(word2index.get(text[i], word2index[unk]))
            x.append(gmodel.vocab[pos[count2][1]].index)
            #x.append(word2index.get(pos[i][1], 0))
            print(i)
            print()
            print(gmodel.vectors[gmodel.vocab[i].index][0:3])
            print("-----------------------------------------------------")
            if (i == '.') : 
                count +=1
            if count == 4: 
                sid = SentimentIntensityAnalyzer()
                sid2 = SentimentIntensityAnalyzer()
                sent1 = ' '.join(text[0:gmodel.vocab[i].index])
                sent2 = ' '.join(text[gmodel.vocab[i].index+1:])
                ss = sid.polarity_scores(sent1)
                ss2 = sid2.polarity_scores(sent2)
                comp = abs(ss['compound'] - ss2['compound']) 
                if comp < 0.5: 
                    x.append(gmodel.vocab['<similar>'].index)
                else: 
                    x.append(gmodel.vocab['<different>'].index)
                count+=1 
           # sentiment.append()
            count2+=1
        y = self.embed(torch.LongTensor([x]))

#        print(pos[0][1])
        
#            
#        for j in range(len(y)): 
#            for i in range(len(pos_list)): 
#                if pos[j][1] == pos_list[i]: 
#                    onehot = np.zeros(len(pos_list))
#                    onehot[i] = 1
#            y[j] = torch.cat((y[j],torch.Tensor(onehot)))
        z1,_ = self.rnn(y,self.h)
        z2 = self.W2(z1[len(z1)-1])
        predicted_vector = self.softmax(z2)
        # Remember to include the predicted unnormalized scores which should be normalized into a (log) probability distribution
        #end code
        return predicted_vector

# You may find the functions make_vocab() and make_indices from ffnn.py useful; you are free to copy them directly (or call those functions from this file)



def main(hidden_dim, number_of_epochs): # Add relevant parameters
        #////// 
    train_data = []
    valid_data = []
    test_data = []
    for x in range(len(data.InputStoryid)): 
        sentence = data.InputSentence1[x] + ' ' + data.InputSentence2[x] + ' ' + data.InputSentence3[x] + ' ' + data.InputSentence4[x]
 
        trueSentence = sentence

        if int(data.AnswerRightEnding[x]) == 1: 
            trueSentence += ' ' + data.RandomFifthSentenceQuiz1[x]
            
            train_data.append((trueSentence,1))
            train_data.append((sentence + ' ' + data.RandomFifthSentenceQuiz2[x], 0))

        else: 
            trueSentence += ' ' + data.RandomFifthSentenceQuiz2[x]

            
            train_data.append((trueSentence,1))
            train_data.append((sentence + ' ' + data.RandomFifthSentenceQuiz1[x], 0))

            
    for x in range(len(vdata.InputStoryid)): 
        
        vsentence = vdata.InputSentence1[x] + ' ' + vdata.InputSentence2[x] + ' ' + vdata.InputSentence3[x] + ' ' + vdata.InputSentence4[x]
        
        vtrueSentence = vsentence

        if int(vdata.AnswerRightEnding[x]) == 1: 

            vtrueSentence += ' ' + vdata.RandomFifthSentenceQuiz1[x]

            valid_data.append((vtrueSentence,1))
            valid_data.append((vsentence + ' ' + vdata.RandomFifthSentenceQuiz2[x], 0))
        else: 

            vtrueSentence += ' ' + vdata.RandomFifthSentenceQuiz2[x]
            
            valid_data.append((vtrueSentence,1))
            valid_data.append((vsentence + ' ' + vdata.RandomFifthSentenceQuiz1[x], 0))
    
    for x in range(len(test.InputStoryid)): 
        
        tsentence = test.InputSentence1[x] + ' ' + test.InputSentence2[x] + ' ' + test.InputSentence3[x] + ' ' + test.InputSentence4[x]
        
        ending1 = tsentence + ' ' + test.RandomFifthSentenceQuiz1[x]
        ending2 = tsentence + ' ' + test.RandomFifthSentenceQuiz2[x]
        
        test_data.append(ending1)
        test_data.append(ending2)

    
    #////   
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
    model = RNN(h = hidden_dim, input_dim = len(gmodel.vocab)) # Fill in parameters
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
#    print((model.parameters()))
    
#    stopping_condition = False
#    
#    while not stopping_condition: # How will you decide to stop training and why

        # You will need further code to operationalize training, ffnn.py may be helpful
    for epoch in range(number_of_epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        
        print("Training started for epoch {}".format(epoch + 1))
        print("weights this epoch")
        print(model.weight)

#        random.shuffle(train_data) # Good practice to shuffle order of training data
        loss=0
        N = len(train_data) 
        for minibatch_index in range(N // minibatch_size):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                z = np.zeros(2)
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
        print("weights after epoch")
        print(model.weight)
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
                predicted_vector = model(input_vector)
                z = np.zeros(2)
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
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))
# if the labels are the different, look at the index of the 1 label and if it is even give 1, if odd give 2 
# if both 0, look at the torch.max of the vector and whichever is greater is false, so look at index of lesser value, even or odd 
# if both 1, look at torch.max and whichever is greater is true, look at index, even or odd? 
        N = len(test_data) 
        predictions = [] 
        score = []
        id_list = test.InputStoryid
        
        for example_index in range(len(test_data)):
            input_vector = test_data[example_index]
            
            predicted_vector = model(input_vector)
            z = np.zeros(2)
            for x in range(len(predicted_vector)): 
                z[torch.argmax(predicted_vector[x]).item()] +=1
            predicted_label = torch.argmax(torch.Tensor(z)).item()
            predictions.append(predicted_label)
            score.append(torch.max(torch.Tensor(z)).item())
        filename = 'predict' + str(epoch) + '-' + str(correct / total) + '.csv'
        print(len(predictions))
        print(len(id_list))
        with open(filename, 'w') as f:
            f.write('Id,Prediction\n')
            for i in range(len(id_list)): 
                if predictions[2*i] != predictions[2*i+1]: 
                    if predictions[2*i] == 1: 
                        f.write("1," + id_list[i] + "\n")
                    else: 
                        f.write("2," + id_list[i] + "\n")
                elif predictions[2*i] == 1: 
                    if score[2*i] > score[2*i+1]: 
                        f.write("1," + id_list[i] + "\n")
                    else: 
                        f.write("2," + id_list[i] + "\n")
                else: 
                    if score[2*i] > score[2*i+1]: 
                        f.write("2," + id_list[i] + "\n")
                    else: 
                        f.write("1," + id_list[i] + "\n")
        
    #////// 
#    train_data = []
#
#
#    for x in range(len(data.InputStoryid)): 
#        sentence = data.InputSentence1[x] + ' ' + data.InputSentence2[x] + ' ' + data.InputSentence3[x] + ' ' + data.InputSentence4[x]
#
#
#        trueSentence = sentence
#        if int(data.AnswerRightEnding[x]) == 1: 
#            trueSentence += ' ' + data.RandomFifthSentenceQuiz1[x]
#            train_data.append((trueSentence,1))
#            train_data.append((sentence + ' ' + data.RandomFifthSentenceQuiz2[x], 0))
#        else: 
#            trueSentence += ' ' + data.RandomFifthSentenceQuiz2[x]
#            train_data.append((trueSentence,1))
#            train_data.append((sentence + ' ' + data.RandomFifthSentenceQuiz1[x], 0))
                
    #////         
