#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import collections
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import math

train_path = "DATASET/train/truthful.txt"
test_path = "DATASET/test/test.txt"
## capitalization? punctuation? contractions?

def make_Unigram(filepath):
    word_count = 0
    smoothing = 0.1
    count = 0
    d = collections.defaultdict(int)
    d['<unk>'] = 0
    with open(filepath) as f:
        for line in f:
            for word in line.split():
                word_count += 1
                d[word.lower()]+=1
##                if word in list(d):
##                    d[word] += 1
##                else:
##                    d['<unk>'] += 0.5
##                    d[word] += 0.5
    d['<unk>'] = smoothing*len(d)
    word_count += d['<unk>']

    #print (d.items())
    for k in d:
    #    print (d[k])
        d[k] = d[k]/word_count #corpus length includes punctation?
        #count+= d[k]
    #print (count)
    return d, word_count

def make_Bigram(filepath):
    smoothing = .01
    acc = 0;
    last_word = '<LASTWORDINITIALIZE>'
    word_count = 0
    d = collections.defaultdict(lambda:collections.defaultdict(int))
    with open(filepath) as f:
        for line in f:
            for word in line.split():
                word = word.lower()
                word_count +=1
                if last_word == '<LASTWORDINITIALIZE>' :
                    last_word = word
                    continue
                
                d[last_word][word]+=1
                last_word = word
    d['<unk>']['<unk>'] = len(d) * smoothing


  ### SHOULD SMOOTHING BE NUM_BIGRAMS = sum(len(bigrams) for bigrams in d.values())
    # d['<unk>']['<unk>'] = NUM_BIGRAMS*smoothing
    # len(d) is just number of unique tokens 
    
    for k in d:
        d[k]['<unk>'] = len(d) * smoothing
        #print (d[k]['<unk>'])
        acc = 0 
        for i in d[k]:
            acc+= d[k][i]
        for j in d[k]:

            d[k][j] = d[k][j]/acc

            #print (k,j, d[k][j])
    #print (d)
    return d, word_count

def perplexity(d, filepath, ngram):
    perplex = 0
    word_count1 = 0
    last_word = '<LASTWORDINITIALIZE>'
    with open(filepath) as f:
        for line in f: 
            for word in line.split():
                word = word.lower()
                word_count1 += 1
                if ngram == 1:
                    if word in d:
                        perplex -= math.log(d[word])
                    else:
                        perplex -= math.log(d['<unk>'])

                ##BIGRAM
                else:
                    if last_word == '<LASTWORDINITIALIZE>' :
                        last_word = word
                        continue
                    if last_word in d:
                        if word in d[last_word]:
                            perplex += -math.log(d[last_word][word])
                        else:
                            perplex += -math.log(d[last_word]['<unk>'])
                    else:
                        perplex += -math.log(d['<unk>']['<unk>'])
                    last_word = word
    ## look at keys from the testing 
##    for k in d:
##        perplex += -math.log(d[k])*d[k]*word_count1
                ## this is wrong bc its not total tokens, it has
                ## it has smoothing
                

    perplex = math.exp(perplex/word_count1)
##    print(perplex) 

    return perplex 
    
##def perplexity2(word_count,d):
##    perplex = 0
##    for k in d:
##        for j in d[k]:
##            perplex += -math.log(d[k][j])
##    print (word_count)
##
##    perplex = math.exp(perplex/word_count)
##
##    return perplex


truth_test1, word_count1 = make_Unigram(train_path)
truth_test2, word_count2 = make_Bigram(train_path)
p1 = perplexity(truth_test1, test_path, 1)
p2 = perplexity(truth_test2, test_path, 2)



print (p1,p2)
