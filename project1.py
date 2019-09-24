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

filepath = "DATASET/train/truthful.txt"

def make_Unigram(filepath):
    word_count = 0
    d = collections.defaultdict(int)
    d['<unk>'] = 0
    with open(filepath) as f:
        for line in f:
            for word in line.split():
                word_count += 1
                if word in list(d):
                    d[word] += 1
                   # print('got here')
                else:
                    d['<unk>'] += 0.5
                    d[word] += 0.5
                    #print ('hello')
    print (d['<unk>'])
    for k in d:
    #    print (d[k])
        d[k] = d[k]/word_count #corpus length includes punctation?
    print (word_count, len(list(d)), (d['<unk>']))
    return d, word_count

def make_Bigram(filepath):
    smoothing = .0001 #0.003
    acc = 0
    acc_init = 0
    last_word = 'grgsgtrgtsyhtsujts'
    word_count = 0

    N0 = 0; # number of zero count bigrams
    N1 = 0; # number of single count bigrams
    N2 = 0;
    N3 = 0;
    N4 = 0;
    N5 = 0; 
    d = collections.defaultdict(lambda:collections.defaultdict(int))
    with open(filepath) as f:
        for line in f:
            for word in line.split():
                if last_word == 'grgsgtrgtsyhtsujts' :
                    last_word = word
                    continue
                word_count +=1
                d[last_word][word]+=1
                last_word = word
    d['<unk>']['<unk>'] = len(list(d)) * smoothing
    for k in d:
        d[k]['<unk>'] = len(list(d)) * smoothing
        #print (d[k]['<unk>'])
        ### GOOD TURING SMOOTHING ATTEMPT
        for i in d[k]:
            acc_init += d[k][i]
            if d[k][i] == 0:
                N0 += 1 
            if d[k][i] == 1:
                N1 += 1
            if d[k][i] == 2:
                N2 += 1
            if d[k][i] == 3:
                N3 += 1
            if d[k][i] == 4: 
                N4 += 1
            if d[k][i] == 5:
                N5 += 1;
    #print(N0, N1, N2, N3, N4, N5)
        ### DONE COUNTING UNRELIABLE BIGRAM COUNTS
        ### CALULATE NEW COUNTS WITH GOOD TURING EQ
    for k in d:
        for j in d[k]:
            if d[k][j] == 0:
                d[k][j] = (d[k][j] + 1)*(N1/N0)
            if d[k][j] == 1: 
                d[k][j] = (d[k][j] + 1)*(N2/N1)
            if d[k][j] == 2: 
                d[k][j] = (d[k][j] + 1)*(N3/N2)
            if d[k][j] == 3: 
                d[k][j] = (d[k][j] + 1)*(N4/N3)
            if d[k][j] == 4: 
                d[k][j] = (d[k][j] + 1)*(N5/N4)

            acc+= d[k][j] #total number og bigrams counted
    print(acc)
    for k in d:
        for l in d[k]:
             #print(d[k]['<unk>'])
             d[k][l] = d[k][l]/acc
             #print(d[l]['<unk>'])
        #print (d[k]['<unk>'])

    return d, word_count

def perplexity1(word_count,d):
    perplex = 0
    for k in d:
        perplex += -math.log(d[k])
    print (word_count)
    perplex = math.exp(perplex/word_count)

    return perplex

def perplexity2(word_count,d):
    perplex = 0
    for k in d:
        for j in d[k]:
            perplex += -math.log(d[k][j])
    print (word_count)

    perplex = math.exp(perplex/word_count)

    return perplex


truth_test1, word_count1 = make_Unigram(filepath)
truth_test2, word_count2 = make_Bigram(filepath)
p1 = perplexity1(word_count1,truth_test1)
p2 = perplexity2(word_count2, truth_test2)
print (p1,p2)
