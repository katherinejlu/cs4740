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
    smoothing = .3
    acc = 0;
    last_word = 'grgsgtrgtsyhtsujts'
    word_count = 0
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
        for i in d[k]:
            acc+= d[k][i]
        for j in d[k]:
            d[k][j] = d[k][j]/acc
    #print (d)
    return d, word_count

def perplexity1(word_count1,d):
    perplex = 0
    for k in d:
        perplex += -math.log(d[k])*d[k]*word_count1
    print (word_count1)
    print(perplex)
    perplex = math.exp(perplex/word_count1)

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
