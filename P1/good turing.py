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

train_Tpath = "DATASET/train/truthful.txt"
train_Dpath = "DATASET/train/deceptive.txt"

test_path = "DATASET/test/test.txt"

val_Tpath = "DATASET/validation/truthful.txt"
val_Dpath = "DATASET/validation/deceptive.txt"

## capitalization? punctuation? contractions?

def make_Unigram(filepath):
    word_count = 0
    smoothing = 0.01
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
    smoothing = 1 #0.003
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
                word = word.lower()
                if last_word == 'grgsgtrgtsyhtsujts' :
                    last_word = word
                    continue
                word_count +=1
                d[last_word][word]+=1
                last_word = word
    NUM_BIGRAMS = sum(len(bigrams) for bigrams in d.values())
    d['<unk>']['<unk>'] = NUM_BIGRAMS * smoothing
    for k in d:
        d[k]['<unk>'] = len(d[k]) * smoothing
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
        acc = 0
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
   # print(acc)
    for k in d:
        for l in d[k]:
             #print(d[k]['<unk>'])
             d[k][l] = d[k][l]/acc
             #print(d[l]['<unk>'])
        #print (d[k]['<unk>'])
    print(d['stayed'].items())
    return d, word_count

def perplexity(d, filepath, ngram):
    perplex = 0
    plist = []
    word_count1 = 0
    last_word = '<LASTWORDINITIALIZE>'
    with open(filepath) as f:
        for line in f:
            perplex = 0
            word_count1 = 0
            last_word = '<LASTWORDINITIALIZE>'
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
            perplex = math.exp(perplex/word_count1)
            plist.append(perplex)   
    ## look at keys from the testing 
##    for k in d:
##        perplex += -math.log(d[k])*d[k]*word_count1
                ## this is wrong bc its not total tokens, it has
                ## it has smoothing
                

    
##    print(perplex) 

    return plist 
    
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


decept_test, word_count1 = make_Bigram(train_Dpath)
truth_test, word_count2 = make_Bigram(train_Tpath)
Dp = perplexity(decept_test, val_Dpath , 2)
Tp = perplexity(truth_test, val_Tpath, 2)


print("deceptive on deceptive, T on T") 
##print (Dp, Tp)
Dp_avg = sum(Dp)/len(Dp)
Tp_avg = sum(Tp)/len(Tp)
print("avg d on d, agv t on t")
print(Dp_avg, Tp_avg)

Dp1 = perplexity(decept_test, val_Tpath , 2)
Tp1 = perplexity(truth_test, val_Dpath, 2)

Dp1_avg = sum(Dp1)/len(Dp1)
Tp1_avg = sum(Tp1)/len(Tp1)
print("avg d on t, agv t on d")
print(Dp1_avg, Tp1_avg)
print("D on T, T on D")
##print(Dp1, Tp1)


Dp2 = perplexity(decept_test, train_Tpath , 2)
Tp2 = perplexity(truth_test, train_Dpath , 2)

Dp_test = perplexity(decept_test, test_path , 2)
Tp_test = perplexity(truth_test, test_path , 2)


##with open('test.csv', 'w') as f:
##    f.write('Id,Prediction\n')
##    for i in range(len(Dp_test)):
##        if Dp_test[i] < 100:
##            f.write("%d,0\n" %i)
##        else:
##            f.write("%d,1\n" %i)
##print(Dp_test, Tp_test)

with open('valtest.csv', 'w') as f:
    f.write('Id,Prediction\n')
    for i in range(len(Tp)):
        if Dp[i] < Tp1[i]:
            f.write("%d,0\n" %i)
        else:
            f.write("%d,1\n" %i)

##print("testD on TestT, and reverse")
##print(Dp2, Tp2)
##
##testP = perplexity(decept_test, train_Dpath, 2)
##print(testP)
##
##testP2 = perplexity(truth_test, train_Tpath, 2)
##print(testP2)


##with open(test_path) as f:
##        for line in f:
##            last_word = '<LASTWORDINITIALIZE>'
##            for word in line.split():





                
    
