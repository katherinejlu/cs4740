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
    smoothing = 1
    acc = 0;
    last_word = '<LASTWORDINITIALIZE>'
    word_count = 0
    d = collections.defaultdict(lambda:collections.defaultdict(int))
    with open(filepath) as f:
        for line in f:
            last_word = '<LASTWORDINITIALIZE>'
            for word in line.split():
                word = word.lower()
                word_count +=1
                if last_word == '<LASTWORDINITIALIZE>' :
                    last_word = word
                    continue
                
                d[last_word][word]+=1
                last_word = word
    NUM_BIGRAMS = sum(len(bigrams) for bigrams in d.values())
    d['<unk>']['<unk>'] = NUM_BIGRAMS * smoothing


  ### SHOULD SMOOTHING BE NUM_BIGRAMS = sum(len(bigrams) for bigrams in d.values())
    # d['<unk>']['<unk>'] = NUM_BIGRAMS*smoothing
    # len(d) is just number of unique tokens 
    
    for k in d:
        d[k]['<unk>'] = NUM_BIGRAMS * smoothing
        #print (d[k]['s<unk>'])
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
print (Dp, Tp)
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
print(Dp1, Tp1)


Dp2 = perplexity(decept_test, train_Tpath , 2)
Tp2 = perplexity(truth_test, train_Dpath , 2)

Dp_test = perplexity(decept_test, test_path , 2)
Tp_test = perplexity(truth_test, test_path , 2)
Dpt_avg = sum(Dp_test)/len(Dp_test)
Tpt_avg = sum(Tp_test)/len(Tp_test)

with open('test.csv', 'w') as f:
    f.write('Id,Prediction\n')
    x = 0
    for i in range(len(Dp_test)):
        if abs(Dp_test[i] - Tp_test[i])  < 0.5*Dpt_avg:
            x = 1
        if abs(Dp_test[i] - Tp_test[i]) < abs(Dpt_avg - Tpt_avg):
            x = 0
        f.write("%d,%d, %d, %d, %d, %d\n" %(i,x, Dp_test[i], Tp_test[i], Dpt_avg, Tpt_avg))
##              
##print(Dp_test, Tp_test)

##with open('valtest.csv', 'w') as f:
##    f.write('Id,Prediction\n')
##    x = 0
##    for i in range(len(Tp)):
##
##        if abs(Dp[i] - Tp1[i]) < 0.5*Dp_avg :
##            x = 1
##
##        if abs(Dp[i] - Tp1[i]) < 1.3*abs(Dp_avg - Tp1_avg):
##            x = 0
##        f.write("%d,%d,%d,%d\n"%(i, x, Dp[i],Tp1[i]))

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

