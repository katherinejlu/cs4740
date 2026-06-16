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
import numpy

train_Tpath = "DATASET/train/truthful.txt"
train_Dpath = "DATASET/train/deceptive.txt"

test_path = "DATASET/test/test.txt"

val_Tpath = "DATASET/validation/truthful.txt"
val_Dpath = "DATASET/validation/deceptive.txt"

## capitalization? punctuation? contractions?

def make_Unigram(filepath):
    word_count = 0
    smoothing = 0.5
    count = 0
    d = collections.defaultdict(int)
    d['<unk>'] = 0
    with open(filepath) as f:
        for line in f:
            for word in line.split():
                word_count += 1
                d[word]+=1
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
        d[k] = (d[k]+0.6)/(word_count+0.6*len(d))#corpus length includes punctation?
        #count+= d[k]
    print (d['<unk>'])
    #print (count)\
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
                #word = word.lower()
                word_count +=1
                if last_word == '<LASTWORDINITIALIZE>' :
                    last_word = word
                    continue
                
                d[last_word][word]+=1
                last_word = word

    NUM_BIGRAMS = sum(len(bigrams) for bigrams in d.values())
    print('this is length d')
    print(len(d))
    print('this is number of bigrams')
    print(NUM_BIGRAMS)
    
    d['<unk>']['<unk>'] = NUM_BIGRAMS * smoothing

  ### SHOULD SMOOTHING BE NUM_BIGRAMS = sum(len(bigrams) for bigrams in d.values())
    # d['<unk>']['<unk>'] = NUM_BIGRAMS*smoothing
    # len(d) is just number of unique tokens 
    unk = 0
    for k in d:
        d[k]['<unk>'] =  len(d[k]) * smoothing ##run code replacing num_bigrams w len(d[k])
                                                 ##you see it accurately predicts next word
                                                 ##at, and old code makes 'unk' = 0.9
        acc = 0 
        for i in d[k]:

            acc+= (d[k][i] + 1)
        for j in d[k]:
            
            d[k][j] = (d[k][j]+1)/acc
            if j == '<unk>':
                unk+=d[k][j]
            #print (k,j, d[k][j])
    print('unknown prob' )      
    print (unk)
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
                #word = word.lower()
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

lastminD, x = make_Unigram(train_Dpath)
lastminT, y = make_Unigram(train_Tpath)
print (x)
print (y)
LDP = perplexity(lastminD, val_Dpath, 1)
LTP = perplexity(lastminT, val_Tpath, 1)

print ("deceptive on deceptive, T on T UNIGRAM")
LDPA = sum(LDP)/len(LDP)
LTPA = sum(LTP)/len(LTP)
print(LDPA,LTPA)

print("deceptive on truth, T on D UNIGRAM")
LDP1 = perplexity(lastminD, val_Tpath, 1)
LTP1 = perplexity(lastminT, val_Dpath, 1)
LDPA1 = sum(LDP1)/len(LDP1)
LTPA1 = sum(LTP1)/len(LTP1)
print(LDPA1,LTPA1)


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
##print("D on T, T on D")

##print (numpy.var(Dp))
##print (numpy.var(Dp1))
##print (numpy.var(Tp))
##print (numpy.var(Tp1))

Dp2 = perplexity(decept_test, train_Tpath , 2)
Tp2 = perplexity(truth_test, train_Dpath , 2)

Dp_test = perplexity(decept_test, test_path , 2)
Tp_test = perplexity(truth_test, test_path , 2)
Dpt_avg = sum(Dp_test)/len(Dp_test)
Tpt_avg = sum(Tp_test)/len(Tp_test)


##
with open('test.csv', 'w') as f:
    f.write('Id,Prediction\n')
    x = 1
    for i in range(len(Dp_test)):
        x = 1
        if (Dp_test[i]< Tp_test[i]):
            x = 0
        else:
            x = 1
##        if (Tp_test[i] > 200) & (Dp_test[i] > 200) & (abs(Tp_test[i]-Dp_test[i]) <Tp_test[1]/3):
##            x = 1
        f.write("%d,%d\n" %(i,x))
              
##print(Dp_test, Tp_test)
##
##with open('DECvaltest.csv', 'w') as f:
##    f.write('Id,Prediction\n')
##    x = 1
##    for i in range(len(Tp)):
##        x = 1
####        if abs(Dp1[i] - Tp[i]) < 0.5*Dp1_avg :
####            x = 1
##
##        if (Dp[i]< Tp1[i]):
##            x = 0
##        else:
##            x = 1
####        if ((Tp1[i] > Dp_avg) | (Dp[i] > Dp_avg)) & (abs(Tp1[i]-Dp[i]) <Tp1[1]/3):
####            x = 1
####        if (abs(Dp1[i] - Tp[i]) > 0.5*Dp1_avg):
####            x = 1
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

