# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:04:06 2019

@author: Katherine
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd 
import collections 
path = 'data_release/train.csv' 
import math

tags = pd.read_csv(path, encoding = 'latin1', names = ['sentence','pos_seq', 'label_seq'], skiprows = 1 )
labels = tags.label_seq
sentences = tags.sentence
pos = tags.pos_seq


vec = CountVectorizer(ngram_range = (1,1))
vec2 = CountVectorizer(ngram_range = (1,1))

wordlist = []
y = []
x = []
p = []
d = collections.defaultdict(lambda:collections.defaultdict(int))
#print(vector)
#print(sentences[0])
for i in range(len(sentences)): 
    p_list = pos[i].strip('][').split(', ') 
    line = sentences[i].split()
    label_list = labels[i].strip('][').split(', ') 
    #print(label_list)
    for j in range(len(sentences[i].split())): 
        #num_items += len(sentence(i))
        p.append(p_list[j])
        x.extend(line)
        y.append(int(label_list[j]))
        word = line[j].lower()

        label = label_list[j]
        #print(label)
        d[word]['<count>'] += 1
        if label == '0': 
            d['<LITERAL>']['<count>'] += 1
            d[word]['litLP'] += 1 
        else: 
            d['<METAPHOR>']['<count>'] += 1
            d[word]['metLP'] += 1

       # print(label[i+1][j+1])
        if j == 0: 
            d['<START.>']['<count>']+=1
            if label == '1':
                d['<START.>']['pMet'] += 1
                
    
        elif label_list[j-1] == '0' :
            if label == '1':     
                d['<LITERAL>']['pLitToMet'] += 1
        else: 
            if label == '1': 
                d['<METAPHOR>']['pMetToMet'] += 1
#print(d['<START.>']['pMet'])   
#small amount of smoothing
for i in range(len(sentences)): 
    line = sentences[i].split()
    for j in range(len(sentences[i].split())): 
        word = line[j].lower()
        if d[word]['<count>'] <= 3: 
            wordlist.append('unk')
        else: 
            wordlist.append(word)
                
wordvector = vec.fit_transform(wordlist).toarray()               
vector = vec2.fit_transform(p).toarray()   

#wordvector = np.expand_dims(wordvector, axis = 1)
vector = np.concatenate((vector,wordvector), axis=1)                
model = MultinomialNB().fit(vector,y)

#print (type(y[2]))
smoothing = 2
for x in d: 
    if d[x]['metLP'] == 0: 
        d[x]['metLP'] += 1
        d[x]['<count>'] += 1
#    if d[x]['metLP'] > 0 and d[x]['metLP'] < 0.1*d[x]['<count>']: 
#        d[x]['metLP'] += d[x]['<count>']*smoothing
#        d[x]['<count>'] += d[x]['<count>']*smoothing
    if d[x]['litLP'] == 0: 
        d[x]['litLP'] += 1
        d[x]['<count>'] += 1
for x in d: 
    #lexical probabilities of all words
    #print(d[x])
    d[x]['metLP'] = d[x]['metLP']/d[x]['<count>'] 
    d[x]['litLP'] = 1 - d[x]['metLP']
    
    #transitional probabilities 
d['<START.>']['pMet'] = d['<START.>']['pMet']/d['<START.>']['<count>']
d['<START.>']['pLit'] = 1 - d['<START.>']['pMet']
d['<LITERAL>']['pLitToMet'] = d['<LITERAL>']['pLitToMet']/d['<LITERAL>']['<count>']
d['<LITERAL>']['pLitToLit'] = 1 - d['<LITERAL>']['pLitToMet']
d['<METAPHOR>']['pMetToMet'] = d['<METAPHOR>']['pMetToMet'] / d['<METAPHOR>']['<count>']
d['<METAPHOR>']['pMetToLit'] = 1 - d['<METAPHOR>']['pMetToMet']
#
#print(d['<LITERAL>']['pLitToMet'])
#print(d['<LITERAL>']['pLitToLit'])
#print(d['<METAPHOR>']['pMetToMet'])
#print(d['<METAPHOR>']['pMetToLit'])
#print(d['<START.>']['<count>'])
#print(d['<START.>']['pMet'])
#print(len(sentences)-1)
#print(d['<START.>']['pMetaphor'])
#print(d['<START.>']['pLit'])

#viterbi 

test_path = 'data_release/test_no_label.csv'
ttags = pd.read_csv(test_path, encoding = 'latin1', names = ['sentence','pos_seq'], skiprows = 1)

test_data = []

tsentences = ttags.sentence
tlabels = ttags.pos_seq
val_path = 'data_release/val.csv'
vtags = pd.read_csv(val_path, encoding = 'latin1', names = ['sentence','pos_seq', 'label_seq'], skiprows = 1)
tpos = ttags.pos_seq
vlabels = vtags.label_seq
vsentences = vtags.sentence
lambdaLL = 50
test_words = []
mylabel = []

for i in range(len(tsentences)):
    p_list = tpos[i].strip('][').split(', ') 
    sent = ttags.sentence[i].split()
    
    for j in range(len(tsentences[i].split())): 
        test_data.append(p_list[j])
        word = sent[j].lower()
        if word in wordlist: 
            test_words.append(word)
        else: 
            test_words.append('unk')
            
wordtestvec = vec.transform(test_words).toarray() 
testvector = vec2.transform(test_data).toarray() 
testvector = np.concatenate((testvector,wordtestvec), axis=1)   
NBlabels = model.predict_proba(testvector)
val_test = [] 
index = 0; 
lambda00 = 100
lambda01 = 0.001
lambda11 = 0.01
lambda10 = 0.001
#print(vsentences[0].split())
#print(len(vsentences[0].split()))
for i in range(len(tsentences)): 
    tlabel_list = tlabels[i].strip('][').split(', ')
    line = tsentences[i].split()
    path1 = []
    path0 = []
    choose= []
    score1 = []
    score0 = []
    for j in range(len(tsentences[i].split())): 
        #num_items += len(sentence(i))
        
        word = line[j].lower()
        val_test.append(tlabel_list[j])
        if word not in d: 
#            print('got there')

            if j == 0: 
                score1.append(1)
                score0.append(1)
            else:
                score1.append(score1[j-1])
                score0.append(score0[j-1])
                path1.append('0')
                path0.append('0')
            
        elif j == 0:
            score1.append(math.exp(math.log(d['<START.>']['pMet'])+math.log(d[word]['metLP'])))
#            path1.append('')
            score0.append(math.exp(math.log(d['<START.>']['pLit'])+math.log(d[word]['litLP'])))
#            path2.append('start')

        else: 
                if math.exp(lambda00*math.log(d['<LITERAL>']['pLitToLit'])+math.log(score0[j-1])+math.log(NBlabels[index][0])) >= math.exp(lambda10*math.log(d['<METAPHOR>']['pMetToLit'])+math.log(score1[j-1])+math.log(NBlabels[index][0])): 
                    score0.append(math.exp(lambda00*math.log(d['<LITERAL>']['pLitToLit'])+math.log(score0[j-1])+math.log(NBlabels[index][0])))
                    path0.append('0')
                else: 
                    score0.append(math.exp(lambda10*math.log(d['<METAPHOR>']['pMetToLit'])+math.log(score1[j-1])+math.log(NBlabels[index][0])))
                    path0.append('1')
                if math.exp(lambda11*math.log(d['<METAPHOR>']['pMetToMet'])+math.log(score1[j-1])+math.log(NBlabels[index][1])) >= math.exp(lambda01*math.log(d['<LITERAL>']['pLitToMet'])+math.log(score0[j-1])+math.log(NBlabels[index][1])):
                    score1.append(math.exp(lambda11*math.log(d['<METAPHOR>']['pMetToMet'])+math.log(score1[j-1])+math.log(NBlabels[index][1])))
                    path1.append('1')
                else: 
                    score1.append(math.exp(lambda01*math.log(d['<LITERAL>']['pLitToMet'])+math.log(score0[j-1])+math.log(NBlabels[index][1])))
                    path1.append('0')
        index += 1

    path1.append('1')
    path0.append('0')
#
#    if len(path0) != len(vsentences[i].split()):
#        if i ==4 :
#            print(len(path0))
#            print(path0)


#    print('score0')
#    print (score0[(len(score0))-1])

    if score1[(len(score1))-1] > score0[(len(score0))-1]:
        mylabel.extend(path1)
    else: 
        mylabel.extend(path0)
print(NBlabels)
count = 0
onecount = 0 
precision = 0
#print(mylabel)
#print(len(val_test))
#for x in range(len(mylabel)):
#    if mylabel[x] == '1': 
#        precision += 1
#    if val_test[x] == '1': 
#        onecount += 1
#    if mylabel[x] == '1' and val_test[x] == '1':
#        count+=1
#print(count)
#print(count/onecount)
#print(count/precision)
#print((2*(count/precision) * (count/onecount)) / ((count/onecount) + (count/precision)))
#print(score1[1:len(score1)-1])
#print(score0[1:(len(score0))-1])
#test_data = np.expand_dims(test_data, axis=0)

#print(NBlabels)
#print(np.shape(NBlabels))

testlabel = 0 
with open('test.csv', 'w') as f:
    f.write('idx,label\n')
    for i in range(1,50176): 
        
        f.write("%d,%d\n" %(i,int(mylabel[i-1])))
    