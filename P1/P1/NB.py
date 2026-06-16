from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
import numpy as np
import math

train_path = "train.txt"
test_path = "test/test.txt"
val_Tpath = "validation/truthful.txt"
val_Dpath = "validation/deceptive.txt"
tval = open(val_Tpath)
dval = open (val_Dpath)
test = open (test_path)
corpus = open(train_path)

t = np.zeros(512)
d = np.ones(512)
y = np.concatenate((d,t),axis = 0)

## capitalization? punctuation? contractions?

vec = CountVectorizer(ngram_range =(1,2))
vector = vec.fit_transform(corpus).toarray()

print (vector.shape[0],vector.shape[1])

'''add len of reviews feature '''
new = np.sum(vector,axis = 1)
new = np.expand_dims(new, axis=0)
vector = np.concatenate((vector,new.T),axis = 1)

''' add unique words feature '''
uniq = np.count_nonzero(vector,axis = 1)
uniq = np.expand_dims(uniq, axis=0)
vector = np.concatenate((vector,uniq.T),axis = 1)

print (vector.shape[0],vector.shape[1])
model = MultinomialNB().fit(vector,y)

## truthful val
t_val_ = vec.transform(tval).toarray()
new = np.sum(t_val_,axis = 1)
new = np.expand_dims(new, axis=0)
t_val_ = np.concatenate((t_val_,new.T),axis = 1)

new = np.count_nonzero(t_val_,axis = 1)
new = np.expand_dims(new, axis=0)
t_val_ = np.concatenate((t_val_,new.T),axis = 1)

## deceptive val
d_val_ = vec.transform(dval).toarray()
new = np.sum(d_val_,axis = 1)
new = np.expand_dims(new, axis=0)
d_val_ = np.concatenate((d_val_,new.T),axis = 1)

new = np.count_nonzero(d_val_,axis = 1)
new = np.expand_dims(new, axis=0)
d_val_ = np.concatenate((d_val_,new.T),axis = 1)

pre_t = model.predict(t_val_)
pre_d = model.predict(d_val_)
# print (pre_d)
print ('deceptive',np.sum(pre_d), np.sum(pre_d)/len(pre_d))
print ('truthful',np.sum(pre_t), 1-np.sum(pre_t)/len(pre_t))
print ('overall', (np.sum(pre_d) + (len(pre_t)-np.sum(pre_t)))/(len(pre_d)+len(pre_t)))

"""test prediction"""

test_arr = vec.transform(test).toarray()

'''add len of reviews feature '''
new = np.sum(test_arr,axis = 1)
new = np.expand_dims(new, axis=0)
test_arr = np.concatenate((test_arr,new.T),axis = 1)
''' add unique words feature '''
new = np.count_nonzero(test_arr,axis = 1)
new = np.expand_dims(new, axis=0)
test_arr = np.concatenate((test_arr,new.T),axis = 1)

pre_test = model.predict(test_arr)


''' writr to csv file '''

with open('testNB.csv', 'w') as f:
   f.write('Id,Prediction\n')
   x = 0
   for i in range(len(pre_test)):

       f.write("%d,%d\n"%(i, pre_test[i]))

corpus.close()
tval.close()
dval.close()
test.close()
