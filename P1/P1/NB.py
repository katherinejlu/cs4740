import collections
import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

TRAIN_T = "DATASET/train/truthful.txt"
TRAIN_D = "DATASET/train/deceptive.txt"
TEST    = "DATASET/test/test.txt"
VAL_T   = "DATASET/validation/truthful.txt"
VAL_D   = "DATASET/validation/deceptive.txt"


def add_features(matrix):
    """
    Appends two engineered features to a document-term matrix:
    - Total word count (review length)
    - Unique word count (lexical diversity)
    Both features are motivated by the observation that deceptive reviews
    may differ systematically in length and vocabulary richness.
    """
    length = np.sum(matrix, axis=1, keepdims=True)
    unique = np.count_nonzero(matrix, axis=1, keepdims=True)
    return np.concatenate((matrix, length, unique), axis=1)


def load_corpus(truthful_path, deceptive_path):
    """
    Loads truthful and deceptive training corpora.
    Returns the combined file handle and binary labels
    (deceptive=1, truthful=0).
    """
    with open(truthful_path) as t, open(deceptive_path) as d:
        truthful_lines = t.readlines()
        deceptive_lines = d.readlines()

    lines = deceptive_lines + truthful_lines
    labels = np.array([1] * len(deceptive_lines) + [0] * len(truthful_lines))
    return lines, labels


# --- Load and vectorize training data ---
train_lines, y = load_corpus(TRAIN_T, TRAIN_D)
vec = CountVectorizer(ngram_range=(1, 2))
X_train = add_features(vec.fit_transform(train_lines).toarray())

print(f"Training matrix shape: {X_train.shape}")

# --- Train classifier ---
model = MultinomialNB()
model.fit(X_train, y)

# --- Validation ---
def evaluate(filepath, label_name):
    with open(filepath) as f:
        X = add_features(vec.transform(f).toarray())
    preds = model.predict(X)
    # deceptive=1, truthful=0; correct predictions are 1s for deceptive, 0s for truthful
    if label_name == 'deceptive':
        acc = np.sum(preds) / len(preds)
    else:
        acc = 1 - np.sum(preds) / len(preds)
    print(f"{label_name} accuracy: {acc:.3f}")
    return preds

preds_d = evaluate(VAL_D, 'deceptive')
preds_t = evaluate(VAL_T, 'truthful')
overall = (np.sum(preds_d) + (len(preds_t) - np.sum(preds_t))) / (len(preds_d) + len(preds_t))
print(f"Overall validation accuracy: {overall:.3f}")

# --- Test predictions ---
with open(TEST) as f:
    X_test = add_features(vec.transform(f).toarray())

preds_test = model.predict(X_test)

with open('testNB.csv', 'w') as f:
    f.write('Id,Prediction\n')
    for i, pred in enumerate(preds_test):
        f.write(f"{i},{pred}\n")model = MultinomialNB().fit(vector,y)

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
