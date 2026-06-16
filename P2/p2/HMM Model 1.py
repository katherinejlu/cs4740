import collections
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

TRAIN_PATH = 'data_release/train.csv'
VAL_PATH   = 'data_release/val.csv'
TEST_PATH  = 'data_release/test_no_label.csv'
OUTPUT_PATH = 'test.csv'

# Lambda weight on literal->literal transition during Viterbi decoding.
# Values < 1 penalize staying literal, encouraging more metaphor predictions.
LAMBDA_LL = 0.3


def load_data(path, has_labels=True):
    """Load CSV data into sentence, POS, and (optionally) label columns."""
    cols = ['sentence', 'pos_seq', 'label_seq'] if has_labels else ['sentence', 'pos_seq']
    return pd.read_csv(path, encoding='latin1', names=cols, skiprows=1)


def build_hmm(sentences, pos_tags, labels):
    """
    Estimate HMM emission and transition probabilities from training data.

    Emission probabilities: per-word metaphor/literal frequency (lexical probabilities).
    Transition probabilities: bigram label sequences (literal->literal, etc.)
    Smoothing: small additive smoothing for zero-count emissions.

    Returns a nested dict d where:
      d[word]['metLP']  = P(metaphor | word)
      d[word]['litLP']  = P(literal  | word)
      d['<START.>']['pMet'] = P(metaphor at start of sentence)
      d['<LITERAL>']['pLitToMet'] = P(metaphor | previous literal)
      etc.
    """
    d = collections.defaultdict(lambda: collections.defaultdict(int))
    smoothing = 0.1

    for i in range(len(sentences)):
        p_list = pos_tags[i].strip('][').split(', ')
        line = sentences[i].split()
        label_list = labels[i].strip('][').split(', ')

        for j, word in enumerate(line):
            label = label_list[j]
            d[word]['<count>'] += 1

            if label == '0':
                d['<LITERAL>']['<count>'] += 1
                d[word]['litLP'] += 1
            else:
                d['<METAPHOR>']['<count>'] += 1
                d[word]['metLP'] += 1

            if j == 0:
                d['<START.>']['<count>'] += 1
                if label == '1':
                    d['<START.>']['pMet'] += 1
            elif label_list[j-1] == '0' and label == '1':
                d['<LITERAL>']['pLitToMet'] += 1
            elif label_list[j-1] == '1' and label == '1':
                d['<METAPHOR>']['pMetToMet'] += 1

    # Smooth zero-count emissions
    for word in d:
        if d[word]['metLP'] == 0:
            d[word]['metLP'] += d[word]['<count>'] * smoothing
            d[word]['<count>'] += d[word]['<count>'] * smoothing
        if d[word]['litLP'] == 0:
            d[word]['litLP'] += 0.1
            d[word]['<count>'] += 0.1

    # Normalize emission probabilities
    for word in d:
        d[word]['metLP'] = d[word]['metLP'] / d[word]['<count>']
        d[word]['litLP'] = 1 - d[word]['metLP']

    # Normalize transition probabilities
    d['<START.>']['pMet']          = d['<START.>']['pMet'] / d['<START.>']['<count>']
    d['<START.>']['pLit']          = 1 - d['<START.>']['pMet']
    d['<LITERAL>']['pLitToMet']    = d['<LITERAL>']['pLitToMet'] / d['<LITERAL>']['<count>']
    d['<LITERAL>']['pLitToLit']    = 1 - d['<LITERAL>']['pLitToMet']
    d['<METAPHOR>']['pMetToMet']   = d['<METAPHOR>']['pMetToMet'] / d['<METAPHOR>']['<count>']
    d['<METAPHOR>']['pMetToLit']   = 1 - d['<METAPHOR>']['pMetToMet']

    return d


def viterbi(sentence, d, lambda_ll=LAMBDA_LL):
    """
    Viterbi decoding for binary metaphor sequence labeling.

    Maintains two score/path sequences:
      score1/path1: best path ending in metaphor (label=1)
      score0/path0: best path ending in literal  (label=0)

    At each step, scores are computed in log-space and converted back
    to avoid underflow. Lambda weights the literal->literal transition
    to tune the balance between the two classes.

    Returns the most likely label sequence as a list of '0'/'1' strings.
    """
    words = sentence.split()
    score1, score0 = [], []
    path1, path0  = [], []

    for j, word in enumerate(words):
        if word not in d:
            # Unknown word: carry forward previous score
            if j == 0:
                score1.append(1)
                score0.append(1)
            else:
                score1.append(score1[j-1])
                score0.append(score0[j-1])
                path1.append('0')
                path0.append('0')

        elif j == 0:
            # Initial probabilities
            score1.append(math.exp(math.log(d['<START.>']['pMet']) + math.log(d[word]['metLP'])))
            score0.append(math.exp(math.log(d['<START.>']['pLit']) + math.log(d[word]['litLP'])))

        else:
            # Literal state: best predecessor for label=0
            lit_from_lit = lambda_ll * math.exp(
                math.log(d['<LITERAL>']['pLitToLit']) +
                math.log(score0[j-1]) +
                math.log(d[word]['litLP'])
            )
            lit_from_met = math.exp(
                math.log(d['<METAPHOR>']['pMetToLit']) +
                math.log(score1[j-1]) +
                math.log(d[word]['litLP'])
