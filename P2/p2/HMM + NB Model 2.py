import collections
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

TRAIN_PATH  = 'data_release/train.csv'
VAL_PATH    = 'data_release/val.csv'
TEST_PATH   = 'data_release/test_no_label.csv'
OUTPUT_PATH = 'test.csv'

# UNK threshold: words appearing <= this many times are collapsed to 'unk'
UNK_THRESHOLD = 3

# Lambda weights on transition probabilities during Viterbi decoding.
# These were tuned on the validation set to balance precision and recall.
# lambda00: weight on literal->literal transition
# lambda10: weight on metaphor->literal transition
# lambda11: weight on metaphor->metaphor transition
# lambda01: weight on literal->metaphor transition
LAMBDA00 = 100
LAMBDA10 = 0.001
LAMBDA11 = 0.01
LAMBDA01 = 0.001


def load_data(path, has_labels=True):
    """Load CSV data into sentence, POS, and (optionally) label columns."""
    cols = ['sentence', 'pos_seq', 'label_seq'] if has_labels else ['sentence', 'pos_seq']
    return pd.read_csv(path, encoding='latin1', names=cols, skiprows=1)


def build_hmm_and_nb(sentences, pos_tags, labels):
    """
    Build HMM transition/emission probabilities and train a Naive Bayes classifier.

    The NB classifier uses two feature sets concatenated:
      1. POS tag unigrams (vec2)
      2. Word unigrams with UNK thresholding (vec) — words appearing <= UNK_THRESHOLD
         times are replaced with 'unk' to reduce sparsity

    The HMM provides transition probabilities between literal/metaphor states.
    At inference time, NB emission probabilities replace the lexical HMM emissions,
    and Viterbi decoding combines both with tunable lambda weights.

    Returns: d (HMM prob dict), model (NB classifier), vec (word vectorizer),
             vec2 (POS vectorizer), wordlist (UNK-thresholded word list)
    """
    d = collections.defaultdict(lambda: collections.defaultdict(int))
    p, y = [], []
    raw_words = []

    for i in range(len(sentences)):
        p_list = pos_tags[i].strip('][').split(', ')
        line = sentences[i].split()
        label_list = labels[i].strip('][').split(', ')

        for j, word in enumerate(line):
            word = word.lower()
            label = label_list[j]

            p.append(p_list[j])
            y.append(int(label))
            raw_words.append(word)

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

    # Build UNK-thresholded word list for NB features
    wordlist = [
        w if d[w]['<count>'] > UNK_THRESHOLD else 'unk'
        for w in raw_words
    ]

    # Fit vectorizers and train NB classifier
    vec  = CountVectorizer(ngram_range=(1, 1))
    vec2 = CountVectorizer(ngram_range=(1, 1))
    word_features = vec.fit_transform(wordlist).toarray()
    pos_features  = vec2.fit_transform(p).toarray()
    X = np.concatenate((pos_features, word_features), axis=1)
    model = MultinomialNB().fit(X, y)

    # Smooth zero-count emissions
    for word in d:
        if d[word]['metLP'] == 0:
            d[word]['metLP'] += 1
            d[word]['<count>'] += 1
        if d[word]['litLP'] == 0:
            d[word]['litLP'] += 1
            d[word]['<count>'] += 1

    # Normalize emission probabilities
    for word in d:
        d[word]['metLP'] = d[word]['metLP'] / d[word]['<count>']
        d[word]['litLP'] = 1 - d[word]['metLP']

    # Normalize transition probabilities
    d['<START.>']['pMet']        = d['<START.>']['pMet'] / d['<START.>']['<count>']
    d['<START.>']['pLit']        = 1 - d['<START.>']['pMet']
    d['<LITERAL>']['pLitToMet']  = d['<LITERAL>']['pLitToMet'] / d['<LITERAL>']['<count>']
    d['<LITERAL>']['pLitToLit']  = 1 - d['<LITERAL>']['pLitToMet']
    d['<METAPHOR>']['pMetToMet'] = d['<METAPHOR>']['pMetToMet'] / d['<METAPHOR>']['<count>']
    d['<METAPHOR>']['pMetToLit'] = 1 - d['<METAPHOR>']['pMetToMet']

    return d, model, vec, vec2, wordlist


def get_nb_probs(sentences, pos_tags, wordlist, vec, vec2):
    """
    Compute NB emission probabilities for each token in the test sentences.
    Returns an array of shape (num_tokens, 2) where columns are P(literal) and P(metaphor).
    """
    test_words, test_pos = [], []
    for i in range(len(sentences)):
        p_list = pos_tags[i].strip('][').split(', ')
        for j, word in enumerate(sentences[i].split()):
            word = word.lower()
            test_words.append(word if word in wordlist else 'unk')
            test_pos.append(p_list[j])

    word_features = vec.transform(test_words).toarray()
    pos_features  = vec2.transform(test_pos).toarray()
    X = np.concatenate((pos_features, word_features), axis=1)
    return model.predict_proba(X)


def viterbi_nb(sentence, d, nb_probs, start_index):
    """
    Viterbi decoding using NB classifier probabilities as emissions.

    Unlike the pure HMM, emission probabilities come from the NB classifier
    rather than lexical frequency counts. Transition probabilities are weighted
    by tunable lambda values to balance the influence of the HMM vs. the NB classifier.

    Returns the most likely label sequence as a list of '0'/'1' strings,
    and the updated token index.
    """
    words = sentence.split()
    score1, score0 = [], []
    path1, path0   = [], []
    index = start_index

    for j, word in enumerate(words):
        word = word.lower()

        if word not in d:
            if j == 0:
                score1.append(1)
                score0.append(1)
            else:
                score1.append(score1[j-1])
                score0.append(score0[j-1])
                path1.append('0')
                path0.append('0')

        elif j == 0:
            score1.append(math.exp(
                math.log(d['<START.>']['pMet']) + math.log(d[word]['metLP'])
            ))
            score0.append(math.exp(
                math.log(d['<START.>']['pLit']) + math.log(d[word]['litLP'])
            ))

        else:
            nb0 = math.log(nb_probs[index][0])  # P(literal)
            nb1 = math.log(nb_probs[index][1])  # P(metaphor)

            # Literal state: best predecessor
            lit_from_lit = math.exp(LAMBDA00 * math.log(d['<LITERAL>']['pLitToLit'])  + math.log(score0[j-1]) + nb0)
            lit_from_met = math.exp(LAMBDA10 * math.log(d['<METAPHOR>']['pMetToLit']) + math.log(score1[j-1]) + nb0)
            if lit_from_lit >= lit_from_met:
                score0.append(lit_from_lit)
                path0.append('0')
            else:
                score0.append(lit_from_met)
                path0.append('1')

            # Metaphor state: best predecessor
            met_from_met = math.exp(LAMBDA11 * math.log(d['<METAPHOR>']['pMetToMet']) + math.log(score1[j-1]) + nb1)
            met_from_lit = math.exp(LAMBDA01 * math.log(d['<LITERAL>']['pLitToMet'])  + math.log(score0[j-1]) + nb1)
            if met_from_met >= met_from_lit:
                score1.append(met_from_met)
                path1.append('1')
            else:
                score1.append(met_from_lit)
                path1.append('0')

        index += 1

    path1.append('1')
    path0.append('0')

    best_path = path1 if score1[-1] > score0[-1] else path0
    return best_path, index


def main():
    # Load data
    train = load_data(TRAIN_PATH, has_labels=True)
    test  = load_data(TEST_PATH,  has_labels=False)

    # Build HMM and NB classifier from training data
    d, model, vec, vec2, wordlist = build_hmm_and_nb(
        train.sentence, train.pos_seq, train.label_seq
    )

    # Get NB emission probabilities for all test tokens
    nb_probs = get_nb_probs(test.sentence, test.pos_seq, wordlist, vec, vec2)

    # Run Viterbi on each test sentence
    all_labels = []
    token_index = 0
    for sentence in test.sentence:
        labels, token_index = viterbi_nb(sentence, d, nb_probs, token_index)
        all_labels.extend(labels)

    # Write predictions
    with open(OUTPUT_PATH, 'w') as f:
        f.write('idx,label\n')
        for i, label in enumerate(all_labels, start=1):
            f.write(f"{i},{label}\n")


if __name__ == '__main__':
    main()
