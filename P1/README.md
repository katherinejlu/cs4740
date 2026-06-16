# Project 1 — Opinion Spam Detection with Language Models & Naive Bayes
**CS 4740: Natural Language Processing, Cornell University, Fall 2019**

Classified hotel reviews as truthful or deceptive using n-gram language models and a Naive Bayes classifier, on the Deceptive Opinion Spam Corpus.

## Task
Binary classification: given a hotel review, predict whether it is truthful (0) or deceptive (1). Two approaches are compared — a language model based classifier using perplexity, and a feature-based Naive Bayes classifier.

## Approach

### Language Model Classifier (LM.py)
Separate unigram and bigram language models are trained on the truthful and deceptive training corpora. At test time, a review is classified by comparing its perplexity under each model — the class whose model assigns lower perplexity wins.

- **Unigram model** with Laplace smoothing and unknown word handling via `<unk>` token
- **Bigram model** with add-1 smoothing; unknown bigrams backed off to `<unk>` probabilities
- Perplexity computed per-review using log-space arithmetic to avoid underflow

### Good-Turing Smoothing Experiment (good_turing.py)
An experimental implementation of Good-Turing smoothing for the bigram model, redistributing probability mass from seen to unseen bigrams based on frequency-of-frequency counts (N1, N2, ... N5). Compared against Laplace smoothing on the validation set.

### Naive Bayes Classifier (NB.py)
Multinomial Naive Bayes trained on bag-of-words and bigram features using scikit-learn, with two additional engineered features:
- **Review length** (total word count)
- **Unique word count** (vocabulary richness)

Both features were motivated by the intuition that deceptive reviews may differ systematically in length and lexical diversity from truthful ones.

## Files
- `LM.py` — final language model classifier (unigram + bigram, Laplace smoothing)
- `good_turing.py` — experimental Good-Turing smoothed bigram model
- `project1.py` — early prototype of the language model
- `project_1_orig.py` — original draft with experimental smoothing variants
- `NB.py` — Naive Bayes classifier with engineered features
- `DATASET/` — truthful and deceptive train/validation/test corpora (not included)

## How to Run

```bash
# Language model classifier:
python LM.py

# Naive Bayes classifier:
python NB.py
```

Requires: scikit-learn, numpy

## Key Design Decisions
- **Separate language models per class** rather than a single model, so perplexity comparison is meaningful across classes
- **Per-review perplexity** computed independently for each review rather than corpus-level, enabling fine-grained classification
- **Log-space arithmetic** throughout to avoid floating point underflow on long reviews
- **UNK threshold** for unigram model: words not seen in training are smoothed via a fractional count assigned to `<unk>` at training time
- **Bigram reset at sentence boundaries** — `<LASTWORDINITIALIZE>` sentinel ensures bigrams don't span across reviews

## Technologies
Python · scikit-learn · NumPy
