# CS 4740 — Natural Language Processing
**Cornell University, Fall 2019**

A collection of four NLP projects spanning classical and neural approaches to language modeling, sequence labeling, and text classification.

## Projects

### Project 1 — Opinion Spam Detection with Language Models & Naive Bayes
Classified hotel reviews as truthful or deceptive using unigram and bigram language models trained separately on each class. Classification is based on perplexity comparison — a review is assigned to the class whose language model assigns it lower perplexity. Implemented Laplace and Good-Turing smoothing and unknown word handling. Also implemented a Multinomial Naive Bayes classifier with bag-of-words and bigram features using scikit-learn as a comparative baseline.

**Key concepts:** n-gram language models, perplexity, Laplace smoothing, Good-Turing smoothing, Naive Bayes, opinion spam detection

### Project 2 — Metaphor Detection with Sequence Labeling
Built two sequence labeling models to detect metaphorical word usage at the token level. Model 1 is a Hidden Markov Model with Viterbi decoding. Model 2 is a hybrid combining a Multinomial Naive Bayes classifier (using POS tags and word-level features) with HMM transition probabilities, decoded via Viterbi. Evaluated using precision, recall, and F1.

**Key concepts:** HMMs, Viterbi decoding, Naive Bayes, sequence labeling, metaphor detection, POS features

### Project 3 — Sentiment Analysis with Neural Networks
Implemented and compared a Feedforward Neural Network (FFNN) and a Recurrent Neural Network (RNN) for 5-class sentiment classification on Yelp reviews. Part 1 involved debugging a buggy FFNN implementation. Part 2 involved implementing an RNN from scratch in PyTorch using word embeddings and minibatch SGD. Models were compared across hidden dimensionality configurations and analyzed qualitatively and quantitatively.

**Key concepts:** FFNN, RNN, PyTorch, word embeddings, sentiment analysis, minibatch training, early stopping

### Project 4 — BERT Fine-tuning for Multiple Choice Reasoning
Fine-tuned a BERT transformer model for multiple choice question answering on the SWAG, RACE, and ARC datasets. Also extended Project 3's FFNN/RNN comparison with additional ablation studies and a hybrid model incorporating GloVe embeddings.

**Key concepts:** BERT, transformers, HuggingFace, fine-tuning, multiple choice QA, GloVe embeddings

## Technologies
Python · PyTorch · scikit-learn · HuggingFace Transformers · NumPy · pandas
