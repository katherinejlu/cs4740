# Project 4 — BERT Fine-tuning for Multiple Choice Reasoning
**CS 4740: Natural Language Processing, Cornell University, Fall 2019**

Fine-tuned a BERT transformer model for multiple choice question answering, and extended Project 3's neural sentiment analysis with GloVe embeddings and additional ablation studies.

## Task
Multiple choice reading comprehension: given a passage and a question, select the correct answer from four options. Evaluated on the SWAG, RACE, and ARC datasets.

## Models

### Part 1 — FFNN + RNN Extended Analysis (GloVe)
Extended the Project 3 FFNN and RNN sentiment classifiers with GloVe pre-trained word embeddings (6B tokens, 100-dimensional) as an alternative to learned embeddings. Compared performance across hidden dimensionalities (32, 64, 128) for ablation study.

### Part 2 — BERT Fine-tuning
Fine-tuned `bert-base-uncased` for multiple choice using the HuggingFace Transformers library. Each answer choice is encoded as a (passage, question + choice) pair and passed through BERT independently. The pooled CLS token representation is used for classification across the four choices.

## Files
- `main.py` — entry point; runs RNN or FFNN with GloVe embeddings
- `p4.py` — RNN implementation extended with GloVe input
- `ffnn1fix.py` — corrected FFNN from Project 3
- `bert_redo.py` — BERT fine-tuning script
- `utils_multiple_choice.py` — HuggingFace data processing utilities for RACE, SWAG, and ARC datasets; handles tokenization, padding, and feature conversion for multiple choice format

## How to Run

**Part 1 (GloVe + RNN/FFNN):**
```bash
# Download GloVe: glove.6B.100d.txt must be in the working directory
# Toggle FLAG in main.py to 'RNN' or 'FFNN', then:
python main.py
```

**Part 2 (BERT):**
```bash
# Requires HuggingFace Transformers and a compatible GPU
# Data should be in the format expected by utils_multiple_choice.py
python bert_redo.py
```

Requires: PyTorch, HuggingFace Transformers, numpy, tqdm

## Key Design Decisions
- **CLS token** used as the sequence-level representation for classification, consistent with BERT's original fine-tuning protocol
- **Separate encoding per choice** rather than concatenating all choices, following the standard multiple choice fine-tuning approach
- **GloVe embeddings** kept frozen during FFNN/RNN training to isolate the effect of pre-trained representations vs. learned ones

## Technologies
Python · PyTorch · HuggingFace Transformers · BERT · GloVe · NumPy
