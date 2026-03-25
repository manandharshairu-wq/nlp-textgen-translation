# Neural Network Approaches for NLP: Text Generation and Machine Translation

## 1. Project Overview
This project implements two core Natural Language Processing (NLP) tasks using deep learning models. All experiments were implemented in PyTorch with a focus on reproducibility, modularity, and clear comparison.

Task 1: Text Generation – predicting the next word in a sequence
Task 2: Machine Translation – translating sentences from English to Spanish

The goal of this assignment is to [1] Compare different neural network architectures (LSTM vs GRU), [2] Evaluate different embedding methods (One-hot vs GloVe) and [3] Analyze how these choices impact model performance. 


## 2. Dataset Description
### Task 1: Text Generation Dataset
Dataset Used: Tiny Shakespeare Dataset
Source: Karpathy’s char-rnn repository
The dataset was tokenized, cleaned, and split into: 80% Training, 10% Validation and 10% Test set
Reason of choice: Tiny Shakespeare was chosen for its simplicity and suitability for sequence modeling.

### Task 2: Machine Translation Dataset
Dataset Used: OPUS Books (English → Spanish)
Source: HuggingFace datasets library
Each sample contains:
English sentence (source)
Spanish sentence (target)
Reason of choice: OPUS Books provides aligned sentence pairs ideal for basic translation experiments.

Since the dataset did not provide predefined splits for this setup, it was manually divided into:
Training= 2000, Validation= 2000 and Test set=2000. 



## 3. Model Architectures Used
### Task 1: Text Generation
Model: LSTM (Long Short-Term Memory)
Architecture: Sequence-to-one (predict next word)
Two variants:
LSTM + One-hot encoding
LSTM + GloVe embeddings 

### Task 2: Machine Translation
Model: GRU-based Seq2Seq (Encoder–Decoder)
Architecture: Encoder processes English sentence, Decoder generates Spanish translation
Training Strategy: Teacher forcing is used during training to improve convergence by feeding the correct target tokens to the decoder
Two variants:
GRU + One-hot encoding
GRU + GloVe-initialized embeddings (source side)


## 4. Word Embedding Methods

### One-hot Encoding
Each word is represented as a sparse vector where only one position is “1” and all others are “0”. This representation does not capture any semantic relationships between words and results in very high-dimensional vectors. It was chosen as a baseline method to evaluate how models perform without any prior semantic information, allowing for a clear comparison with more advanced embeddings.

### GloVe Embeddings
Pre-trained dense vectors (100-dimensional) that capture meaningful semantic relationships between words based on their co-occurrence in large text corpora. In this project, GloVe embeddings were chosen to assess how pre-trained semantic representations enhance model learning and generalization compared to one-hot encoding. 


## 5. Experimental Results
<img width="606" height="239" alt="Screenshot 2026-03-25 at 10 33 31 AM" src="https://github.com/user-attachments/assets/e4459663-40ca-49f3-a029-56efe3caf360" />



## 6. Comparison of Models

<img width="615" height="239" alt="Screenshot 2026-03-25 at 10 34 22 AM" src="https://github.com/user-attachments/assets/699faf39-e85a-4928-8dd7-7298d2f4e7b7" />

For Task 1 (Text Generation), the LSTM model with GloVe embeddings outperformed the LSTM model with one-hot encoding, achieving a lower test loss (5.6891 vs 5.7696) and lower perplexity (295.61 vs 320.41). This indicates that pretrained embeddings helped the model make more confident next-word predictions. The improvement can be attributed to the fact that GloVe embeddings encode semantic relationships between words, allowing the model to start with a more informative representation compared to sparse one-hot vectors. Although the numerical improvement is moderate, the trend is consistent and shows that semantic information improves language modeling performance.

The training curves further support this observation. In both models, training loss decreases steadily, indicating that the models successfully learn from the training data. However, validation loss begins to flatten and slightly increase after a few epochs, suggesting mild overfitting. This means that while the models continue to fit the training data, their ability to generalize to unseen data does not improve beyond a certain point. The generated outputs reflect this behavior, as they show partial sentence structure learning but still contain 'unk' tokens and repetitive phrasing, indicating limited vocabulary coverage and incomplete language understanding.

<img width="610" height="232" alt="Screenshot 2026-03-25 at 10 34 50 AM" src="https://github.com/user-attachments/assets/328c9e64-bc1c-482d-bcd3-6005405897b1" />


For Task 2 (Machine Translation), the GRU-based Seq2Seq model with GloVe embeddings again outperformed the one-hot version, achieving a lower test loss (5.6417 vs 5.7791) and a higher BLEU score (7.47 vs 5.89). This suggests that pretrained embeddings also improve translation performance by providing better semantic representations of the source language. Since translation requires understanding relationships between words and phrases, initializing embeddings with GloVe helps the encoder produce more meaningful hidden representations.

Despite this improvement, overall translation quality remains limited. The BLEU scores are relatively low, and the generated translations contain many 'unk' tokens and fragmented phrases. This indicates that the model struggles with vocabulary coverage and precise word alignment between source and target sentences. Additionally, the gap between decreasing training loss and relatively flat validation loss suggests overfitting, where the model learns training patterns but does not generalize effectively. This limitation is likely due to the relatively small dataset size and the absence of more advanced mechanisms such as attention.
Across both tasks, models using GloVe embeddings consistently outperformed those using one-hot encoding, demonstrating the advantage of incorporating semantic information into word representations. 

## 7. Challenges Faced During Implementation
### Handling Unknown Tokens (unk):
A limited vocabulary led to a high number of 'unk' tokens in generated outputs, particularly in the translation task. This reduced the overall quality and interpretability of results.

### Overfitting During Training:
In task 1 mainly, as training progressed, the training loss continued to decrease while validation loss began to increase after around epoch 4. This indicated that the models were overfitting and memorizing training data rather than generalizing. The optimal performance was observed around 3–5 epochs, after which test performance degraded.

### GloVe Embedding Integration:
Incorporating pre-trained GloVe embeddings required careful alignment between the embedding vectors and the model’s vocabulary indices. Mismatches or missing words had to be handled properly to avoid errors or performance issues.

### Lack of Predefined Data Splits (Translation Task):
The OPUS Books dataset did not provide predefined training, validation, and test splits. As a result, the data had to be manually partitioned using random shuffling, which required careful handling to maintain a fair evaluation setup.

### Trade-off Between Vocabulary Size and Training Efficiency:
Increasing vocabulary size and dataset complexity initially led to worse performance due to insufficient training time. This highlighted a trade-off between model capacity and the ability to effectively learn from larger datasets within limited epochs.

## 8. Limitations of the Considered Models

### High Perplexity in Text Generation
The LSTM models show high perplexity (295), indicating uncertainty in predicting the next word and weaker language fluency.

### Low BLEU Scores in Translation
The translation models achieve low BLEU scores (7.47 and 5.89), suggesting poor translation accuracy and limited similarity to reference outputs.

### Frequent 'unk' Tokens
Both tasks produce many 'unk' tokens due to limited vocabulary coverage, reducing the quality and readability of outputs.

### Limited Gains from GloVe Embeddings
Although GloVe embeddings improve performance over one-hot encoding, the improvement is relatively small, indicating limited impact within the current model setup.

### Simple Model Architecture
The use of basic LSTM and GRU models without attention mechanisms limits the ability to capture long-range dependencies and produce high-quality outputs.

Although GloVe improved performance, both tasks still showed limitations such as repetitive outputs, many unknown tokens, and signs of overfitting, indicating that larger vocabularies, better model capacity, or more advanced architectures could improve results further.

## 9. Possible Future Improvements

### Add Attention Mechanisms
Incorporating attention into the Seq2Seq model would help the decoder focus on relevant parts of the input, improving translation accuracy and reducing fragmented outputs.

### Use Subword Tokenization (BPE/WordPiece)
Applying subword techniques would reduce the number of 'unk' tokens and improve handling of rare or unseen words, leading to better text generation and translation quality.

### Adopt Transformer-Based Models
Replacing LSTM/GRU with transformer architectures can better capture long-range dependencies and significantly improve both perplexity and BLEU scores.

### Fine-tune or Use Contextual Embeddings
Allowing embeddings to be trainable or using contextual embeddings (e.g., BERT, FastText) can provide richer semantic understanding compared to static GloVe embeddings.

### Improve Training Strategy
Using techniques like early stopping, better regularization, and hyperparameter tuning can reduce overfitting and improve generalization across both tasks.


## 10. How to Run

### 1. git clone: this repo link
### 2. Run cd assignment3
### 3. pip install -r requirements.txt
### 4. python run_all.py

All outputs are saved in the results/ directory.

