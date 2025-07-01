# Text-Sentiment-Classification

## Project Overview

This project explores sentiment classification—a key task in Natural Language Processing (NLP). It compares traditional machine learning methods (Naive Bayes, SVM, Logistic Regression, Random Forest) and deep learning models (CNN, GRU, BERT) using the MELD dataset, which contains multi-party dialogue from the TV series *Friends*.
BERT outperforms both traditional ML and other deep learning models, achieving the highest classification accuracy (83%).

---

## Dataset

- **Name**: MELD (Multimodal EmotionLines Dataset)
- **Source**: Dialogues from *Friends*
- **Size**: ~13,000 utterances
- **Labels**: Joy, Sadness, Anger, Fear, Neutral
- **Split**: 70% Train, 15% Validation, 15% Test

---

## 🛠️ System Pipeline

### 1. Traditional Machine Learning
- **Preprocessing**: cleaning, tokenization, stemming, negation handling
- **Feature Extraction**: TF-IDF (1-gram + 2-gram)
- **Models**:
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Random Forest

### 2. Deep Learning
- **CNN**: 1D convolution + global max pooling
- **GRU**: Bi-directional GRU with recurrent dropout
- **BERT**: Pretrained Transformer-based model with fine-tuning

---

## Model Performance

| Model              | Accuracy  |
|--------------------|-----------|
| Naive Bayes        | 67.02%    |
| Random Forest      | 67.83%    |
| Logistic Regression| 69.35%    |
| SVM                | 72.71%    |
| CNN                | 75.48%    |
| GRU                | 73.56%    |
| **BERT**           | **83.00%**|

---

## Key Insights

- TF-IDF is effective but struggles with contextual sentiment like “neutral”.
- CNN is efficient for short text but limited in capturing long dependencies.
- GRU generalizes better than CNN but takes longer to train.
- BERT captures bidirectional context and provides superior performance, especially for nuanced emotions.

---

## Confusion Matrix Highlights

- All traditional models struggled with the “neutral” class due to lack of contextual understanding.
- BERT reduced misclassifications across all classes, especially for subtle sentiments.

---

## Limitations

- **Data size**: Small-scale due to hardware constraints.
- **Embedding**: Only Word2Vec was used; GloVe/FastText were not tested.
- **Domain adaptation**: No transfer learning across domains was implemented.

---


## Future Work

- Evaluate alternative embeddings (GloVe, FastText)
- Apply domain adaptation and transfer learning
- Extend to multimodal sentiment analysis (text + audio + video)

