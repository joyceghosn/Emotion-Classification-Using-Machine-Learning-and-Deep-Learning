Emotion Classification with Machine Learning and Deep Learning Models
Introduction

This project investigates the performance of multiple machine learning and deep learning models for multi-class emotion classification in short text messages. It builds a complete NLP pipeline that includes data cleaning, class rebalancing, feature engineering, TF-IDF vectorization, tokenization, and the evaluation of traditional classifiers, neural network architectures, and unsupervised clustering. The objective is to understand how different modeling strategies behave on the same dataset and identify which techniques are most effective for recognizing emotional cues in real-world, short-form text.

Table of Contents

Project Overview

Dataset Description

Methodology

Machine Learning Models

Deep Learning Models

Unsupervised Learning

Results

Key Findings

What I Learned

Limitations and Future Work

Conclusion

Project Overview

The task addressed in this project is the automatic classification of emotions expressed in short text samples. Short messages on the internet often contain emotional cues that are difficult for simplistic models to interpret. This project compares a spectrum of approaches—from linear models to CNNs—to understand which architectures best capture emotional signals and why.

The workflow integrates:

Dataset loading and exploration

Text normalization and stopword removal

Class balancing through targeted downsampling

TF-IDF vectorization for traditional models

Tokenization and sequence padding for neural networks

Training and evaluation of multiple classification algorithms

Visualization and interpretability components such as word clouds and training curves

All implementation details are based on the notebook code. 

emotion_classification_ml

Dataset Description

The dataset consists of short text messages labeled with an emotion category (e.g., love, fun, worry, boredom, hate, happiness, relief, neutral).
Each row contains:

text — raw user message

Emotion — target class

clean_text — preprocessed version

text_length — number of tokens

Key properties:

Highly imbalanced, with neutral being the most frequent category

Duplicate rows removed

Balanced through downsampling to achieve a uniform distribution across labels

Short message lengths (typically < 20 words), influencing neural model design such as maximum sequence length and padding strategy

Methodology
Preprocessing

Lowercasing

Removal of punctuation and non-alphabetic characters

Stopword filtering (scikit-learn’s built-in list)

Creation of clean_text

Word-count distribution analysis

Class rebalancing to mitigate severe label imbalance

Feature Engineering

Two parallel representations were created:

1. TF–IDF vectors (5,000 features)

Used for traditional ML classifiers.

2. Tokenized + padded sequences (maxlen=100)

Used for deep learning models (Embedding → NN architectures).

Both pipelines are independently trained and evaluated to highlight strengths and weaknesses.

Machine Learning Models

Four supervised traditional classifiers were implemented on TF-IDF features:

Logistic Regression

Strong generalization and stable performance

Hyperparameter tuning (C) optimized via GridSearchCV

Balanced F1-scores across classes

Multinomial Naive Bayes

Fast but limited

Underperforms due to conditional independence assumptions

Linear Support Vector Machine (SVM)

One of the strongest traditional baselines

High F1 scores across all labels

Robust to noise and high-dimensional sparsity

Decision Tree

High accuracy but severe overfitting

Hyperparameters (max_depth, min_samples_split) confirm tendency to memorize training data

These models provide strong baselines before moving to neural architectures.

Deep Learning Models

Deep learning models were trained using tokenized and padded sequences, enabling comparison against traditional approaches.

Feedforward Neural Network (FFNN)

Architecture:
Embedding → GlobalAveragePooling1D → Dense → Softmax

Performance:

~95.8% validation accuracy

Strong generalization

Minor weaknesses on rare classes such as boredom and empty

Convolutional Neural Network (CNN)

Architecture:
Embedding → Conv1D → MaxPooling → Flatten → Dense → Softmax

Performance:

~99% accuracy and near-perfect macro and weighted F1

Excels at capturing short-range patterns (n-grams) such as “not happy”, “feel empty”, etc.

Outperforms all other models, including FFNN and traditional ML

Fastest and most stable training curves among all deep learning models

LSTM

Architecture:
Embedding → LSTM → Dense

Performance:

Accuracy stagnant at ~22%

Fails to learn emotional structure

Underperforms due to short sequences that do not benefit from recurrent architecture

Demonstrates the importance of matching model architecture to data characteristics

Unsupervised Learning
KMeans Clustering

Trained on TF-IDF vectors

Evaluated with Adjusted Rand Index: 0.0878

Indicates clustering is essentially random

Confirms unsupervised methods are unsuitable for multi-class emotional interpretation

Results
Traditional ML Summary
Model	Performance Notes
Logistic Regression	Strong baseline, stable results
SVM	Best among traditional ML, high F1 across all emotions
Naive Bayes	Underperformed significantly
Decision Tree	Overfitting, unrealistic scores
Deep Learning Summary
Model	Accuracy	Observations
FFNN	~97%	Solid performance; struggles with minority classes
CNN	~99%	Best overall model; excellent on rare and frequent classes alike
LSTM	~22%	Failed to learn; unsuitable for short texts
Unsupervised
Model	Metric	Score
KMeans	ARI	0.0878
Key Findings

CNN is the superior architecture for short-text emotion classification.
It captures local patterns (n-grams) that strongly correlate with emotional content.

LSTM underperforms when text sequences are short.
Long-range dependencies are minimal in this dataset, making LSTMs ineffective.

SVM is the most reliable traditional model, offering strong generalization without overfitting.

Decision Trees achieve deceptively high accuracy due to memorization, not true pattern learning.

TF-IDF is a powerful and surprisingly competitive baseline, often rivaling neural models except CNN.

Unsupervised learning is not suitable for emotional inference in sparse, high-dimensional text.

Class balance is crucial; downsampling ensures that models do not default to the majority (neutral) class.

What I Learned

How to build an end-to-end NLP classification pipeline.

The importance of text preprocessing and effective feature engineering.

Practical understanding of TF-IDF vs tokenized sequence representations.

Differences in behavior between linear models, tree-based models, CNNs, FFNNs, and LSTMs.

How architectural choices must align with data properties (e.g., CNNs for short phrases).

How to interpret evaluation metrics beyond accuracy and detect overfitting.

How unsupervised methods behave on labeled text.

How to visualize textual data through word clouds and distribution plots.

How to save, compare, and analyze predictions across models.

Deep practical intuition for why CNNs excel in NLP tasks involving short-form text.

Limitations and Future Work
Limitations

Neural models rely on randomly initialized embeddings; pretrained embeddings could provide richer semantics.

LSTM architecture was not tuned extensively; alternative recurrent designs might improve performance.

TF-IDF does not capture word order or semantic nuance.

Dataset may contain labeling noise typical of user-generated emotional data.

Future Work

Integrate pretrained embeddings (e.g., GloVe, FastText) or transformer models (BERT, DistilBERT).

Add regularization or pruning for Decision Trees.

Explore data augmentation for rare emotional classes.

Implement attention-based architectures.

Extend the pipeline to production-level deployment for emotion-aware applications.

Conclusion

This project presents a comprehensive comparison of machine learning and deep learning techniques for multi-class emotion classification in short text messages. The findings demonstrate that convolutional neural networks significantly outperform both traditional models and other neural architectures, confirming their ability to capture localized emotional expressions. By building and evaluating a unified pipeline, this work highlights how model selection, feature representation, and data characteristics interact to influence performance. The results provide a solid foundation for future work using advanced embeddings or transformer-based models, and offer valuable insights for real-world applications requiring automated emotion detection.
