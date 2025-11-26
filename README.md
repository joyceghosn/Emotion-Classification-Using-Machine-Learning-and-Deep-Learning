**Emotion Classification with Machine Learning and Deep Learning Models**
**Introduction:**
This project presents a comprehensive analysis of emotion classification in short text messages using a range of machine learning and deep learning models. The goal is to understand how different architectures behave when exposed to short, expressive, and highly imbalanced emotional text. The pipeline includes dataset exploration, text cleaning, class balancing, feature engineering, TF-IDF vectorization, tokenization, and the evaluation of both supervised and unsupervised models. By comparing traditional ML models with FFNN, CNN, and LSTM architectures, the project identifies which techniques best capture emotional patterns in real-world short text.

**Table of Contents:**

* Project Overview

* Dataset Description

* Methodology

* Machine Learning Models

* Deep Learning Models

* Unsupervised Learning

* Results

* Key Findings

* What I Learned

* Limitations and Future Work

*Conclusion

**Project Overview**

This project investigates the automatic detection of emotions in short text messages. Short-form user text often carries cues such as sentiment, intensity, and linguistic markers that require models to capture subtle local patterns. The project evaluates a broad spectrum of modeling strategies—traditional ML, feedforward neural networks, convolutional architectures, recurrent networks, and clustering—to understand how model structure and data properties interact. Through a unified pipeline, the project benchmarks each approach and analyzes their strengths, weaknesses, and suitability for short emotional text classification.

Dataset Description

The dataset contains short text snippets labeled with emotion categories (e.g., love, fun, worry, boredom, hate, happiness, relief, neutral).
Each row contains:

text — raw input message

Emotion — corresponding label

clean_text — lowercased, punctuation-free, stopword-filtered version

text_length — token count

Key characteristics:

Strong class imbalance with neutral dominating

Balanced via controlled downsampling

Typical text length < 20 words

Final cleaned dataset stored locally and in Drive

Used for both TF-IDF and tokenized sequence pipelines

Methodology
Preprocessing

Lowercasing

Removal of non-alphabetic characters

Stopword filtering

Duplicate removal

Computation of text lengths for padding decisions

Class downsampling to mitigate imbalance

Feature Engineering

Two parallel representations were used:

TF-IDF (5,000 max features)

For linear models, probabilistic models, and tree-based classifiers.

Tokenized & Padded Sequences

Using Keras Tokenizer with:

num_words=10000

OOV token

maxlen=100 padding

These representations support deep learning architectures.

Machine Learning Models

Traditional ML models were trained on TF-IDF features:

Logistic Regression

Robust, stable, strong F1 performance

Improved via C-parameter tuning (GridSearchCV)

Multinomial Naive Bayes

Lightweight, but underperforms on nuanced emotional text

Suffers from independence assumptions

Linear SVM

Best-performing traditional classifier

Strong generalization, consistent across all emotional categories

Decision Tree

High accuracy but extensive overfitting

GridSearch confirms tendency to fully memorize structure

Deep Learning Models

Deep learning models were trained on tokenized sequences:

Feedforward Neural Network (FFNN)

Architecture:
Embedding → GlobalAveragePooling → Dense → Softmax

Updated interpretation:

Trains steadily with no signs of overfitting

However: limited structural capacity

Does not model n-grams, local patterns, or sequential context

Underperforms CNN and traditional ML models despite high accuracy

Highlights the importance of architectural alignment with data structure

Convolutional Neural Network (CNN)

Architecture:
Embedding → Conv1D → MaxPooling → Flatten → Dense → Softmax

Performance:

Best overall model (~99% accuracy)

Captures local emotional patterns effectively

Handles both frequent and minority classes

Fastest convergence and most stable validation performance

LSTM

Architecture:
Embedding → LSTM → Dense

Performance:

Fails to learn (accuracy ~22%)

Short text sequences lack long dependency structures

Shows why recurrent models are unnecessary here

Unsupervised Learning
KMeans (TF-IDF)

Adjusted Rand Index: 0.0878

Near-random clustering

Confirms unsupervised methods are unsuitable for emotion inference

Results
Traditional ML Summary
Model	Observations
Logistic Regression	Strong baseline; balanced F1 across classes
SVM	Best traditional model; robust generalization
Naive Bayes	Underperformed significantly
Decision Tree	Severe overfitting; unrealistic per-class scores
Deep Learning Summary
Model	Accuracy	Observations
FFNN	~97%	Trained steadily but structurally limited; cannot capture n-grams or sequential cues; significantly weaker than CNN and SVM
CNN	~99%	Best-performing model; excellent on all classes, including rare emotions; captures local emotional patterns
LSTM	~22%	Failed to learn; dataset too short for recurrent modeling
Unsupervised
Model	Metric	Score
KMeans	ARI	0.0878
Key Findings

CNN is the optimal architecture for short emotional text because emotions are often encoded in local phrase patterns (e.g., “feel empty”, “not happy”).

FFNNs, despite high accuracy, are fundamentally limited—they cannot model word order, n-grams, or contextual structure.

SVM is the strongest traditional model, offering competitive performance close to FFNN but below CNN.

LSTMs underperform on short text, demonstrating a mismatch between architecture and data properties.

Decision Trees overfit aggressively, making raw accuracy misleading.

TF-IDF remains a strong classical baseline, but lacks semantic depth compared to embeddings.

Unsupervised clustering is ineffective for multi-class emotional interpretation.

What I Learned

How to design and implement a full NLP pipeline from raw text to evaluation

Techniques for cleaning and balancing large text datasets

Differences between sparse (TF-IDF) and embedded sequence representations

When classical models (SVM, LR) outperform neural networks

Why architectural alignment matters: CNN excels, FFNN is limited, LSTM fails

How to diagnose model behavior through classification reports and curves

How to interpret overfitting vs generalization

The limitations of unsupervised learning in complex classification tasks

Best practices for saving models, generating predictions, and organizing results

Limitations and Future Work
Limitations

Random-initialized embeddings limit semantic richness

LSTM architecture not optimized for short sequences

TF-IDF cannot encode semantic or positional information

Dataset labeling may contain noise typical of social text

Future Work

Integrate pretrained embeddings (GloVe, FastText) or transformer models (BERT, DistilBERT)

Investigate class-weighted training for minority emotions

Add attention layers or hybrid CNN-LSTM architectures

Explore explainability tools such as SHAP or Grad-CAM for CNN

Extend to production deployment or real-time emotion monitoring

**Conclusion:**

This project provides a rigorous comparative study of machine learning and deep learning methods for emotion classification in short text. The results demonstrate that convolutional neural networks outperform all other models due to their ability to capture local linguistic patterns strongly tied to emotional expression. FFNNs, although achieving high accuracy, fail to model contextual or sequential information, and LSTMs underperform due to the brevity of the text. Traditional ML models, particularly SVM, remain competitive baselines. Overall, the project highlights how architectural suitability, feature representation, and text length fundamentally shape model performance in emotion detection tasks.
