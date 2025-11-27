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
* Conclusion

**Project Overview**
This project investigates the automatic detection of emotions in short text messages. Short-form user text often carries cues such as sentiment, intensity, and linguistic markers that require models to capture subtle local patterns. The project evaluates a broad spectrum of modeling strategies—traditional ML, feedforward neural networks, convolutional architectures, recurrent networks, and clustering—to understand how model structure and data properties interact. Through a unified pipeline, the project benchmarks each approach and analyzes their strengths, weaknesses, and suitability for short emotional text classification.

**Dataset Description:**
The dataset contains short text snippets labeled with emotion categories (e.g., love, fun, worry, boredom, hate, happiness, relief, neutral).
Each row contains:
* text — raw input message

* Emotion — corresponding label

* clean_text — lowercased, punctuation-free, stopword-filtered version

* text_length — token count

**Key characteristics:**
* Strong class imbalance with neutral dominating
* Balanced via controlled downsampling
* Typical text length < 20 words
* Final cleaned dataset stored locally and in Drive
* Used for both TF-IDF and tokenized sequence pipelines

**Methodology:**
**Preprocessing**
* Lowercasing
* Removal of non-alphabetic characters
* Stopword filtering
* Duplicate removal
* Computation of text lengths for padding decisions
* Class downsampling to mitigate imbalance

**Feature Engineering:**
Two parallel representations were used:

**TF-IDF (5,000 max features)**
For linear models, probabilistic models, and tree-based classifiers.

**Tokenized & Padded Sequences**
Using Keras Tokenizer with:
* num_words=10000
* OOV token
* maxlen=100 padding
These representations support deep learning architectures.

**Machine Learning Models:**
Traditional ML models were trained on TF-IDF features:

**Logistic Regression:**
Robust, stable, strong F1 performance
Improved via C-parameter tuning (GridSearchCV)

**Multinomial Naive Bayes**
Lightweight, but underperforms on nuanced emotional text
Suffers from independence assumptions

**Linear SVM**
Best-performing traditional classifier
Strong generalization, consistent across all emotional categories

**Decision Tree**
High accuracy but extensive overfitting
GridSearch confirms tendency to fully memorize structure

**Deep Learning Models**
Deep learning models were trained on tokenized sequences:

**Feedforward Neural Network (FFNN)**
Architecture:
Embedding → GlobalAveragePooling → Dense → Softmax
Trains steadily with no signs of overfitting
However: limited structural capacity
Does not model n-grams, local patterns, or sequential context
Underperforms CNN and traditional ML models despite high accuracy

**Convolutional Neural Network (CNN)**
Architecture:
Embedding → Conv1D → MaxPooling → Flatten → Dense → Softmax

**Performance:**
* Best overall model (~99% accuracy)
* Captures local emotional patterns effectively
* Handles both frequent and minority classes
* Fastest convergence and most stable validation performance

**LSTM:**
Architecture:
Embedding → LSTM → Dense

**Performance:**
* Fails to learn (accuracy ~22%)
* Short text sequences lack long dependency structures
* Shows why recurrent models are unnecessary here

**Unsupervised Learning:**
**KMeans (TF-IDF)**
*Adjusted Rand Index: 0.0878
*Near-random clustering
*Confirms unsupervised methods are unsuitable for emotion inference

**Results:**
**Traditional Machine Learning Models**
Logistic Regression performed as a strong and stable baseline, showing balanced F1-scores across all classes. The Support Vector Machine (SVM) was the best-performing traditional model, demonstrating robust generalization and consistently strong results across emotion categories. Naive Bayes showed significantly weaker performance, struggling with the nuances of emotional language. The Decision Tree classifier achieved very high accuracy but suffered from severe overfitting, producing unrealistic class-level scores due to memorizing patterns rather than generalizing from them.

**Deep Learning Models**
The Feedforward Neural Network (FFNN) reached an accuracy of about 97%. Although it trained steadily with no overfitting, its simple architecture proved structurally limited. It failed to capture important n-gram relationships or sequential cues in text, causing it to perform noticeably worse than both CNN and SVM despite its high accuracy.

The Convolutional Neural Network (CNN) was the best-performing model overall, achieving close to 99% accuracy. It handled both frequent and rare emotion classes very well and excelled at capturing short-range emotional patterns that appear in local phrases.

In contrast, the LSTM model performed poorly, with accuracy around 22%. Because the dataset contains very short texts, the LSTM was unable to leverage long-range dependencies and therefore failed to learn meaningful patterns, showing a clear mismatch between the model architecture and the properties of the data.

**Unsupervised Learning**
KMeans clustering performed poorly, with an Adjusted Rand Index of just 0.0878. This score is close to random, indicating that unsupervised clustering is not suitable for multi-class emotion inference in sparse or short text data.

**Key Findings**
* CNN is the dominant model, outperforming all others due to its ability to capture local phrase-level emotional patterns.
* FFNN achieves high accuracy but is fundamentally limited; absence of sequence modeling leads to significantly weaker true performance.
* SVM remains a strong baseline, rivaling deep models except CNN.
* LSTM is ineffective due to short input sequences that don’t require long-term dependencies.
* Decision Trees overfit heavily, giving misleading accuracy.
* Clustering is not viable for emotion classification in sparse text.
* TF–IDF is surprisingly competitive, but lacks semantic depth compared to learned embeddings.
  <img width="522" height="305" alt="image" src="https://github.com/user-attachments/assets/ee319648-01b5-4c46-a1a1-1e3fd0ef2a29" />



**What I Learned:**
* How to build a complete NLP classification pipeline.
* Differences between TF–IDF and embedded representations.
* Why model choice must align with data characteristics.
* How to analyze training curves, confusion patterns, and overfitting.
* Practical experience with Logistic Regression, SVM, Naive Bayes, Decision Trees, FFNN, CNN, LSTM, and KMeans.
* How short text length affects sequence-based networks.
* How to evaluate models beyond accuracy using F1-scores and qualitative insights.

**Limitations and Future Work**
**Limitations**
* Random embeddings limit semantic richness
* Minority classes remain challenging
* TF–IDF discards positional and semantic information
* Dataset contains natural noise inherent to emotional text

**Future Work**
* Use pretrained embeddings (GloVe, FastText)
* Test BERT, DistilBERT, or transformer-based models
* Explore class-weighting and data augmentation
* Investigate CNN + Attention hybrids
* Add explainability methods (SHAP, Grad-CAM)

**Conclusion**
This project delivers a rigorous, multi-model comparison for short-text emotion classification. The results show that CNNs provide the most accurate and reliable performance, outperforming both traditional ML models and other deep learning architectures. FFNNs, while stable, lack structural capacity, and LSTMs fail to learn due to the nature of the dataset. The work highlights how emotional text requires models capable of capturing local phrase-level patterns, and provides a strong foundation for future experimentation with pretrained and transformer-based architectures.
