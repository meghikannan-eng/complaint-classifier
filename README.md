# 🚨 Customer Complaint Severity Classifier

An end-to-end NLP + Machine Learning project that automatically 
classifies customer complaints into **High**, **Medium**, or **Low** priority.

---

## 📌 Project Overview

In customer support, manually triaging hundreds of complaints is 
slow and error-prone. This project builds an automated classifier 
that reads a complaint message and instantly predicts its urgency level.

| Priority | Example |
|---|---|
| 🔴 High | "Unauthorized transaction on my account, fix immediately!" |
| 🟡 Medium | "My order hasn't arrived after 10 days" |
| 🟢 Low | "The app UI could look better" |

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **NLP:** NLTK, SpaCy
- **ML Models:** Logistic Regression, SVM, XGBoost, RandomForest
- **Deep Learning:** LSTM (Keras + TensorFlow)
- **Feature Extraction:** TF-IDF, CountVectorizer, Word2Vec
- **Deployment:** Streamlit
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---
# 🚨 Customer Complaint Severity Classifier

An end-to-end NLP + Machine Learning project that automatically 
classifies customer complaints into **High**, **Medium**, or **Low** priority.

---

## 📌 Project Overview

In customer support, manually triaging hundreds of complaints is 
slow and error-prone. This project builds an automated classifier 
that reads a complaint message and instantly predicts its urgency level.

| Priority | Example |
|---|---|
| 🔴 High | "Unauthorized transaction on my account, fix immediately!" |
| 🟡 Medium | "My order hasn't arrived after 10 days" |
| 🟢 Low | "The app UI could look better" |

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **NLP:** NLTK, SpaCy
- **ML Models:** Logistic Regression, SVM, XGBoost, RandomForest
- **Deep Learning:** LSTM (Keras + TensorFlow)
- **Feature Extraction:** TF-IDF, CountVectorizer, Word2Vec
- **Deployment:** Streamlit
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

## 📂 Project Structure

complaint-classifier/
├── streamlit_app.py          # Streamlit deployment app
├── ml_classifiers.py         # ML models (LR, SVM, XGBoost)
├── lstm_classifier.py        # Deep Learning LSTM model
├── text_preprocessing.py     # Text cleaning pipeline
├── feature_extraction.py     # TF-IDF, BoW, Word2Vec
├── rf_feature_importance.py  # Feature importance ranking
├── model_evaluation.py       # Final model comparison
├── complaints.csv            # Dataset
└── requirements.txt          # Dependencies

---

## ⚙️ Skills Demonstrated

- Text preprocessing (cleaning, tokenization, lemmatization)
- Feature extraction using TF-IDF, CountVectorizer, Bag of Words
- Deep learning with LSTM and Embedding layers
- ML model training and hyperparameter tuning
- Model evaluation (F1, ROC-AUC, Confusion Matrix)
- Streamlit web app deployment

---

## 🚀 How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the app
streamlit run streamlit_app.py
```

---

## 📊 Model Results

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | ~85% | ~84% |
| SVM | ~87% | ~86% |
| XGBoost | ~88% | ~87% |
| LSTM | ~89% | ~88% |

---

## 👩‍💻 Author

**Meghikannan**  
Data Science Project — NLP & Machine Learning
