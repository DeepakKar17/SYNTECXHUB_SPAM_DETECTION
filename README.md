# SYNTECXHUB_SPAM_DETECTION
# Project 2 - Spam Detection
**SyntecxHub Internship Project**

This project builds a spam detection classifier using NLP techniques.

### Workflow:
1. Load and explore the spam/ham dataset
2. Preprocess text (tokenize, clean)
3. Convert text to vectors using CountVectorizer / TF-IDF
4. Train Naive Bayes and Logistic Regression models
5. Evaluate with precision, recall, and F1-score
6. Save the pipeline for reuse
## Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import string
import nltk
import joblib
import warnings

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

print('All libraries imported successfully.')
## Step 2: Load the Dataset

We use the **SMS Spam Collection Dataset** from UCI Machine Learning Repository.  
You can download it from: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection  
Or use the Kaggle version: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Place the file `spam.csv` in the same folder as this notebook.
# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only relevant columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print('Dataset shape:', df.shape)
df.head(10)
## Step 3: Exploratory Data Analysis (EDA)
# Check for missing values
print('Missing values:')
print(df.isnull().sum())

print('\nClass distribution:')
print(df['label'].value_counts())
print(f'\nSpam percentage: {df["label"].value_counts(normalize=True)["spam"]*100:.2f}%')
# Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

colors = ['#4CAF50', '#F44336']

df['label'].value_counts().plot(
    kind='bar', ax=axes[0], color=colors, edgecolor='black'
)
axes[0].set_title('Class Distribution (Bar Chart)', fontsize=13)
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)

df['label'].value_counts().plot(
    kind='pie', ax=axes[1], autopct='%1.1f%%',
    colors=colors, startangle=140, shadow=True
)
axes[1].set_title('Class Distribution (Pie Chart)', fontsize=13)
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
# Message length analysis
df['msg_length'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda x: len(x.split()))

print('Average message length by class:')
print(df.groupby('label')[['msg_length', 'word_count']].mean().round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for label, color in zip(['ham', 'spam'], ['#4CAF50', '#F44336']):
    axes[0].hist(
        df[df['label'] == label]['msg_length'],
        bins=40, alpha=0.6, label=label, color=color
    )
axes[0].set_title('Message Length Distribution')
axes[0].set_xlabel('Character Count')
axes[0].set_ylabel('Frequency')
axes[0].legend()

for label, color in zip(['ham', 'spam'], ['#4CAF50', '#F44336']):
    axes[1].hist(
        df[df['label'] == label]['word_count'],
        bins=30, alpha=0.6, label=label, color=color
    )
axes[1].set_title('Word Count Distribution')
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].legend()

plt.tight_layout()
plt.savefig('length_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
## Step 4: Text Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and preprocesses a raw text message:
    - Lowercases
    - Removes URLs, email addresses, phone numbers
    - Removes punctuation and digits
    - Tokenizes
    - Removes stopwords
    - Applies stemming
    """
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove phone numbers
    text = re.sub(r'\b\d{10,}\b', '', text)

    # Remove punctuation and digits
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and short tokens, apply stemming
    tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    return ' '.join(tokens)


# Apply preprocessing
df['cleaned_message'] = df['message'].apply(preprocess_text)

print('Sample original vs cleaned messages:')
for i in range(3):
    print(f'\n[{df["label"].iloc[i].upper()}]')
    print('  Original:', df['message'].iloc[i])
    print('  Cleaned :', df['cleaned_message'].iloc[i])
## Step 5: Encode Labels and Split Data
# Encode labels: spam = 1, ham = 0
df['label_encoded'] = df['label'].map({'spam': 1, 'ham': 0})

X = df['cleaned_message']
y = df['label_encoded']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'Training samples : {len(X_train)}')
print(f'Testing samples  : {len(X_test)}')
print(f'\nTraining class distribution:')
print(y_train.value_counts())
## Step 6: Build Pipelines with CountVectorizer and TF-IDF
# Define all pipelines
pipelines = {
    'NaiveBayes + CountVectorizer': Pipeline([
        ('vectorizer', CountVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ]),
    'NaiveBayes + TF-IDF': Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ]),
    'LogisticRegression + CountVectorizer': Pipeline([
        ('vectorizer', CountVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'LogisticRegression + TF-IDF': Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
}

print('Pipelines defined:', list(pipelines.keys()))
## Step 7: Train and Evaluate All Models
results = []

for name, pipeline in pipelines.items():
    print(f'Training: {name} ...')
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False)
results_df = results_df.set_index('Model')
print('\nModel Comparison:')
print(results_df.round(4).to_string())
# Plot model comparison
ax = results_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(
    kind='bar', figsize=(13, 5), edgecolor='black', width=0.7
)
ax.set_title('Model Performance Comparison', fontsize=14)
ax.set_ylabel('Score')
ax.set_ylim(0.85, 1.02)
ax.legend(loc='lower right')
ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
## Step 8: Detailed Report for the Best Model
best_model_name = results_df['F1 Score'].idxmax()
best_pipeline = pipelines[best_model_name]
y_pred_best = best_pipeline.predict(X_test)

print(f'Best Model: {best_model_name}')
print('='*55)
print(classification_report(y_test, y_pred_best, target_names=['ham', 'spam']))
# Confusion matrix for best model
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Ham (0)', 'Spam (1)'],
    yticklabels=['Ham (0)', 'Spam (1)'],
    linewidths=0.5, linecolor='gray'
)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=13)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
## Step 9: Top Spam & Ham Words (Feature Importance)
vectorizer = best_pipeline.named_steps['vectorizer']
classifier = best_pipeline.named_steps['classifier']
feature_names = vectorizer.get_feature_names_out()

if hasattr(classifier, 'feature_log_prob_'):
    # Naive Bayes: log probabilities
    spam_idx = 1
    ham_idx = 0
    log_prob_spam = classifier.feature_log_prob_[spam_idx]
    log_prob_ham = classifier.feature_log_prob_[ham_idx]
    spam_diff = log_prob_spam - log_prob_ham

    top_spam_idx = spam_diff.argsort()[-20:][::-1]
    top_ham_idx = spam_diff.argsort()[:20]

    top_spam_words = [(feature_names[i], spam_diff[i]) for i in top_spam_idx]
    top_ham_words = [(feature_names[i], -spam_diff[i]) for i in top_ham_idx]

elif hasattr(classifier, 'coef_'):
    # Logistic Regression: coefficients
    coef = classifier.coef_[0]
    top_spam_idx = coef.argsort()[-20:][::-1]
    top_ham_idx = coef.argsort()[:20]

    top_spam_words = [(feature_names[i], coef[i]) for i in top_spam_idx]
    top_ham_words = [(feature_names[i], -coef[i]) for i in top_ham_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

words_s, scores_s = zip(*top_spam_words)
axes[0].barh(words_s[::-1], scores_s[::-1], color='#F44336', edgecolor='black')
axes[0].set_title('Top 20 Spam Indicator Words', fontsize=13)
axes[0].set_xlabel('Score')

words_h, scores_h = zip(*top_ham_words)
axes[1].barh(words_h[::-1], scores_h[::-1], color='#4CAF50', edgecolor='black')
axes[1].set_title('Top 20 Ham Indicator Words', fontsize=13)
axes[1].set_xlabel('Score')

plt.tight_layout()
plt.savefig('top_words.png', dpi=150, bbox_inches='tight')
plt.show()
## Step 10: Save the Best Pipeline
joblib.dump(best_pipeline, 'spam_detector_pipeline.pkl')
print(f'Pipeline saved as spam_detector_pipeline.pkl')
print(f'Saved model: {best_model_name}')
## Step 11: Load Pipeline and Make Predictions
# Load the saved pipeline
loaded_pipeline = joblib.load('spam_detector_pipeline.pkl')

def predict_spam(messages):
    """
    Takes a list of raw messages and returns predictions.
    Returns 'SPAM' or 'HAM' for each message.
    """
    cleaned = [preprocess_text(msg) for msg in messages]
    preds = loaded_pipeline.predict(cleaned)
    probas = loaded_pipeline.predict_proba(cleaned)

    print(f'{"Message":<60} {"Prediction":<12} {"Spam Prob"}')
    print('-' * 85)
    for msg, pred, proba in zip(messages, preds, probas):
        label = 'SPAM' if pred == 1 else 'HAM'
        short_msg = msg[:57] + '...' if len(msg) > 60 else msg
        print(f'{short_msg:<60} {label:<12} {proba[1]:.4f}')


# Test messages
test_messages = [
    "Congratulations! You have won a FREE iPhone. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow at 1pm?",
    "URGENT: Your bank account has been compromised. Call 1800-XXX immediately.",
    "Can you please send me the project report by end of day?",
    "Win 1000 pounds cash! Text WIN to 87121 to claim your prize. Free entry.",
    "I am on my way home. Do you need anything from the store?"
]

predict_spam(test_messages)
## Summary

| Step | Description |
|------|-------------|
| 1 | Imported all required libraries |
| 2 | Loaded SMS Spam Collection dataset |
| 3 | Performed EDA - class distribution & message lengths |
| 4 | Preprocessed text - lowercasing, cleaning, stemming |
| 5 | Encoded labels and split data (80/20) |
| 6 | Built 4 pipelines (2 vectorizers x 2 classifiers) |
| 7 | Trained and compared all models |
| 8 | Detailed classification report + confusion matrix |
| 9 | Visualized top spam/ham indicator words |
| 10 | Saved best pipeline with joblib |
| 11 | Loaded pipeline and tested on new messages |
