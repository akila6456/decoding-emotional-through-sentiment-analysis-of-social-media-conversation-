#  Project Title
#**Decode Emotional Through Sentiment Analysis of Social Media Conversation**

# Install required libraries
import os
import zipfile
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gradio as gr

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# required files train.txt, test.txt, val.txt
from google.colab import files
uploaded = files.upload()

# Extract the ZIP file (manually upload it or ensure it's present)
zip_file = "emotional.zip"
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall("emotion_data")
print("Files extracted to 'emotion_data' folder")

# Load datasets
train_df = pd.read_csv("emotion_data/train.txt", sep=';', header=None, names=["text", "emotion"])
test_df = pd.read_csv("emotion_data/test.txt", sep=';', header=None, names=["text", "emotion"])
val_df = pd.read_csv("emotion_data/val.txt", sep=';', header=None, names=["text", "emotion"])

# Combine all data
df = pd.concat([train_df, test_df, val_df], ignore_index=True)

# Basic EDA
df.columns = ['text', 'emotion']
df['emotion'] = df['emotion'].replace({
    'sadness': 'negative',
    'joy': 'positive',
    'love': 'positive',
    'anger': 'negative',
    'fear': 'negative',
    'surprise': 'positive'
})

# Plot emotion distribution
sns.countplot(x='emotion', data=df)
plt.title("Sentiment Distribution")
plt.show()

# Dataset info
print("Dataset Info:")
print(df.info())
print("First 5 Records:")
print(df.head())
print("Dataset Shape:", df.shape)
print("Dataset Columns:", df.columns)

# Text Preprocessing
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+|[^A-Za-z\s]", '', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(preprocess_text)

# Check for missing and duplicated values
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicated rows:", df.duplicated().sum())

# Emotion Class Distribution
print("\nEmotion Class Distribution:\n", df['emotion'].value_counts())

# Visualize Emotion Distribution
plt.figure(figsize=(12,6))
sns.countplot(x='emotion', data=df, order=df['emotion'].value_counts().index, palette='Set2')
plt.title('Emotion Distribution in Dataset')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Define Features and Labels
X = df['text']
y = df['emotion']

print("Sample Features:\n", X.head())
print("Sample Labels:\n", y.head())

# Vectorize Text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
print("Training Data:", X_train.shape, y_train.shape)
print("Testing Data:", X_test.shape, y_test.shape)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Define Prediction Function
def predict_emotion(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return prediction

# Gradio Interface
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=2, placeholder="Enter a social media post here..."),
    outputs="text",
    title="💬 Emotion Decoder",
    description="Enter a message to detect its emotion using a trained ML model."
)
interface.launch()
