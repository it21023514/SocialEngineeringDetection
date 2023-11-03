from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

nltk.download('stopwords')

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
with open('model.pkl', 'rb') as model_file, open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    model = pickle.load(model_file)
    tfidf_vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        # Preprocess the input text
        text = preprocess_text(text)
        # Vectorize the text using the TF-IDF vectorizer
        text_vectorized = tfidf_vectorizer.transform([text])
        # Predict using the trained model
        prediction = model.predict(text_vectorized)
        result = 'Social Engineering Attempt' if prediction[0] == 1 else 'Legitimate Communication'
        return render_template('index.html', result=result, text=text)

def preprocess_text(text):
    # Preprocessing steps (remove stopwords, punctuation, convert to lowercase, etc.)
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

if __name__ == '__main__':
    app.run(debug=True)
