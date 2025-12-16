import streamlit as st
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')  # Kaggle dataset
    df = df[['v1','v2']]
    df.columns = ['label','text']
    df['label'] = df['label'].map({'ham':0, 'spam':1})
    df['text'] = df['text'].apply(clean_text)
    return shuffle(df)

df = load_data()
emails = df['text'].tolist()
labels = df['label'].tolist()

@st.cache_resource
def train_model():
    x_train, x_test, y_train, y_test = train_test_split(
        emails, labels, test_size=0.3, random_state=42, stratify=labels
    )
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english')),
        ('clf', MultinomialNB())
    ])
    pipeline.fit(x_train, y_train)
    acc = accuracy_score(y_test, pipeline.predict(x_test))
    joblib.dump(pipeline, "spam_model.joblib")
    return pipeline, acc

model, accuracy = train_model()

st.set_page_config(page_title="Email Spam Detection", page_icon="📧", layout="centered")
st.title("📧 Email Spam Detection")
st.write(f"Model Accuracy: **{accuracy:.2f}**")

user_input = st.text_area("Enter email text here:")

if st.button("Check Email"):
    if user_input.strip() == "":
        st.warning("Please enter some email text")
    else:
        cleaned = clean_text(user_input)
        prediction = model.predict([cleaned])[0]
        if prediction == 1:
            st.error("🚨 This Email is SPAM")
        else:
            st.success("✅ This Email is NOT SPAM")
