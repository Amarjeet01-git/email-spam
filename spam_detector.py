import streamlit as st
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

emails = [
    "Congratulations! You have won a $1000 Walmart gift card. Click here to claim now!",
    "Get rich quick with this simple investment plan. Join today!",
    "You have been selected for a free iPhone. Claim your prize immediately!",
    "Earn $5000 per week from home. No experience required!",
    "Limited time offer! Buy one get one free on all luxury watches!",
    "Your account has been compromised. Reset your password immediately!",
    "Exclusive lottery winner! Confirm your bank details to receive the prize.",
    "Make money online fast. Work from home and earn thousands!",
    "Win a free cruise vacation. Click here to register!",
    "Special promotion: Get 70% off on premium software today!",

    "Please review the attached report and send your feedback by tomorrow.",
    "Meeting scheduled for next Monday at 10 AM. Confirm your attendance.",
    "The team project deadline has been extended to next Friday.",
    "Reminder: Parent-teacher meeting will be held this Wednesday.",
    "Can you please share the minutes of today's meeting?",
    "Your subscription invoice for December has been generated.",
    "Kindly approve the leave request submitted by John.",
    "Please find attached the presentation for tomorrow's client meeting.",
    "The server maintenance is scheduled for 2 AM tonight.",
    "Update your contact details in the HR portal before Friday.",

    "Act now! Limited time offer on cryptocurrency investment. Don't miss!",
    "Get your free Amazon voucher by clicking this link immediately!",
    "Congratulations! You have been pre-approved for a $5000 loan.",
    "Earn money fast by participating in our survey program.",
    "Free access to premium courses for a limited period. Register now!",
    "Hot singles in your area want to meet you. Join now!",
    "Exclusive deal: 90% discount on all electronic gadgets today!",
    "Claim your free cashback reward by verifying your details now!",
    "Limited spots available for our paid webinar. Register before it's full!",
    "You are a winner! Confirm your prize before the deadline expires."
]

labels = [
    1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1
]

emails = [clean_text(e) for e in emails]
emails, labels = shuffle(emails, labels, random_state=42)

@st.cache_resource
def train_model():
    x_train, x_test, y_train, y_test = train_test_split(
        emails, labels, test_size=0.3, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
        ('clf', MultinomialNB())
    ])

    pipeline.fit(x_train, y_train)
    acc = accuracy_score(y_test, pipeline.predict(x_test))
    joblib.dump(pipeline, "spam_model.joblib")
    return pipeline, acc

model, accuracy = train_model()

st.title("📧 Email Spam Detection")
st.write(f"Model Accuracy: **{accuracy:.2f}**")

user_input = st.text_area("Enter email text:")

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
