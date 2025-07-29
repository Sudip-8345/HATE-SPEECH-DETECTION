import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Constants
max_len = 300
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ✅ Use Streamlit caching to avoid reloading model/tokenizer every time
@st.cache_resource
def load_model_and_tokenizer():
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = tf.keras.models.load_model('models/model.h5')
    return model, tokenizer

# ✅ Text cleaning function
def clean_text(words):
    words = str(words).lower()
    words = re.sub(r'\[.*?\]', '', words)
    words = re.sub(r'https?://\S+|www\.\S+', '', words)
    words = re.sub(r'<.*?>+', '', words)
    words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
    words = re.sub(r'\n', '', words)
    words = re.sub(r'\w*\d\w*', '', words)

    words = [word for word in words.split(' ') if word not in stop_words and word != '']
    words = [stemmer.stem(word) for word in words]
    words = " ".join(words)

    return words

# ✅ Streamlit UI wrapped in main()
def main():
    st.title("🛡️ Hate Speech Detector")
    st.write("Paste a message to detect whether it's hate speech or not.")

    user_input = st.text_area("💬 Enter your message here:")

    if st.button("🧠 Predict"):
        model, tokenizer = load_model_and_tokenizer()
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len)
        prediction = model.predict(padded)[0][0]

        if prediction > 0.5:
            st.error(f"⚠️ Hate/Offensive Message (Confidence: {prediction:.2f})")
        else:
            st.success(f"✅ Not Hate Message (Confidence: {1 - prediction:.2f})")

# ✅ Run the main() function
if __name__ == "__main__":
    main()
