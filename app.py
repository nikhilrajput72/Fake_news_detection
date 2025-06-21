import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# 🎨 Custom Page Style
# =========================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown("""
    <style>
        .main {
            background-color: #f4f4f4;
        }
        .stTextArea textarea {
            background-color: #fffbe6;
            color: #000000;
            font-size: 16px;
        }
        .stButton button {
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #FF1C1C;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# 📦 Load Model & Vectorizer
# =========================
port_stem = PorterStemmer()

vectorizer = pickle.load(open(r"D:\Personal\Project\Python\Exception\fake_news_detection\vector.pkl", "rb"))
model = pickle.load(open(r"D:\Personal\Project\Python\Exception\fake_news_detection\model.pkl", "rb"))

# =========================
# 🧼 Preprocessing Function
# =========================
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# 🧠 Prediction Function
# =========================
def Fake_news(news):
    news = wordopt(news)
    input_data = [news]
    vector = vectorizer.transform(input_data)
    prediction = model.predict(vector)
    return prediction

# =========================
# 🚀 Streamlit App UI
# =========================
st.title("📰 Fake News Detection App")
st.subheader("Check if the news you read is real or fake 🤔")

sentence = st.text_area("📥 Paste or type your news content below:", height=200)

if st.button("🔍 Predict"):
    if sentence.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        prediction_class = Fake_news(sentence)
        if prediction_class == [0]:
            st.success("✅ This looks like **reliable** news! Keep reading with confidence. 💡")
        elif prediction_class == [1]:
            st.error("🚨 This might be **fake or misleading** news! Cross-check from a trusted source. 🔎")
        else:
            st.warning("🤷 Unable to classify the news content.")
