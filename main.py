import streamlit as st
import joblib
import requests
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("app/model/ai_fake_news_model.pkl")

model = load_model()

# Label map
label_map = {0: "❌ Fake", 1: "✅ Real"}

# App UI
st.set_page_config(page_title="Fake News Checker", page_icon="🧠")
st.title("🧠 AI-Powered Real-Time Fake News Checker")
st.write("Enter a news headline or short article text:")

# Text input
query = st.text_area("📰 Your News Text", height=150)

# Extract keywords for API query
def clean_query(text):
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return ' '.join(keywords[:6])  # limit to top 6 keywords

# Generate possible real alternative
def suggest_real_alternative(fake_text):
    alt_text = fake_text

    # Replace specific verbs with negative phrasing
    alt_text = re.sub(r"\bbuys\b", "did not buy", alt_text, flags=re.IGNORECASE)
    alt_text = re.sub(r"\bhas bought\b", "has not bought", alt_text, flags=re.IGNORECASE)
    alt_text = re.sub(r"\bwill buy\b", "will not buy", alt_text, flags=re.IGNORECASE)
    alt_text = re.sub(r"\bbought\b", "did not buy", alt_text, flags=re.IGNORECASE)
    alt_text = re.sub(r"\bis\b", "is not", alt_text, flags=re.IGNORECASE)
    alt_text = re.sub(r"\bare\b", "are not", alt_text, flags=re.IGNORECASE)

    # Avoid repetitive "not not"
    alt_text = re.sub(r"\bnot not\b", "not", alt_text, flags=re.IGNORECASE)

    # Fallback if no change
    if alt_text.strip().lower() == fake_text.strip().lower():
        alt_text = "This claim is not supported by real events."

    return alt_text


# Fetch related real news articles using NewsAPI
def fetch_real_news_links(query):
    api_key = "d53925bcc4c346929031ba0232dfa340"
    url = "https://newsapi.org/v2/everything"
    cleaned_query = clean_query(query)

    st.caption(f"🔍 Searching NewsAPI for: `{cleaned_query}`")

    params = {
        "q": cleaned_query,
        "language": "en",
        "pageSize": 5,
        "sortBy": "relevancy",
        "apiKey": api_key
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if data.get("status") != "ok":
            return []

        return [(article["title"], article["url"]) for article in data["articles"]]

    except Exception as e:
        st.error(f"News fetch error: {e}")
        return []

# Handle prediction and output
if st.button("🔍 Check"):
    if query.strip():
        prediction = model.predict([query])[0]
        predicted_label = label_map[prediction]

        st.markdown("### 🧠 AI Prediction:")
        if prediction == 1:
            st.success(predicted_label)
        else:
            st.error(predicted_label)

        if prediction == 1:
            st.markdown("### 📡 Real News Articles Related to Your Input:")
            links = fetch_real_news_links(query)

            if links:
                for i, (title, link) in enumerate(links, 1):
                    st.markdown(f"{i}. [{title}]({link})")
            else:
                st.warning("No related real news found. Try rephrasing or using keywords.")
        else:
            st.markdown("### 🪄 Possible Real News Interpretation:")
            suggestion = suggest_real_alternative(query)
            st.info(suggestion)

            st.markdown("### 📡 Real News Articles Related to Your Input:")
            st.info("This appears to be fake news. Real articles are not shown to avoid amplifying false claims.")
    else:
        st.warning("Please enter a news headline or short article.")
