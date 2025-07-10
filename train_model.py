import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib

# Load the combined dataset
df = pd.read_csv("fake_or_real_news.csv")

# Use the 'text' column as input and 'label' as target
X = df["text"]
y = df["label"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a text classification pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "ai_fake_news_model.pkl")

print("✅ Model trained and saved as ai_fake_news_model.pkl")
