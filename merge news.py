import pandas as pd

# Load the fake and real news datasets
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

# Add labels
fake_df["label"] = "FAKE"
real_df["label"] = "REAL"

# Combine and shuffle
df = pd.concat([fake_df, real_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Keep only what's needed
df = df[["title", "text", "label"]]

# Save to CSV
df.to_csv("fake_or_real_news.csv", index=False)
print("✅ Combined dataset saved as fake_or_real_news.csv")
