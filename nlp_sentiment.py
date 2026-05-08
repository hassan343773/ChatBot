import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = ''.join([c for c in text if c.isalpha() or c == ' '])
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# ── 1. Load Airline Dataset ──────────────────────────────
print("📂 Loading Tweets.csv (Airline)...")
df1 = pd.read_csv('Tweets.csv')
df1 = df1[['text', 'airline_sentiment']].dropna()
df1.columns = ['text', 'label']
df1 = df1[df1['label'].isin(['positive', 'negative', 'neutral'])]
print(f"✅ Airline dataset: {len(df1)} samples")

# ── 2. Load Twitter Dataset ──────────────────────────────
print("📂 Loading Twitter_Data.csv...")
df2 = pd.read_csv('Twitter_Data.csv')
df2 = df2[['clean_text', 'category']].dropna()
df2.columns = ['text', 'label']
df2['label'] = df2['label'].astype(float).map({
    -1.0: 'negative',
     0.0: 'neutral',
     1.0: 'positive'
})
df2 = df2.dropna()
print(f"✅ Twitter dataset: {len(df2)} samples")

# ── 3. Combine Both ──────────────────────────────────────
df = pd.concat([df1, df2], ignore_index=True).sample(frac=1, random_state=42)
print(f"\n✅ Combined dataset: {len(df)} samples")
print(f"✅ Class distribution:\n{df['label'].value_counts()}\n")

# ── 4. Preprocess ────────────────────────────────────────
print("🔄 Preprocessing text...")
df['clean'] = df['text'].apply(preprocess)
print("✅ Done!\n")

# ── 5. Split ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}\n")

# ── 6. Vectorize ─────────────────────────────────────────
print("🔄 Vectorizing with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3))
X_train_v  = vectorizer.fit_transform(X_train)
X_test_v   = vectorizer.transform(X_test)
print("✅ Done!\n")

# ── 7. Train ─────────────────────────────────────────────
print("🚀 Training model on combined dataset...")
model = LogisticRegression(max_iter=1000, C=5.0, random_state=42)
model.fit(X_train_v, y_train)
print("✅ Training complete!\n")

# ── 8. Evaluate ──────────────────────────────────────────
y_pred   = model.predict(X_test_v)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Accuracy: {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# ── 9. Confusion Matrix ──────────────────────────────────
cm = confusion_matrix(y_test, y_pred,
                      labels=['positive', 'negative', 'neutral'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Positive', 'Negative', 'Neutral'],
            yticklabels=['Positive', 'Negative', 'Neutral'])
plt.title('Confusion Matrix - Combined Sentiment Analysis')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('sentiment_confusion_matrix.png')
plt.show()

# ── 10. Save ─────────────────────────────────────────────
print("\n💾 Saving model...")
with open('sentiment_model.pkl',      'wb') as f:
    pickle.dump(model, f)
with open('sentiment_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✅ Saved!")

# ── 11. Test with real examples ──────────────────────────
print("\n💬 Testing with real sentences:")
tests = [
    "I am a student",
    "I love this so much!",
    "This is absolutely terrible",
    "The weather is okay today",
    "I hate Mondays",
    "Best day of my life!",
    "It is what it is",
]
for t in tests:
    c    = preprocess(t)
    v    = vectorizer.transform([c])
    p    = model.predict(v)[0]
    conf = model.predict_proba(v).max() * 100
    print(f"  '{t}' → {p.upper()} ({conf:.1f}%)")