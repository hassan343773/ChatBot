import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── 1. Download NLTK Resources ───────────────────────────
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ── 2. Load Dataset ──────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv('Tweets.csv')

# Keep only what we need
df = df[['text', 'airline_sentiment']]
df.columns = ['text', 'label']

# Map to 3 classes
df = df[df['label'].isin(['positive', 'negative', 'neutral'])]
print(f"✅ Dataset loaded: {len(df)} samples")
print(f"✅ Class distribution:\n{df['label'].value_counts()}\n")

# ── 3. Text Preprocessing ────────────────────────────────
print("🔄 Preprocessing text...")
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def preprocess(text):
    # lowercase
    text = text.lower()
    # remove special characters
    text = ''.join([c for c in text if c.isalpha() or c == ' '])
    # remove stopwords and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(preprocess)
print("✅ Text preprocessing done!\n")

# ── 4. Split Data ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
print(f"✅ Train size: {len(X_train)}")
print(f"✅ Test size:  {len(X_test)}\n")

# ── 5. TF-IDF Vectorization ──────────────────────────────
print("🔄 Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)
print("✅ Vectorization done!\n")

# ── 6. Train Model ───────────────────────────────────────
print("🚀 Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)
print("✅ Training complete!\n")

# ── 7. Evaluate Model ────────────────────────────────────
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Accuracy: {accuracy * 100:.2f}%\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ── 8. Confusion Matrix ──────────────────────────────────
cm = confusion_matrix(y_test, y_pred,
                      labels=['positive', 'negative', 'neutral'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Positive', 'Negative', 'Neutral'],
            yticklabels=['Positive', 'Negative', 'Neutral'])
plt.title('Confusion Matrix - Sentiment Analysis')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('sentiment_confusion_matrix.png')
plt.show()
print("✅ Confusion matrix saved!")

# ── 9. Test With Custom Input ────────────────────────────
print("\n💬 Test the model yourself!")
print("Type a sentence and see if it's Positive, Negative or Neutral")
print("Type 'quit' to exit\n")

while True:
    user_input = input("Enter text: ").strip()

    if user_input.lower() == 'quit':
        print("Goodbye!")
        break

    if not user_input:
        continue

    cleaned    = preprocess(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized).max() * 100

    print(f"➡️  Sentiment: {prediction.upper()} "
          f"(Confidence: {confidence:.1f}%)\n")