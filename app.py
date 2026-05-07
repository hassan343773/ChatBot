from flask import Flask, render_template, request, jsonify
import pandas as pd
from groq import Groq
import numpy as np
import nltk
import pickle
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

app = Flask(__name__)



GROQ_API_KEY = "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"
groq_client  = Groq(api_key=GROQ_API_KEY)

# ── NLP Setup ────────────────────────────────────────────
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

# Train NLP model on startup
print("🔄 Loading NLP model...")
df = pd.read_csv('Tweets.csv')
df = df[['text', 'airline_sentiment']]
df.columns = ['text', 'label']
df = df[df['label'].isin(['positive', 'negative', 'neutral'])]
df['clean_text'] = df['text'].apply(preprocess)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['clean_text'])
nlp_model = LogisticRegression(max_iter=1000, random_state=42)
nlp_model.fit(X, df['label'])
print("✅ NLP model ready!")

# ── Image Model Setup ────────────────────────────────────
CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
img_model = None
if os.path.exists('flower_classifier.h5'):
    print("🔄 Loading image model...")
    img_model = load_model('flower_classifier.h5')
    print("✅ Image model ready!")

# ── Routes ───────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/agenticai')
def agenticai():
    return render_template('agenticai.html')

@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')

@app.route('/imageclassifier')
def imageclassifier():
    return render_template('imageclassifier.html')

# ── API Endpoints ─────────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data       = request.json
        user_input = data.get('message', '')
        agent_type = data.get('agent', 'Education Agent')

        prompts = {
            'Education Agent': 'You are an expert teacher. Answer in simple clear educational language.',
            'Creative Agent':  'You are a creative writer. Respond with imagination and creativity.',
            'Technical Agent': 'You are a senior software engineer. Give technical answers with code examples.'
        }

        system_prompt = prompts.get(agent_type, prompts['Education Agent'])

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # free, fast, no limits
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_input}
            ]
        )

        return jsonify({'response': response.choices[0].message.content})

    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'})
@app.route('/api/sentiment', methods=['POST'])
def api_sentiment():
    data       = request.json
    user_input = data.get('text', '')
    cleaned    = preprocess(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = nlp_model.predict(vectorized)[0]
    confidence = nlp_model.predict_proba(vectorized).max() * 100
    return jsonify({
        'sentiment':   prediction.upper(),
        'confidence':  f"{confidence:.1f}"
    })

@app.route('/api/classify', methods=['POST'])
def api_classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    if img_model is None:
        return jsonify({'error': 'Image model not loaded. Run image_classifier.py first!'})

    file = request.files['image']
    file.save('temp_image.jpg')

    img = keras_image.load_img('temp_image.jpg', target_size=(224, 224))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = img_model.predict(img_array)
    class_idx   = np.argmax(predictions)
    confidence  = predictions[0][class_idx] * 100

    return jsonify({
        'class':      CLASSES[class_idx].upper(),
        'confidence': f"{confidence:.1f}"
    })

if __name__ == '__main__':
    app.run(debug=True)