from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('C:/Users/Bhindu Suma/Downloads/mlproject/sentiment_moodel.keras')

# Parameters
vocab_size = 10000
max_length = 200
word_index = imdb.get_word_index()

def preprocess_text(text):
    words = text.lower().split()
    sequences = [[word_index.get(word, 0) for word in words]]
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    return padded_sequences

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text data provided'}), 400

    text = data['text']
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)
    sentiment = 'positive' if prediction[0][0] >= 0.5 else 'negative'

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)


