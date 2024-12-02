from flask import Flask, request, jsonify
import tensorflow as tf
from flask import render_template

model = tf.keras.models.load_model('model_path.keras')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('input')

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    response = model.predict([user_input])

    return jsonify({"response": response[0]})

if __name__ == '__main__':
    app.run(debug=True)
