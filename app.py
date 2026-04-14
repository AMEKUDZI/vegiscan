from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import base64
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

IMG_SIZE = (224, 224)
MODEL_PATH = 'vegiscan_model.h5'

CLASS_NAMES = [
    'Bean', 'Bitter Gourd', 'Bottle Gourd', 'Brinjal', 'Broccoli',
    'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
    'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
]

VEG_EMOJIS = {
    'Bean': '🫘', 'Bitter Gourd': '🥒', 'Bottle Gourd': '🥒',
    'Brinjal': '🍆', 'Broccoli': '🥦', 'Cabbage': '🥬',
    'Capsicum': '🫑', 'Carrot': '🥕', 'Cauliflower': '🥦',
    'Cucumber': '🥒', 'Papaya': '🍈', 'Potato': '🥔',
    'Pumpkin': '🎃', 'Radish': '🌱', 'Tomato': '🍅'
}

def build_and_save_model():
    print("Building model with MobileNetV2 weights...")
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet'
    )
    base.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.save(MODEL_PATH)
    print("Model saved.")
    return model

# Load or build model
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = build_and_save_model()

print("Ready!")

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(IMG_SIZE)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img, dtype=np.float32))
    return np.expand_dims(arr, axis=0)

@app.route('/')
def index():
    return app.send_static_file('vegiscan.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        image_bytes = base64.b64decode(data.get('image', ''))
        tensor = preprocess(image_bytes)

        preds = model.predict(tensor, verbose=0)[0]
        top3_idx = preds.argsort()[-3:][::-1]

        top3 = [{'name': CLASS_NAMES[i], 'confidence': float(preds[i]), 'emoji': VEG_EMOJIS[CLASS_NAMES[i]]} for i in top3_idx]
        top = top3[0]
        confidence_score = round(top['confidence'] * 100, 1)
        confidence_label = 'High' if top['confidence'] > 0.7 else 'Medium' if top['confidence'] > 0.4 else 'Low'

        # Not sure if confidence is below 75%
        if top['confidence'] < 0.75:
            return jsonify({
                'not_sure': True,
                'confidence_score': confidence_score,
                'top3': top3,
                'message': "I'm not confident enough to identify this image. Please try a clearer photo of a vegetable."
            })

        return jsonify({
            'not_sure': False,
            'name': top['name'],
            'emoji': top['emoji'],
            'confidence': confidence_label,
            'confidence_score': confidence_score,
            'top3': top3
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=8080, debug=False)
