from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load model and class labels
model = tf.keras.models.load_model("best_model.keras")
with open("class_indices.json", "r") as f:
    class_names = list(json.load(f).keys())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classification')
def classification():
    return render_template('classification.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('predict.html', prediction=None)

        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Preprocess image
            img = Image.open(filepath).convert('RGB')
            img = img.resize((300, 300))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            # Predict top-3
            preds = model.predict(img_array)[0]
            top_indices = preds.argsort()[-3:][::-1]
            top_3 = [(class_names[i], f'{preds[i] * 100:.2f}%') for i in top_indices]

            return render_template('predict.html',
                                   image_path=filepath,
                                   top_3=top_3)

    return render_template('predict.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
