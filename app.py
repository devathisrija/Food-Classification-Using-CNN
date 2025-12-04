from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import pickle
import numpy as np
import json
import cv2

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace-this-with-a-secret'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

labels_ = [
    'apple_pie','Baked_Potato','burger','butter_naan','chai',
    'chapati','cheesecake','chicken_curry','chole_bhature',
    'Crispy_Chicken','dal_makhani','dhokla','Donut',
    'fried_rice','Fries','Hotdog','ice_cream','idli','jalebi','kaathi_rolls',
    'kadai_paneer','kulfi','masala_dosa','momos','omelette','paani_puri',
    'pakode','pav_bhaji','pizza','samosa','Sandwich','sushi','Taco','Taquito'
]

# -------------------------
# Utility functions
# -------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(choice):
    model_map = {
        'cnn': 'models/cnn.pkl',
        'vgg16': 'models/vgg16.pkl',
        'resnet': 'models/resnet.pkl'
    }
    path = model_map.get(choice.lower())
    if not path or not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_validation_metrics(choice):
    metrics_path = f'models/{choice.lower()}_validation_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def preprocess_image(image_path, size=(224,224)):
    img = cv2.imread(image_path, 1)
    if img is None:
        raise ValueError(f"Failed to read image at {image_path}")
    img = cv2.resize(img, size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def save_prediction_json(class_name, json_content, metrics):
    result = {
        "predicted_class": class_name,
        "food_details": json_content,
        "metrics": metrics
    }
    json_path = os.path.join(app.config['OUTPUT_FOLDER'], "prediction.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, default=str)
    return json_path

# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    return render_template('index.html', uploaded_image=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['image']
    image_name = request.form.get('image_name', '')
    provided_class = request.form.get('provided_class', '').strip()

    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if image_name:
            name, ext = os.path.splitext(filename)
            filename = secure_filename(image_name) + ext
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        return render_template('index.html', uploaded_image=save_path.replace('\\','/'), provided_class=provided_class)

    flash('Allowed image types are png, jpg, jpeg')
    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "Error: JSON file not exist", 404

@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form.get('model_choice')
    uploaded_image = request.form.get('uploaded_image')
    provided_class = request.form.get('provided_class', '').strip()

    if not model_choice or not uploaded_image:
        return jsonify({'error': 'Missing model_choice or uploaded_image'}), 400

    if uploaded_image.startswith('/'):
        uploaded_image = uploaded_image[1:]

    if not os.path.exists(uploaded_image):
        return jsonify({'error': 'Uploaded image not found on server'}), 400

    # Load model
    model = load_model(model_choice)
    if model is None:
        return jsonify({'error': f'Model for {model_choice} not found on server'}), 500

    # Load precomputed validation metrics
    validation_metrics = load_validation_metrics(model_choice)

    # Predict uploaded image
    img_size = (256, 256) if model_choice.lower() == 'cnn' else (224, 224)
    arr = preprocess_image(uploaded_image, img_size)
    try:
        preds = model.predict(arr)
        idx = int(np.argmax(preds, axis=1)[0] if preds.ndim == 2 else np.argmax(preds))
        class_name = labels_[idx] if idx < len(labels_) else str(idx)
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'exception': str(e)}), 500

    # Load JSON details of predicted class
    json_path = os.path.join('data', f'{class_name}.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as jf:
            json_content = json.load(jf)
    else:
        json_content = {'error': f'No json file found for {class_name}', 'path': json_path}

    # Save prediction + metrics JSON
    prediction_json_path = save_prediction_json(class_name, json_content, validation_metrics)

    # Prepare metrics for predicted class only
    predicted_metrics = None
    if validation_metrics:
        report = validation_metrics.get('classification_report', {}).get(class_name, {})
        cm = validation_metrics.get('confusion_matrix', [])
        if cm and class_name in labels_:
            class_idx = labels_.index(class_name)
            TP = cm[class_idx][class_idx]
            FP = sum(row[class_idx] for row in cm) - TP
            FN = sum(cm[class_idx]) - TP
            TN = sum(sum(row) for row in cm) - TP - FP - FN
        else:
            TP = TN = FP = FN = 0

        predicted_metrics = {
            'accuracy': validation_metrics.get('accuracy', 0),
            'precision': report.get('precision', 0),
            'recall': report.get('recall', 0),
            'f1_score': report.get('f1-score', 0),
            'support': report.get('support', 0),
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN
        }

    # Pass all data to result.html
    result = {
        'predicted_class': class_name,
        'provided_class': provided_class or None,
        'json_content': json_content,
        'uploaded_image': '/' + uploaded_image.replace('\\','/'),
        'prediction_json_path': '/' + prediction_json_path.replace('\\','/'),
        'predicted_metrics': predicted_metrics
    }

    return render_template('result.html', labels_=labels_, **result)

if __name__ == '__main__':
    app.run(debug=True)
