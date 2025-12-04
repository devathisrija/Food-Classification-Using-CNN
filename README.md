# 🍽️ Food Classification Using CNN

A deep learning–based web application for real-time food image classification and nutrition analysis using CNN, VGG16, and ResNet models. Users can upload images, select a model, view predicted food class, analyze nutritional values, and download prediction results in JSON format.

---

## 📌 Project Overview

With the rise of fitness tracking and personalized nutrition, automated food recognition has become essential. This project provides a Flask-based web application that allows users to upload food images in JPG/PNG format, classify them using deep learning models, and display detailed nutritional information such as calories, protein, fat, and carbohydrates.

The system supports multiple models (Custom CNN, VGG16, and ResNet) for comparative performance analysis and generates accuracy scores and classification reports.

---

## 🚀 Features

- ✅ Upload food images (JPG, PNG, JPEG)
- ✅ Live image preview
- ✅ Multiple model selection (CNN, VGG16, ResNet)
- ✅ Real-time food classification
- ✅ Nutritional information display using JSON
- ✅ Model accuracy & classification report
- ✅ Save and download prediction results as JSON
- ✅ User-friendly and responsive web interface
- ✅ Robust error handling

---

## 🧠 Models Used

- Custom CNN – Built from scratch  
- VGG16 – Transfer learning model  
- ResNet50 – Deep residual network  

---

## 🛠️ Technologies Used

### Backend
- Python 3.10+
- Flask
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Pickle
- JSON

### Frontend
- HTML5
- CSS3
- JavaScript

### Tools
- Visual Studio Code
- Git & GitHub

---

## ⚙️ System Requirements

### Minimum
- RAM: 4 GB  
- Storage: 2 GB  
- Processor: Dual-core  
- OS: Windows / Linux / macOS  

### Recommended
- RAM: 8 GB+
- GPU: NVIDIA (for training)
- Storage: 10 GB+

---

## 🏗️ Project Architecture

User → Flask Web App → Image Preprocessing → Model Prediction → JSON Mapping → Result Display

---

## 🔄 Workflow

1. Upload food image  
2. Select model (CNN / VGG16 / ResNet)  
3. Image preprocessing  
4. Model prediction  
5. Class mapping  
6. Nutritional data retrieval  
7. Result display with accuracy & JSON download  

---

## 📂 Project Structure

```
Food-Classification-Using-CNN/
│
├── app.py                     # Main Flask backend
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
│
├── models/                    # Trained deep learning models
│   ├── cnn.pkl
│   ├── vgg16.pkl
│   └── resnet.pkl
│
├── data/                      # Nutrition JSON files
│   ├── burger.json
│   ├── pizza.json
│   ├── samosa.json
│   ├── idli.json
│   └── dosa.json
│
├── static/                    # Static assets
│   ├── uploads/               # Uploaded input images
│   └── outputs/               # Prediction results
│
├── templates/                 # HTML templates
│   ├── index.html             # Home page
│   └── result.html            # Prediction result page
│
└── screenshots/               # Application screenshots (optional)
    ├── home.png
    ├── upload.png
    └── result.png
```
