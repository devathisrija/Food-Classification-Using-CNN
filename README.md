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

## 🧩 Installation & Setup

### ✅ Prerequisites
- Python 3.8 or above  
- pip  
- Git  
- Virtual environment (optional)

### 📥 Step 1: Clone the Repository
git clone https://github.com/devathisrija/Food-Classification-Using-CNN.git

cd Food-Classification-Using-CNN


### 📦 Step 2: Create Virtual Environment (Optional)
python -m venv venv

Windows:
venv\Scripts\activate


### 📚 Step 3: Install Dependencies
pip install -r requirements.txt


### ▶️ Step 4: Run the Application
python app.py


### 🌐 Step 5: Access the Application
http://127.0.0.1:5000/


✅ Your Food Classification system is now successfully running!

---

## 🔄 Project Workflow

1. User uploads a food image  
2. Image preprocessing is performed  
3. Selected model predicts the food class  
4. Nutrition details are fetched using JSON  
5. Result is displayed on the web page  
6. User can download the prediction report  

---

## ✅ Testing & Validation

- Image format validation  
- Model prediction validation  
- Nutrition data verification  
- Frontend responsiveness testing  
- Error handling for missing files and models  

---

## 🎯 Key Achievements

- Built a complete AI-powered food recognition system  
- Implemented multiple deep learning models  
- Integrated nutrition analysis using JSON  
- Designed an interactive and responsive UI  
- Achieved high accuracy using transfer learning  

---

## 🔮 Future Enhancements

- Real-time webcam-based detection  
- Mobile application version  
- Multi-food detection  
- Portion size estimation  
- Nutrition API integration  
- Database integration  
- Multi-language support  

---

## 🖼️ Screenshots

### 🏠 Home Page

## 11. Screenshots

This section provides visual references of **Food Classification Using CNN** application.  

**Upload Image**  
![Upload Image](https://github.com/devathisrija/Food-Classification-Using-CNN/blob/main/screenshots/Screenshot%20(710).png)

![Upload Image](https://github.com/devathisrija/Food-Classification-Using-CNN/blob/main/screenshots/Screenshot%20(711).png)


**Uploaded Image preview**  
![preview](https://github.com/devathisrija/Food-Classification-Using-CNN/blob/main/screenshots/Screenshot%20(712).png)

**Model Selection**  
![Model Selection](https://github.com/devathisrija/Food-Classification-Using-CNN/blob/main/screenshots/Screenshot%20(714).png)

**Result Visualization**  
![Predicted class](https://github.com/devathisrija/Food-Classification-Using-CNN/blob/main/screenshots/Screenshot%20(724).png)

**Predicted class details**  
![JSON file](https://github.com/devathisrija/Food-Classification-Using-CNN/blob/main/screenshots/Screenshot%20(722).png)

**Selected Model metrics**  
![Metrics](https://github.com/devathisrija/Food-Classification-Using-CNN/blob/main/screenshots/Screenshot%20(723).png)  


---

## 👩‍💻 Developer

**Devathi Srija**  
AI & Machine Learning Developer  

---

## 📚 References

- TensorFlow Documentation  
- Keras Documentation  
- OpenCV Documentation  
- Flask Official Website  
- Kaggle Food Datasets  

---

## ⭐ If you like this project, don’t forget to give it a star on GitHub!

