# BoneScan AI - Bone Fracture Detection System

![Version](https://img.shields.io/badge/version-1.2.0-blue) ![Python](https://img.shields.io/badge/Python-3.8+-yellow) ![Flask](https://img.shields.io/badge/Backend-Flask-green) ![Status](https://img.shields.io/badge/Status-Beta-orange) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## About the Project
BoneScan AI is our hands-on project where we took a pre-trained YOLOv8 model and fine-tuned it ourselves on 8K+ real X-ray images to detect bone fractures. Doctors can upload an X-ray, and within seconds get bounding boxes around fractures, confidence scores, and helpful medical insights.

The system combines cutting-edge machine learning technology with a user-friendly interface, allowing doctors and radiologists to simply upload X-ray images and receive immediate AI-assisted analysis with confidence scores and medical recommendations.

## 📁 Project Structure
```bash

├── data/
│   ├── raw/                   # Train daata
│   │   └──images/            # Original X-ray images (~8k+)
│   │   └──labels/            # YOLO annotation files (.txt)
│   └── processed/             # Preprocessed images & augmentations
├── models/
│   ├── pretrained/            # Base YOLOv8 weights (yolov8n.pt)
│   └── trained/               # Fine-tuned fracture models
├── results/
│   ├── metrics/               # mAP, precision, recall plots/JSON
│   └── predictions/           # Sample detection outputs
├── static/                    # CSS, JS for web UI
├── uploads/                   # User-uploaded images (Flask)
├── .gitignore                 # Python/ML ignores
├── version.json               # App & model versions
├── README.md                  # This file
├── app.py                     # Flask web server
├── bone_fracture_detector.py  # Core YOLO detection
└── requirements.txt           # Dependencies

```
## 🛠️ Technologies Used
**Backend**: Python • Flask • **YOLOv8 (fine-tuned)** • PyTorch • OpenCV • Ultralytics  
**Frontend**: HTML5 • CSS3 • JavaScript • Font Awesome  
**Deployment**: Gunicorn • Render

## 🚀 Results & Performance 
![Results](results/metrics/results.png)

## 🚀 Confusion Metrics
![Metrics](results/metrics/confusion_matrix.png)

## 🚀 labels
![Metrics](results/metrics/labels.jpg)

## 🚀 labels_correlogram
![Results](results/metrics/labels_correlogram.jpg)


## ⚡ Quick Start
Clone the repository

bash
git clone https://github.com/nabakrishna/bone-fracture-detection.git
cd Bone-Fracture-Detection
Install dependencies

bash
pip install -r requirements.txt
Run the application

bash
python app.py
Open your browser

text
http://localhost:5000
🎯 How It Works
Upload an X-ray image through the web interface

AI model processes the image and detects potential fractures

Results display original image alongside annotated version with bounding boxes

Get detailed analysis with confidence scores and medical recommendations

📋 Features

✅ Real-time fracture detection

✅ Visual results with bounding boxes

✅ Confidence scoring

✅ Medical recommendations

✅ Responsive web design

✅ Privacy-focused (local processing)

### ⚠️ Medical Disclaimer
This tool is designed to assist healthcare professionals and should not replace professional medical diagnosis or treatment decisions.

### 📄 License
MIT License - feel free to use and modify for your projects.
