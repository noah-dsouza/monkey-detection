# 🐒 Monkey Detection  
### Real-Time Facial Expression Recognition using Deep Learning & Computer Vision

This project is a real-time **facial expression recognition system** built from scratch using **PyTorch**, **OpenCV**, and **Python**.  
It classifies facial expressions captured from a webcam and displays a matching trigger image on screen — all happening live.

---

## 🧠 Core Concept
I trained a **Convolutional Neural Network (CNN)** to recognize facial expressions (happy, neutral, etc.) from images of my own face.  
Using **OpenCV**, I stream video frames from the webcam, preprocess them, feed them to the model, and visualize the prediction side-by-side with a trigger image that corresponds to the detected emotion.

---

## ⚙️ Tech Stack Breakdown

### 🧩 **Python 3.13**
The base language for the entire project — chosen for its deep learning ecosystem and extensive computer vision libraries.

### 🔥 **PyTorch**
Used to:
- Build a fully custom **CNN model** (no pre-trained weights).
- Handle **tensor operations**, **forward propagation**, and **softmax classification**.
- Train and evaluate the model efficiently using GPU acceleration (via Apple’s MPS backend).

### 🎥 **OpenCV**
Used to:
- Capture frames from the webcam in real-time.
- Display live video feeds and trigger images in a single window.
- Handle color space conversions and image resizing.
- Perform efficient I/O operations for dataset creation and inference.

### 🧠 **NumPy**
Used for:
- Frame concatenation (merging webcam feed and trigger images).
- Efficient numeric operations while handling real-time image arrays.

### 🖼️ **Pillow (PIL)**
Used to:
- Convert OpenCV frames into format compatible with PyTorch transforms.
- Resize and preprocess images before feeding them to the model.

### 🧰 **Torchvision**
Used for:
- `ImageFolder` — automatically managing labeled training data.
- Data transformations like `Resize()` and `ToTensor()`.
- Streamlining dataset management for the CNN.

### 🧪 **MediaPipe (optional in earlier versions)**
Initially explored for face tracking and landmark detection before switching fully to CNN-based classification.  
Learned how to preprocess and crop faces effectively for ML tasks.

---

## 🧠 What I Learned
- How to architect and train a **Convolutional Neural Network** from the ground up.  
- How to use **PyTorch** for model definition, training loops, and GPU acceleration.  
- How to integrate **OpenCV** with deep learning models for real-time applications.  
- How to automate **dataset collection** through webcam capture.  
- The importance of **balanced datasets** and diverse image samples.  
- How to optimize preprocessing pipelines for real-time performance.  
- How to debug and tune models for improved accuracy.

---

## 🚀 How to Run the Project

### 1️⃣ Create & Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
