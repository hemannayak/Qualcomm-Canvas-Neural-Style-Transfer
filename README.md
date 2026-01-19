# Qualcomm Canvas – Neural Style Transfer Web Application 🎨🧠

## Overview

> **Qualcomm Canvas** is an AI-powered **Neural Style Transfer** web application developed as part of an academic–industry exposure initiative associated with **IIIT Hyderabad (IIITH)** and **Qualcomm**.

The project demonstrates how deep learning models can be deployed in a web environment to apply artistic styles to images in real time. Users can upload images or capture photos using a camera and transform them into artistic paintings using pre-trained neural networks.

---

## Project Objectives 🎯

* **Deep Learning Implementation:** Practical application of Neural Style Transfer.
* **Web Integration:** Connecting machine learning models with a Flask-based backend.
* **Efficiency:** Enabling real-time image stylization.
* **User Experience:** Building a responsive and intuitive web interface.
* **Deployment:** Demonstrating how AI models move from development to production.

---

## Neural Style Transfer 🖌️

Neural Style Transfer (NST) is a technique that blends two images: a **Content** image and a **Style** image (like a famous painting).

This project utilizes pre-trained **feed-forward neural networks**. Unlike the original optimization-based NST, this method allows for fast style application without needing iterative updates during runtime. The core objective is to minimize a loss function defined as:

Where  and  are weights representing the emphasis on content and style, respectively.

---

## Features ✨

* **Multiple Styles:** Choose from various artistic filters.
* **Flexible Input:** Upload from local storage or use a live **device camera**.
* **High Performance:** Fast inference powered by **ONNX Runtime**.
* **Easy Export:** Download stylized images instantly.
* **Web-First:** Fully accessible via any modern browser.

---

## Tech Stack 🧩

| Category | Tools |
| --- | --- |
| **Backend** | Python 3, Flask |
| **AI/ML** | PyTorch, ONNX Runtime |
| **Frontend** | HTML5, CSS3, JavaScript, Bootstrap |
| **Image Processing** | Pillow (PIL) |

---

## Project Structure 📁

```bash
Qualcomm-Canvas-Neural-Style-Transfer/
│
├── neural_style/
│   ├── transformer_net.py        # Neural network architecture
│   ├── stylize_image.py          # Style transfer logic
│   └── utils.py                  # Utility functions
│
├── templates/
│   └── desx.html                 # Frontend UI template
│
├── uploads/                      # Temporary storage for uploads
├── outputs/                      # Storage for processed images
│
├── *.onnx                        # Optimized ONNX style models
├── *.pth                         # PyTorch model weights
│
├── web_interface.py              # Flask application entry point
├── requirements.txt              # Dependency list
└── README.md                     # Documentation

```

---

## Installation and Setup ⚙️

### Prerequisites

* Python 3.9+
* Git
* Webcam (optional)

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/hemannayak/Qualcomm-Canvas-Neural-Style-Transfer.git
cd Qualcomm-Canvas-Neural-Style-Transfer

```


2. **Create a virtual environment**
```bash
python -m venv venv
# For Windows:
venv\Scripts\activate
# For Mac/Linux:
source venv/bin/activate

```


3. **Install dependencies**
```bash
pip install -r requirements.txt

```


4. **Run the application**
```bash
python web_interface.py

```


5. **Open in browser**
Navigate to: `http://localhost:5001`

---

## Usage Instructions 🚀

### Option A: Image Upload

1. Select an artistic style from the sidebar/menu.
2. Upload a photo from your computer.
3. Click **Apply Style**.
4. View and download your stylized artwork.

### Option B: Camera Mode

1. Select your preferred style.
2. Grant camera permissions and capture a photo.
3. Click **Apply Style** to process the capture.

---

## Learning Outcomes 📚

* Practical understanding of **Computer Vision** and NST.
* Experience in **Model Quantization** and conversion to ONNX format.
* Hands-on experience with **Full-stack AI deployment**.

---

## License 📄

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this project with proper attribution.

## Acknowledgements 🙏

* **IIIT Hyderabad (IIITH)** for project guidance.
* **Qualcomm Technologies** for industry exposure.
* The open-source **Neural Style Transfer** research community.

---

## Author 👨‍💻

**Hemanth Nayak**

* GitHub: [@hemannayak](https://github.com/hemannayak)

> ⭐ If you find this project useful, consider starring the repository!

---
