```markdown
# Qualcomm Canvas – Neural Style Transfer Web Application 🎨🧠

## Overview

**Qualcomm Canvas** is an AI-powered **Neural Style Transfer** web application developed as part of an academic–industry exposure initiative associated with **IIIT Hyderabad (IIITH)** and **Qualcomm**.  
The project demonstrates how deep learning models can be deployed in a web environment to apply artistic styles to images in real time.

Users can upload images or capture photos using a camera and transform them into artistic paintings using pre-trained neural networks.

---

## Project Objectives 🎯

- Implement Neural Style Transfer using deep learning  
- Integrate machine learning models with a Flask web application  
- Enable real-time image stylization  
- Build a responsive and user-friendly web interface  
- Demonstrate practical deployment of AI models  

---

## Features ✨

- Apply multiple artistic styles to images  
- Upload images from local storage  
- Capture images using device camera  
- Fast inference using ONNX Runtime  
- Download high-quality stylized images  
- Web-based interface accessible via browser  

---

## Neural Style Transfer 🖌️

Neural Style Transfer is a deep learning technique that combines:
- **Content** of one image  
- **Style** of another image  

This project uses pre-trained feed-forward neural networks, enabling fast style application without iterative optimization during runtime.

---

## Tech Stack 🧩

### Programming & Frameworks
- Python 3  
- Flask  

### Machine Learning & AI
- PyTorch  
- ONNX Runtime  
- Pre-trained Neural Style Transfer models  

### Frontend
- HTML5  
- CSS3  
- JavaScript  
- Bootstrap  

### Image Processing
- Pillow (PIL)  

---

## Project Structure 📁

```

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
├── uploads/                      # Uploaded images
├── outputs/                      # Stylized output images
│
├── *.onnx                        # ONNX style models
├── *.pth                         # PyTorch model weights
│
├── web_interface.py              # Flask application entry point
├── requirements.txt              # Python dependencies
└── README.md

````

---

## Installation and Setup ⚙️

### Prerequisites
- Python 3.9 or above  
- Git  
- Webcam (optional, for camera mode)  

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/hemannayak/Qualcomm-Canvas-Neural-Style-Transfer.git
   cd Qualcomm-Canvas-Neural-Style-Transfer
````

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
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

   ```
   http://localhost:5001
   ```

---

## Usage Instructions 🚀

### Image Upload Mode

1. Select an artistic style
2. Upload an image
3. Click **Apply Style**
4. Download the stylized output

### Camera Mode

1. Select a style
2. Enable camera
3. Capture image
4. Apply style and download result

---

## Model Details 🧠

* Uses pre-trained feed-forward neural style transfer networks
* Models converted to ONNX format for faster inference
* No training required during runtime

---

## Learning Outcomes 📚

* Practical understanding of Neural Style Transfer
* Experience deploying ML models using Flask
* Hands-on exposure to real-time AI inference
* Integration of frontend and backend for ML applications

---

## License 📄

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute this project with proper attribution.

---

## Acknowledgements 🙏

* IIIT Hyderabad (IIITH)
* Qualcomm Technologies
* Neural Style Transfer research community

---

## Author 👨‍💻

**Hemanth Nayak**
GitHub: [https://github.com/hemannayak](https://github.com/hemannayak)

---

⭐ If you find this project useful, consider starring the repository!

```
```
