# IIITH Megathon 2025 -Qualcomm

# 🎨 Qualcomm Canvas - AI Neural Style Transfer Web Application

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-2.3+-green.svg)](https://flask.palletsprojects.com/)

## 🌟 Overview

**Qualcomm Canvas** is a cutting-edge web application that transforms images using AI-powered neural style transfer. Built during the IIITH-Qualcomm collaboration, this application combines modern web technologies with advanced machine learning to create stunning artistic transformations.

## ✨ Features

### 🎭 **Neural Style Transfer**
- **Multiple Artistic Styles**: Mosaic, Rain Princess, Candy
- **Real-time Processing**: Fast ONNX model inference
- **High-Quality Output**: Professional-grade image transformation

### 📸 **Camera Integration**
- **Live Camera Preview**: Real-time camera feed
- **One-Click Capture**: Instant photo capture and processing
- **Auto-Style Application**: Automatically applies selected style to captured images

### 🎨 **Modern UI/UX**
- **Glassmorphism Design**: Beautiful gradient backgrounds with glass effects
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile
- **Interactive Elements**: Smooth animations and hover effects
- **Intuitive Controls**: Drag & drop file upload and camera controls

### 📥 **Flexible Downloads**
- **Dual Format Support**: Download as JPG or PNG
- **Smart Naming**: Automatic file naming with timestamps
- **High Resolution**: Maintains original image quality

## 🚀 Technical Stack

### **Backend**
- **Flask**: Python web framework
- **ONNX Runtime**: Fast neural network inference
- **PyTorch**: Deep learning model processing
- **Pillow (PIL)**: Image manipulation and processing

### **Frontend**
- **HTML5**: Modern semantic markup
- **CSS3**: Advanced styling with gradients and animations
- **JavaScript ES6+**: Modern async/await patterns
- **Bootstrap 5**: Responsive design framework

### **AI/ML Models**
- **Neural Style Transfer**: Pre-trained ONNX models
- **Real-time Inference**: Optimized for web deployment
- **Multiple Styles**: Professional artistic transformations

## 🛠️ Installation & Setup

### **Prerequisites**
- Python 3.10 or higher
- Modern web browser with camera support
- Git (for cloning the repository)

### **Quick Start**

1. **Clone the Repository**
```bash
git clone https://github.com/hemannayak/IIITH-Qualcomm.git
cd IIITH-Qualcomm
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Application**
```bash
python web_interface.py
```

4. **Open in Browser**
```
http://localhost:5001
```

## 📱 Usage Guide

### **Desktop Usage**
1. **Select Style**: Choose from Mosaic, Rain Princess, or Candy
2. **Upload Image**: Drag & drop or browse for files
3. **Process**: Click "Apply Artistic Style"
4. **Download**: Choose JPG or PNG format

### **Camera Usage**
1. **Select Style**: Choose desired artistic style first
2. **Use Camera**: Click "Use Camera" to access camera feed
3. **Capture**: Click "Capture & Create Art" for instant processing
4. **Download**: Choose format and download your artwork

## 🎯 Project Structure

```
IIITH-Qualcomm/
├── neural_style/           # Core neural network modules
│   ├── transformer_net.py  # Neural style transfer model
│   ├── stylize_image.py    # Image processing utilities
│   └── utils.py           # Helper functions
├── templates/             # HTML templates
│   └── desx.html         # Main application interface
├── outputs/              # Processed images output directory
├── uploads/              # User uploaded images
├── *.onnx               # ONNX model files for each style
├── *.pth                # PyTorch model weights
├── web_interface.py     # Flask web application
├── convert_to_onnx.py   # Model conversion utilities
└── requirements.txt     # Python dependencies
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

### **Development Setup**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IIITH (International Institute of Information Technology, Hyderabad)**
- **Qualcomm** - For the collaboration and support
- **Neural Style Transfer Research Community**
- **Open Source Contributors**

## 📞 Contact

**Project Maintainer**: Hemanth Nayak
- Email: hemannayakpangoth@gmail.com
- GitHub: [@hemannayak](https://github.com/hemannayak)

---

**⭐ If you find this project helpful, please give it a star!**
