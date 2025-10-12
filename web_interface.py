import os
import json
import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import onnxruntime as ort
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Available styles
STYLES = {
    'mosaic': 'Mosaic',
    'rain_princess': 'Rain Princess',
    'candy': 'Candy'
}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Preprocess and postprocess functions
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

postprocess = transforms.Compose([
    transforms.Lambda(lambda x: x.mul(1.0 / 255)),
    transforms.ToPILImage()
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('desx.html', styles=STYLES)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return json.dumps({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return json.dumps({'error': 'No file selected'}), 400

        if not file or not allowed_file(file.filename):
            return json.dumps({'error': 'Invalid file format. Please upload a PNG or JPG image.'}), 400

        # Get selected style
        style = request.form.get('style')
        if style not in STYLES:
            return json.dumps({'error': 'Invalid style selected'}), 400

        # Save the uploaded file
        filename = file.filename
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        # Process the image
        # Load the content image
        img = Image.open(input_path).convert("RGB")
        original_size = img.size
        img = img.resize((1024, 1024), Image.LANCZOS)
        content_tensor = preprocess(img).unsqueeze(0).cpu().numpy()

        # Load the ONNX model
        model_path = f"{style}.onnx"
        ort_session = ort.InferenceSession(model_path)

        # Run style transfer
        stylized = ort_session.run(None, {"input": content_tensor})[0]

        # Convert to image
        output_tensor = torch.from_numpy(stylized.squeeze())
        output_image = postprocess(output_tensor)

        # Resize to original size
        output_image = output_image.resize(original_size, Image.LANCZOS)

        # Apply sharpening and contrast enhancement
        sharp_img = output_image.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(sharp_img)
        final_img = enhancer.enhance(1.2)

        # Save the output image
        output_filename = f"output_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        final_img.save(output_path)

        # Return JSON response with image paths
        return json.dumps({
            'success': True,
            'input_image': filename,
            'output_image': output_filename,
            'style_name': STYLES[style]
        })

    except Exception as e:
        return json.dumps({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)