import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import onnxruntime as ort
import numpy as np
import os
import sys

# -----------------------------
# CONFIG
# -----------------------------
# Available styles with their corresponding ONNX model files
STYLES = {
    '1': {'name': 'Mosaic', 'model': 'mosaic.onnx'},
    '2': {'name': 'Ghibli', 'model': 'ghibli_style.onnx'},
    '3': {'name': 'udnie', 'model': 'udnie.onnx'},      # You'll need to add this file
    '4': {'name': 'Rain Princess', 'model': 'rain_princess.onnx'},  # You'll need to add this file
    '5': {'name': 'Candy', 'model': 'candy.onnx'}       # You'll need to add this file
}

# Default paths
content_image_path = "test_image.jpg"
output_image_path = "stylized_output.jpg"
final_output_path = "final_output.jpg"
blended_output_path = "blended_output.jpg"
imsize = 1024  # Increase to 2048 if your system can handle it
style_alpha = 0.8  # Blend intensity (0.0 = no style, 1.0 = full style)

# -----------------------------
# Style Selection Menu
# -----------------------------
def select_style():
    print("\n" + "="*50)
    print("NEURAL STYLE TRANSFER - STYLE SELECTION")
    print("="*50)
    print("\nAvailable Styles:")
    for key, style in STYLES.items():
        print(f"{key}. {style['name']}")
    
    while True:
        choice = input("\nSelect a style (1-5): ")
        if choice in STYLES:
            return STYLES[choice]['model']
        print("Invalid choice. Please try again.")

# -----------------------------
# Load content image
# -----------------------------
def load_image(filename, size=None):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ File not found: {filename}")
    img = Image.open(filename).convert("RGB")
    if size:
        img = img.resize((size, size), Image.LANCZOS)
    return img

# -----------------------------
# Preprocessing & Postprocessing
# -----------------------------
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

postprocess = transforms.Compose([
    transforms.Lambda(lambda x: x.mul(1.0 / 255)),
    transforms.ToPILImage()
])

# -----------------------------
# Main function
# -----------------------------
def main():
    # Select style
    onnx_model_path = select_style()
    print(f"\n✅ Selected style: {onnx_model_path}")
    
    # Check if model exists
    if not os.path.exists(onnx_model_path):
        print(f"❌ Model not found: {onnx_model_path}")
        print("Please make sure the model file is in the current directory.")
        return
    
    try:
        # Load content image
        print(f"\n📷 Loading content image: {content_image_path}")
        content_image = load_image(content_image_path, imsize)
        content_tensor = preprocess(content_image).unsqueeze(0).cpu().numpy()

        # Load ONNX model
        print(f"🧠 Loading ONNX model: {onnx_model_path}")
        ort_session = ort.InferenceSession(onnx_model_path)

        # Perform inference
        print("🎨 Applying style transfer...")
        stylized = ort_session.run(None, {"input": content_tensor})[0]

        # Convert output to PIL image
        output_tensor = torch.from_numpy(stylized.squeeze())
        output_image = postprocess(output_tensor)
        output_image.save(output_image_path)
        print(f"✅ Saved raw stylized image to {output_image_path}")

        # Optional sharpening & contrast enhancement
        print("🔧 Applying sharpening and contrast enhancement...")
        sharp_img = output_image.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(sharp_img)
        final_img = enhancer.enhance(1.2)
        final_img.save(final_output_path)
        print(f"✨ Final enhanced image saved as {final_output_path}")

        # Optional blending with original for style intensity
        print("🎨 Creating blended image...")
        content_img = load_image(content_image_path, imsize)
        blended_img = Image.blend(content_img, final_img, alpha=style_alpha)
        blended_img.save(blended_output_path)
        print(f"🎭 Blended style output saved as {blended_output_path}")
        
        print("\n" + "="*50)
        print("STYLE TRANSFER COMPLETE!")
        print("="*50)

    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()