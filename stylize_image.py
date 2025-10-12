import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import onnxruntime as ort
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
# Paths (update if your files are elsewhere)
onnx_model_path = "mosaic.onnx"       # your ONNX model for Mosaic
content_image_path = "test_image.jpg" # your input image
output_image_path = "stylized_output.jpg"
final_output_path = "final_output.jpg"
blended_output_path = "blended_mosaic_output.jpg"

# Image resolution
imsize = 1024  # Increase to 2048 if your system can handle it

# Blend intensity (0.0 = no style, 1.0 = full style)
style_alpha = 0.8

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
# Run style transfer
# -----------------------------
content_image = load_image(content_image_path, imsize)
content_tensor = preprocess(content_image).unsqueeze(0).cpu().numpy()

# Load ONNX model
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"❌ ONNX model not found: {onnx_model_path}")
ort_session = ort.InferenceSession(onnx_model_path)

# Perform inference
stylized = ort_session.run(None, {"input": content_tensor})[0]

# Convert output to PIL image
output_tensor = torch.from_numpy(stylized.squeeze())
output_image = postprocess(output_tensor)
output_image.save(output_image_path)
print(f"✅ Saved raw stylized image to {output_image_path}")

# -----------------------------
# Optional sharpening & contrast enhancement
# -----------------------------
# Sharpen
sharp_img = output_image.filter(ImageFilter.SHARPEN)
# Enhance contrast
enhancer = ImageEnhance.Contrast(sharp_img)
final_img = enhancer.enhance(1.2)
final_img.save(final_output_path)
print(f"✨ Final sharpened and enhanced image saved as {final_output_path}")

# -----------------------------
# Optional blending with original for style intensity
# -----------------------------
content_img = load_image(content_image_path, imsize)
blended_img = Image.blend(content_img, final_img, alpha=style_alpha)
blended_img.save(blended_output_path)
print(f"🎨 Blended Mosaic style output saved as {blended_output_path}")
