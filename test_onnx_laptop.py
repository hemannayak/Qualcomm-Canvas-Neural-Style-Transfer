import cv2
import numpy as np
import onnxruntime as ort

# --- Configuration ---
ONNX_MODEL_PATH = "ghibli_style.onnx"
IMAGE_TO_TEST_PATH = "test_image.jpg"
OUTPUT_IMAGE_PATH = "stylized_output.jpg"
IMAGE_SIZE = 1024

def preprocess(image):
    """Prepares the image for the ONNX model."""
    # Resize the image to the size the model expects
    img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert from BGR (OpenCV's default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Reshape from (Height, Width, Channels) to (Channels, Height, Width)
    img = img.transpose(2, 0, 1)
    
    # Add a batch dimension and convert to float32
    img = np.expand_dims(img, axis=0).astype(np.float32)
    
    return img

def postprocess(model_output):
    """Converts the model's output back into a viewable image."""
    # Remove the batch dimension
    result = np.squeeze(model_output, axis=0)
    
    # Clip values to the valid range [0, 255]
    result = np.clip(result, 0, 255)
    
    # Reshape from (Channels, Height, Width) back to (Height, Width, Channels)
    result = result.transpose(1, 2, 0)
    
    # Convert from RGB back to BGR for OpenCV
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    # Convert to integer type for saving
    return result.astype(np.uint8)

# --- Main Execution ---
print("🤖 Loading the ONNX model...")
session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"🎨 Loading and processing '{IMAGE_TO_TEST_PATH}'...")
image = cv2.imread(IMAGE_TO_TEST_PATH)

if image is None:
    print(f"❌ Error: Could not load the image. Make sure 'test_image.jpg' is in the folder.")
else:
    # 1. Prepare the image
    input_tensor = preprocess(image)
    
    # 2. Run the model
    result_tensor = session.run([output_name], {input_name: input_tensor})[0]
    
    # 3. Convert the output back to an image
    final_image = postprocess(result_tensor)
    
    # 4. Save the result
    cv2.imwrite(OUTPUT_IMAGE_PATH, final_image)
    print(f"✅ Success! Stylized image saved as '{OUTPUT_IMAGE_PATH}'")