import torch
from neural_style.transformer_net import TransformerNet

# --- Configuration ---
MODEL_FILE = "mosaic.pth"           # path to your trained model
OUTPUT_ONNX_FILE = "mosaic.onnx"
IMAGE_SIZE = 1024

# --- Load model architecture ---
model = TransformerNet()

# --- Load checkpoint ---
loaded = torch.load(MODEL_FILE, map_location='cpu')
if isinstance(loaded, dict) and 'state_dict' in loaded:
    state_dict = loaded['state_dict']
else:
    state_dict = loaded

# --- Clean checkpoint (remove unsupported buffers) ---
keys_to_remove = [k for k in state_dict.keys() if "running_mean" in k or "running_var" in k]
for k in keys_to_remove:
    print(f"🧹 Removed {k} from state_dict")
    state_dict.pop(k)

# --- Fix ConvTranspose2d weights shape if needed ---
for name in ["deconv1.conv2d.weight", "deconv2.conv2d.weight"]:
    if name in state_dict:
        w = state_dict[name]
        expected_in_channels = getattr(model, name.split('.')[0]).conv2d.in_channels
        if w.shape[0] != expected_in_channels:
            state_dict[name] = w.transpose(0, 1)
            print(f"✅ Transposed weights for {name}")
    else:
        print(f"⚠️ Warning: {name} not found in checkpoint")

# --- Load weights into model ---
load_result = model.load_state_dict(state_dict, strict=False)
if load_result.missing_keys:
    print(f"⚠️ Missing keys when loading state_dict: {load_result.missing_keys}")
if load_result.unexpected_keys:
    print(f"⚠️ Unexpected keys in state_dict: {load_result.unexpected_keys}")

# --- Set model to evaluation mode ---
model.eval()

# --- Dummy input with fixed batch size ---
_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

# --- Export ONNX ---
print(f"🚀 Starting ONNX conversion for {MODEL_FILE} ...")
torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_ONNX_FILE,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None   # ✅ fixed shape to avoid InstanceNorm errors
)
print(f"✅ ONNX model saved as {OUTPUT_ONNX_FILE}")
