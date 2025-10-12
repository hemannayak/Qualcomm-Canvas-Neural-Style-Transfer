import torch
from neural_style.transformer_net import TransformerNet
import os

# Configuration for all styles
STYLES = [
    {'name': 'Mosaic', 'model': 'mosaic.pth', 'output': 'mosaic.onnx'},
    {'name': 'udnie', 'model': 'udnie.pth', 'output': 'udnie.onnx'},  # You'll need this file
    {'name': 'Rain Princess', 'model': 'rain_princess.pth', 'output': 'rain_princess.onnx'},  # You'll need this file
    {'name': 'Candy', 'model': 'candy.pth', 'output': 'candy.onnx'}  # You'll need this file
]

IMAGE_SIZE = 1024

def convert_model(model_file, output_file):
    print(f"\n🔄 Converting {model_file} to {output_file}...")
    
    # Load model architecture
    model = TransformerNet()
    
    # Load checkpoint
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        return False
    
    loaded = torch.load(model_file, map_location='cpu')
    if isinstance(loaded, dict) and 'state_dict' in loaded:
        state_dict = loaded['state_dict']
    else:
        state_dict = loaded
    
    # Clean checkpoint (remove unsupported buffers)
    keys_to_remove = [k for k in state_dict.keys() if "running_mean" in k or "running_var" in k]
    for k in keys_to_remove:
        print(f"🧹 Removed {k} from state_dict")
        state_dict.pop(k)
    
    # Fix ConvTranspose2d weights shape if needed
    for name in ["deconv1.conv2d.weight", "deconv2.conv2d.weight"]:
        if name in state_dict:
            w = state_dict[name]
            expected_in_channels = getattr(model, name.split('.')[0]).conv2d.in_channels
            if w.shape[0] != expected_in_channels:
                state_dict[name] = w.transpose(0, 1)
                print(f"✅ Transposed weights for {name}")
        else:
            print(f"⚠️ Warning: {name} not found in checkpoint")
    
    # Load weights into model
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"⚠️ Missing keys when loading state_dict: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"⚠️ Unexpected keys in state_dict: {load_result.unexpected_keys}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Dummy input with fixed batch size
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    
    # Export ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_file,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None   # Fixed shape to avoid InstanceNorm errors
        )
        print(f"✅ ONNX model saved as {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error converting {model_file}: {str(e)}")
        return False

def main():
    print("="*60)
    print("NEURAL STYLE TRANSFER - BATCH MODEL CONVERSION")
    print("="*60)
    
    success_count = 0
    total_count = len(STYLES)
    
    for style in STYLES:
        print(f"\n🎨 Processing style: {style['name']}")
        if convert_model(style['model'], style['output']):
            success_count += 1
    
    print("\n" + "="*60)
    print(f"CONVERSION COMPLETE: {success_count}/{total_count} models converted successfully")
    print("="*60)

if __name__ == "__main__":
    main()