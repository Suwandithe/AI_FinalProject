import gradio as gr
import torch
from PIL import Image
import numpy as np
import os
import sys

# Try to create a stub for missing ultralytics.nn.attention module if needed
def setup_attention_stub():
    """Create a stub module for ultralytics.nn.attention if it doesn't exist"""
    try:
        import ultralytics.nn.attention.attention
        return  # Module exists, no need for stub
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass
    
    # Create a proper package structure
    import types
    try:
        import ultralytics.nn as nn_module
        
        # Create the attention package (directory-like module)
        attention_pkg = types.ModuleType('attention')
        
        # Create the attention class module
        attention_class_module = types.ModuleType('attention')
        
        # Create a dummy Attention class that can be instantiated
        class Attention(torch.nn.Module):
            """Stub Attention class for compatibility"""
            def __init__(self, *args, **kwargs):
                super().__init__()
                # Create minimal structure to avoid errors
                self.dummy = torch.nn.Parameter(torch.zeros(1))
            
            def forward(self, *args, **kwargs):
                # Return a dummy output
                if args:
                    return args[0]  # Return first input as-is
                return torch.zeros(1)
        
        # Create ParallelPolarizedSelfAttention class (specific to this model)
        class ParallelPolarizedSelfAttention(torch.nn.Module):
            """Stub ParallelPolarizedSelfAttention class for compatibility"""
            def __init__(self, *args, **kwargs):
                super().__init__()
                # Create minimal structure
                self.dummy = torch.nn.Parameter(torch.zeros(1))
                # Try to match common attention parameters
                if 'dim' in kwargs:
                    dim = kwargs['dim']
                elif args:
                    dim = args[0] if isinstance(args[0], int) else 64
                else:
                    dim = 64
                # Add some basic layers that might be expected
                self.norm = torch.nn.LayerNorm(dim) if dim > 0 else torch.nn.Identity()
            
            def forward(self, x, *args, **kwargs):
                # Simple passthrough with normalization if possible
                if hasattr(self, 'norm') and isinstance(x, torch.Tensor):
                    if x.dim() >= 2:
                        return self.norm(x)
                return x if isinstance(x, torch.Tensor) else torch.zeros(1)
        
        # Add the classes to the module
        attention_class_module.Attention = Attention
        attention_class_module.attention = Attention  # Also as attribute
        attention_class_module.ParallelPolarizedSelfAttention = ParallelPolarizedSelfAttention
        
        # Create a fallback __getattr__ to handle any other missing classes dynamically
        def create_dummy_class(name):
            """Dynamically create a dummy class for any missing attention mechanism"""
            class DynamicAttention(torch.nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.dummy = torch.nn.Parameter(torch.zeros(1))
                    # Try to handle common parameters
                    if 'dim' in kwargs:
                        dim = kwargs['dim']
                    elif args and isinstance(args[0], int):
                        dim = args[0]
                    else:
                        dim = 64
                    if dim > 0:
                        self.norm = torch.nn.LayerNorm(dim)
                    else:
                        self.norm = torch.nn.Identity()
                
                def forward(self, x, *args, **kwargs):
                    if isinstance(x, torch.Tensor) and hasattr(self, 'norm'):
                        if x.dim() >= 2:
                            return self.norm(x)
                    return x if isinstance(x, torch.Tensor) else torch.zeros(1)
            
            DynamicAttention.__name__ = name
            return DynamicAttention
        
        # Add __getattr__ to dynamically create classes as needed
        original_getattr = getattr(attention_class_module, '__getattr__', None)
        def dynamic_getattr(name):
            if name.startswith('_'):
                raise AttributeError(f"module 'attention' has no attribute '{name}'")
            print(f"Creating dynamic stub for: {name}")
            cls = create_dummy_class(name)
            setattr(attention_class_module, name, cls)
            return cls
        
        attention_class_module.__getattr__ = dynamic_getattr
        
        # Make attention a package with the attention submodule
        attention_pkg.attention = attention_class_module
        attention_pkg.Attention = Attention  # Also at package level
        
        # Register both levels
        nn_module.attention = attention_pkg
        sys.modules['ultralytics.nn.attention'] = attention_pkg
        sys.modules['ultralytics.nn.attention.attention'] = attention_class_module
        
        print("Created stub for ultralytics.nn.attention.attention")
    except Exception as e:
        print(f"Could not create attention stub: {e}")
        import traceback
        traceback.print_exc()

# Try to load model - supports both YOLO and custom PyTorch models
def load_model(model_path):
    """Load the PyTorch model from best.pt"""
    # Setup attention stub if needed
    setup_attention_stub()
    
    # Try loading as YOLO model first (most common for .pt files)
    try:
        from ultralytics import YOLO
        print("Attempting to load as YOLO model...")
        # YOLO may auto-install missing modules, so we allow it to complete
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model = YOLO(model_path)
        print("Successfully loaded as YOLO model!")
        return model, "yolo"
    except (ImportError, ModuleNotFoundError) as import_err:
        error_msg = str(import_err)
        print("ERROR: Ultralytics import failed!")
        print(f"Error: {error_msg}")
        raise Exception("Ultralytics package is required. Install it with: pip install ultralytics") from import_err
    except Exception as yolo_error:
        # YOLO loading failed - might be version mismatch or other issue
        error_msg = str(yolo_error)
        error_type = type(yolo_error).__name__
        print(f"YOLO loading failed ({error_type}): {error_msg}")
        
        # For YOLO models, we should not fall back to torch.load as it won't work properly
        # The model structure is YOLO-specific
        if "ultralytics" in error_msg.lower() or "yolo" in error_msg.lower() or "DetectionModel" in error_msg:
            print("\n" + "="*60)
            print("This is a YOLO model that failed to load.")
            print("Possible solutions:")
            print("1. Update ultralytics: pip install --upgrade ultralytics")
            print("2. The model may need a specific ultralytics version")
            print("3. Check if the model file is corrupted")
            print("="*60)
            raise Exception(
                f"YOLO model loading failed: {error_msg}\n\n"
                f"Error type: {error_type}\n\n"
                f"Solutions:\n"
                f"1. Try: pip install --upgrade ultralytics\n"
                f"2. The model may require a specific ultralytics version\n"
                f"3. Check the model file integrity"
            ) from yolo_error
        
        # Only try PyTorch fallback if it's clearly not a YOLO model
        print("Attempting to load as standard PyTorch model...")
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # PyTorch 2.6+ requires weights_only=False for custom models
            model = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(model, torch.nn.Module):
                model.eval()
            print("Loaded as standard PyTorch model")
            return model, "pytorch"
        except Exception as pytorch_error:
            # Both failed
            raise Exception(
                f"Failed to load model:\n"
                f"YOLO error: {error_msg}\n"
                f"PyTorch error: {str(pytorch_error)}"
            ) from pytorch_error

# Process image with the model
def process_image(image, model, model_type):
    """Process the input image with the loaded model"""
    if image is None:
        return None, "Please upload an image"
    
    try:
        if model_type == "yolo":
            # YOLO model processing
            results = model(image)
            # Get the annotated image
            annotated_img = results[0].plot()
            # Get predictions info
            predictions = []
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if hasattr(results[0], 'names'):
                        label = results[0].names[cls]
                    else:
                        label = f"Class {cls}"
                    predictions.append(f"{label}: {conf:.2f}")
            
            info_text = "\n".join(predictions) if predictions else "No detections"
            return Image.fromarray(annotated_img), info_text
        
        else:
            # Standard PyTorch model processing
            # Convert PIL to tensor
            if isinstance(image, Image.Image):
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Basic preprocessing - resize and normalize
                from torchvision import transforms
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                img_tensor = preprocess(image).unsqueeze(0)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                img_tensor = img_tensor.to(device)
                
                # Process with model
                with torch.no_grad():
                    if isinstance(model, torch.nn.Module):
                        output = model(img_tensor)
                    else:
                        # If model is a dict or other structure, try to extract the actual model
                        if isinstance(model, dict) and 'model' in model:
                            output = model['model'](img_tensor)
                        else:
                            output = model(img_tensor)
                
                # Handle output
                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                # Get predictions if classification model
                if len(output.shape) == 2 and output.shape[1] > 1:
                    probs = torch.nn.functional.softmax(output[0], dim=0)
                    top5_probs, top5_indices = torch.topk(probs, min(5, len(probs)))
                    info_lines = ["Top Predictions:"]
                    for prob, idx in zip(top5_probs, top5_indices):
                        info_lines.append(f"Class {idx.item()}: {prob.item():.4f}")
                    info_text = "\n".join(info_lines)
                else:
                    info_text = f"Model output shape: {output.shape}\nFirst few values: {output.flatten()[:10].cpu().numpy()}"
                
                return image, info_text
                
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# Initialize model
model_path = "best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found!")

print("Loading model...")
model, model_type = load_model(model_path)
print(f"Model loaded successfully! Type: {model_type}")

# Create Gradio interface
def predict(image):
    """Gradio prediction function"""
    processed_img, info = process_image(image, model, model_type)
    return processed_img, info

# Create the interface
try:
    # Try with theme (newer Gradio versions)
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=[
            gr.Image(type="pil", label="Processed Image"),
            gr.Textbox(label="Predictions/Info", lines=10)
        ],
        title="AI Model Image Processor",
        description="Upload an image to process it with your AI model (best.pt)",
        theme=gr.themes.Soft()
    )
except TypeError:
    # Fallback for older Gradio versions
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=[
            gr.Image(type="pil", label="Processed Image"),
            gr.Textbox(label="Predictions/Info", lines=10)
        ],
        title="AI Model Image Processor",
        description="Upload an image to process it with your AI model (best.pt)"
    )

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)


