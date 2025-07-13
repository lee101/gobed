import onnx

# Load the ONNX model to inspect its structure
model_path = "model/embedding_model.onnx"

try:
    model = onnx.load(model_path)
    
    print("Model inputs:")
    for input_info in model.graph.input:
        print(f"  Name: {input_info.name}")
        print(f"  Type: {input_info.type}")
        print()
    
    print("Model outputs:")
    for output_info in model.graph.output:
        print(f"  Name: {output_info.name}")
        print(f"  Type: {output_info.type}")
        print()
        
except Exception as e:
    print(f"Error loading model: {e}")
    print("Model file might not exist or be corrupted")
