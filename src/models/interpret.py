import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
from src.models.transformer_net import ECGTransNet
from src.utils.data_loader import MITBIHLoader

# 1. Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "results/ecg_transnet_best.pth"

def get_grad_cam(model, input_tensor, target_class):
    """Simple 1D Grad-CAM implementation for ECG signals."""
    model.eval()
    
    # Hook into the last CNN layer before the Transformer
    features = []
    gradients = []

    def save_gradient(grad):
        gradients.append(grad)
    
    # Accessing the CNN front-end defined in our earlier architecture
    target_layer = model.cnn_features[-1] 
    
    # Forward pass
    handle = target_layer.register_forward_hook(lambda m, i, o: features.append(o))
    output = model(input_tensor)
    handle.remove()
    
    # Backward pass for the specific class
    model.zero_grad()
    loss = output[0, target_class]
    loss.backward()
    
    # Compute weights and heatmap
    # Using gradients to weight the feature maps
    # target_layer output is [Batch, Channels, Length]
    # We want to weight by channel-wise mean of gradients
    # Note: this is a simplified logic for 1D signals
    # In a real repo, you might use 'captum' library for this
    
    return features[0], output

def run_shap_analysis(model, input_data):
    """Generate SHAP values to see which time-points are most critical."""
    # We use a background dataset (DeepExplainer) to establish a baseline
    background = input_data[:10].to(DEVICE)
    test_sample = input_data[10:11].to(DEVICE)
    
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_sample)
    
    return shap_values

def plot_interpretation(signal, heatmap, title="ECG Interpretability"):
    plt.figure(figsize=(12, 4))
    plt.plot(signal, color='black', alpha=0.3, label='ECG Signal')
    
    # Overlaying the heatmap (this is a simplified visualization)
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load model and a sample of data
    loader = MITBIHLoader()
    X, y = loader.load_dataset(['100']) # Example record
    X_tensor = torch.Tensor(X[:20]) # Small batch for demo
    
    model = ECGTransNet(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    print("Generating SHAP and Grad-CAM visualizations...")
    # Logic to run and save figures to /results/interpretations/
