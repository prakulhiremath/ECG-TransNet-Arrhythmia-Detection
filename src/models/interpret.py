import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
from scipy.interpolate import interp1d
from src.models.transformer_net import ECGTransNet
from src.utils.data_loader import MITBIHLoader

# 1. Configuration & Device Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "results/ecg_transnet_best.pth"
OUTPUT_DIR = "results/interpretations/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ECGInterpreter:
    def __init__(self, model_path, num_classes=5):
        self.model = ECGTransNet(num_classes=num_classes).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        self.features = None
        self.gradients = None

    def _save_gradients(self, grad):
        self.gradients = grad

    def _save_features(self, module, input, output):
        self.features = output

    def get_grad_cam_1d(self, input_tensor, target_class):
        """
        Calculates 1D Grad-CAM.
        Targeting the last convolutional layer before the Transformer blocks.
        """
        # Change 'cnn[-1]' if your architecture uses a different attribute name
        target_layer = self.model.cnn[-1] 
        
        # Register Hooks
        forward_handle = target_layer.register_forward_hook(self._save_features)
        backward_handle = target_layer.register_full_backward_hook(lambda m, i, o: self._save_gradients(o[0]))

        # Forward Pass
        output = self.model(input_tensor)
        
        # Backward Pass
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        # Compute Grad-CAM weights (Mean of gradients per channel)
        weights = torch.mean(self.gradients, dim=2, keepdim=True)
        cam = torch.sum(weights * self.features, dim=1).squeeze()
        
        # Process Heatmap: ReLU + Normalization
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) + 1e-10)

        # Cleanup hooks
        forward_handle.remove()
        backward_handle.remove()
        
        return cam, output.detach().cpu().numpy()

    def run_shap_analysis(self, background_data, test_sample):
        """
        DeepExplainer SHAP: Measures the contribution of each time-step.
        """
        explainer = shap.DeepExplainer(self.model, background_data)
        shap_values = explainer.shap_values(test_sample)
        # For multi-class, shap_values is a list. We take the one for predicted class.
        return shap_values

def plot_detailed_interpretation(signal, cam, shap_vals, pred_label, true_label, save_path):
    """
    Generates a professional multi-panel figure.
    """
    classes = ['N', 'S', 'V', 'F', 'Q']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=300, sharex=True)
    
    # --- Panel 1: Grad-CAM (Morphological Attention) ---
    x = np.linspace(0, len(signal), len(cam))
    f = interp1d(x, cam, kind='cubic')
    cam_interp = f(np.arange(len(signal)))
    
    ax1.plot(signal, color='black', linewidth=1.5, label='ECG Lead II')
    # Use pcolormesh for a heatmap 'glow' behind the signal
    im = ax1.pcolormesh(np.arange(len(signal)), [min(signal), max(signal)], 
                        cam_interp.reshape(1, -1), cmap='YlOrRd', alpha=0.4, shading='auto')
    ax1.set_title(f"Spatial Attention (Grad-CAM) | Pred: {classes[pred_label]} (True: {classes[true_label]})", fontweight='bold')
    ax1.set_ylabel("Voltage (mV)")
    plt.colorbar(im, ax=ax1, label="Importance")

    # --- Panel 2: SHAP Values (Temporal Contribution) ---
    # Reshape SHAP values to match signal
    shap_plot = shap_vals[pred_label].flatten()
    ax2.fill_between(range(len(signal)), shap_plot, color='blue', alpha=0.6)
    ax2.set_title("Temporal Feature Attribution (SHAP)", fontweight='bold')
    ax2.set_ylabel("SHAP Value")
    ax2.set_xlabel("Time Samples (250 Hz)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Initialize Interpreter
    interpreter = ECGInterpreter(MODEL_PATH)
    
    # Load Real Data
    loader = MITBIHLoader()
    X, y = loader.load_dataset(['100']) # Using a test record
    
    # Select a specific beat (e.g., a Ventricular beat)
    sample_idx = 5
    input_tensor = torch.Tensor(X[sample_idx:sample_idx+1]).to(DEVICE)
    
    # 1. Generate Grad-CAM
    cam, raw_output = interpreter.get_grad_cam_1d(input_tensor, y[sample_idx])
    pred_class = np.argmax(raw_output)

    # 2. Generate SHAP (using first 50 beats as background baseline)
    background = torch.Tensor(X[:50]).to(DEVICE)
    shap_vals = interpreter.run_shap_analysis(background, input_tensor)

    # 3. Export High-Res PNG
    save_file = os.path.join(OUTPUT_DIR, f"beat_{sample_idx}_analysis.png")
    plot_detailed_interpretation(X[sample_idx], cam, shap_vals, pred_class, y[sample_idx], save_file)
    
    print(f"📊 Detailed interpretation report saved to: {save_file}")
