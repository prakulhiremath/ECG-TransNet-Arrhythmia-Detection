import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from src.models.transformer_net import ECGTransNet
from src.utils.data_loader import MITBIHLoader

# 1. Setup and Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "results/ecg_transnet_best.pth"  # Ensure your best model is saved here
CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

def evaluate():
    # 2. Load Test Data
    # Use records that were NOT in your training set to ensure a fair test
    loader = MITBIHLoader()
    test_records = ['102', '104', '107', '217'] 
    X_test, y_test = loader.load_dataset(test_records)
    
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. Load Model
    model = ECGTransNet(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    # 4. Inference
    print("Running evaluation...")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            outputs = model(batch_x)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    # 5. Generate Metrics
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Calculate AUROC (Standard for Research Papers)
    auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    print(f"Overall Macro AUROC: {auroc:.4f}")

    # 6. Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('ECG-TransNet Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.savefig('results/confusion_matrix.png')
    print("Confusion matrix saved to results/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    evaluate()
