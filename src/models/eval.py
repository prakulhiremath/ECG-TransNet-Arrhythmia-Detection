import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset
from src.models.transformer_net import ECGTransNet
from src.utils.data_loader import MITBIHLoader

# 1. Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "results/ecg_transnet_best.pth"
OUTPUT_DIR = "results/evaluation/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q'] # Standard AAMI short-hand

def evaluate():
    # 2. Load Data (Ensuring 'unseen' records)
    loader = MITBIHLoader()
    # Records 102, 104, 107, 217 are excellent for testing (paced/complex beats)
    test_records = ['102', '104', '107', '217'] 
    X_test, y_test = loader.load_dataset(test_records)
    
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 3. Load Model
    model = ECGTransNet(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    # 4. Inference
    print(f"🚀 Running Inference on {len(X_test)} beats...")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            outputs = model(batch_x)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 5. Advanced Metrics Calculation
    print("\n" + "="*30)
    print("      FINAL TEST METRICS")
    print("="*30)
    
    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    print(report)

    # Macro AUROC
    macro_auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    print(f"Overall Macro AUROC: {macro_auroc:.4f}")

    # 6. ROC Curve Visualization (Professional Multi-class)
    plt.figure(figsize=(8, 6), dpi=300)
    for i in range(len(CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {CLASS_NAMES[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(f'{OUTPUT_DIR}roc_curves.png')

    # 7. Normalized Confusion Matrix (Better for Medical Imbalance)
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize by row
    
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='magma', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Normalized Confusion Matrix (Recall per Class)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{OUTPUT_DIR}confusion_matrix_norm.png')

    # 8. Export Data for MATLAB / interpret.py
    np.savez(f'results/eval_results.npz', 
             y_true=all_labels, 
             y_pred=all_preds, 
             y_probs=all_probs)
    
    print(f"\n✅ Evaluation complete. Files saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    evaluate()
