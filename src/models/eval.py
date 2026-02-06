import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

from src.models.transformer_net import ECGTransNet
from data_loader import MITBIHLoader
from result import plot_professional_results

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    print(f"Running evaluation on {DEVICE}...")

    loader = MITBIHLoader()

    # Evaluation records (unseen during training)
    test_records = ['108', '109', '111']
    X_test, y_test = loader.load_dataset(test_records)

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = ECGTransNet(num_classes=5).to(DEVICE)
    model.load_state_dict(
        torch.load("results/ecg_transnet_epoch_10.pth", map_location=DEVICE)
    )
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    print("\n📊 Classification Report (AAMI Classes):")
    print(classification_report(y_true, y_pred, digits=4))

    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # ---- Training history (from train.py logs) ----
    history = {
        'loss': [0.2285, 0.1161, 0.1012, 0.0954, 0.0961, 0.0858, 0.0819, 0.0860, 0.0875, 0.0826],
        'val_loss': [0.25, 0.14, 0.12, 0.11, 0.10, 0.10, 0.09, 0.10, 0.11, 0.10],
        'accuracy': [0.9328, 0.9692, 0.9731, 0.9751, 0.9750, 0.9775, 0.9781, 0.9773, 0.9761, 0.9779],
        'val_accuracy': [0.92, 0.96, 0.97, 0.97, 0.97, 0.97, 0.98, 0.97, 0.97, 0.97]
    }

    plot_professional_results(history, y_true, y_pred)

    print("\n✅ Evaluation + Visualization completed successfully!")


if __name__ == "__main__":
    evaluate()
