import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter

from src.models.transformer_net import ECGTransNet
from data_loader import MITBIHLoader


# ===============================
# 1. Hyperparameters
# ===============================
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():

    # ===============================
    # 2. Load Dataset
    # ===============================
    loader = MITBIHLoader()

    train_records = [
        '100','101','102','103','104','105','106',
        '107','108','109','111','112','113','114',
        '115','116','117','118','119','121','122'
    ]

    X, y = loader.load_dataset(train_records)

    # ===============================
    # 3. Train/Validation Split
    # ===============================
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ===============================
    # 4. Compute Class Weights
    # ===============================
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    num_classes = 5

    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weights.append(total_samples / (num_classes * count))

    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    # ===============================
    # 5. Model, Loss, Optimizer
    # ===============================
    model = ECGTransNet(num_classes=5).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    os.makedirs("results", exist_ok=True)

    print(f"\nTraining on {DEVICE}")
    print("=" * 50)

    best_val_acc = 0.0

    # ===============================
    # 6. Training Loop
    # ===============================
    for epoch in range(EPOCHS):

        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_acc = 100.0 * train_correct / train_total
        train_loss /= len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:

                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = 100.0 * val_correct / val_total
        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "results/best_ecg_transnet.pth")

    print("\nTraining completed.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("Best model saved as results/best_ecg_transnet.pth")


if __name__ == "__main__":
    train()
