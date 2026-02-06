import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.transformer_net import ECGTransNet
from data_loader import MITBIHLoader


# ===============================
# 1. Hyperparameters
# ===============================
EPOCHS = 10          # 🔥 start with 10 (we’ll switch to 50 later)
BATCH_SIZE = 32
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    # ===============================
    # 2. Load Dataset
    # ===============================
    loader = MITBIHLoader()
    train_records = [
    '100','101','102','103','104','105','106',
    '107','108','109','111','112','113','114',
    '115','116','117','118','119','121','122']


    X, y = loader.load_dataset(train_records)

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ===============================
    # 3. Model, Loss, Optimizer
    # ===============================
    model = ECGTransNet(num_classes=5).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # Create results folder
    os.makedirs("results", exist_ok=True)

    print(f"\n🚀 Training started on {DEVICE}")
    print("=" * 50)

    # ===============================
    # 4. Training Loop
    # ===============================
    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

        # Save model every epoch (since only 10 epochs)
        torch.save(
            model.state_dict(),
            f"results/ecg_transnet_epoch_{epoch+1}.pth"
        )

    print("\n✅ Training completed successfully!")
    print("📁 Models saved in /results folder")


if __name__ == "__main__":
    train()
