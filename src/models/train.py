import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.transformer_net import ECGTransNet
from src.utils.data_loader import MITBIHLoader

# 1. Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 2. Load Data (Using our custom MITBIHLoader)
    loader = MITBIHLoader()
    # Example records - you can expand this list
    train_records = ['100', '101', '103', '105', '106'] 
    X, y = loader.load_dataset(train_records)
    
    dataset = TensorDataset(torch.Tensor(X), torch.LongTensor(y))
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize Model, Loss, and Optimizer
    model = ECGTransNet(num_classes=5).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # 4. Training Loop
    print(f"Starting training on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"results/ecg_transnet_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
