import torch
import torch.nn as nn

class ECGTransNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ECGTransNet, self).__init__()
        # 1D-CNN Front-end for local morphology
        self.cnn_features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Transformer Encoder for global rhythm context
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.cnn_features(x)
        x = x.permute(2, 0, 1) # Prep for Transformer (Seq, Batch, Feature)
        x = self.transformer(x)
        return self.classifier(x.mean(dim=0))
