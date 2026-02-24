import torch
import torch.nn as nn


class ECGTransNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ECGTransNet, self).__init__()

        # ===============================
        # 1. CNN Front-End
        # ===============================
        self.cnn_features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # ===============================
        # 2. Transformer Encoder
        # ===============================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            dim_feedforward=128,
            dropout=0.3,
            activation="relu",
            batch_first=False
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        # ===============================
        # 3. Regularization + Classifier
        # ===============================
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):

        # x shape: (batch, 1, sequence_length)

        # CNN feature extraction
        x = self.cnn_features(x)

        # Prepare for Transformer: (seq_len, batch, feature_dim)
        x = x.permute(2, 0, 1)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling over sequence
        x = x.mean(dim=0)

        # Dropout before classification
        x = self.dropout(x)

        # Final classification
        x = self.classifier(x)

        return x
