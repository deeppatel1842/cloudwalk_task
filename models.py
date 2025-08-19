import os
import torch
import torch.nn as nn
from config import SAVED_MODELS_PATH, N_MELS

class BaseModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def save_model(self, filename="best_model.pth"):
        if not os.path.exists(SAVED_MODELS_PATH):
            os.makedirs(SAVED_MODELS_PATH)
        save_path = os.path.join(SAVED_MODELS_PATH, self.model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.state_dict(), os.path.join(save_path, filename))
        print(f"Model saved to {os.path.join(save_path, filename)}")

    def load_model(self, filename="best_model.pth"):
        load_path = os.path.join(SAVED_MODELS_PATH, self.model_name, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No model file found at {load_path}")
        self.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
        print(f"Model loaded from {load_path}")

class CNNModel(BaseModel):
    def __init__(self, in_channels, num_classes, model_name):
        super().__init__(model_name)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        
        # Global Average Pooling layer makes the model robust to input size variations
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            # The input size is now fixed to the number of channels from the last conv layer
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = self.fc_layers(x)
        return x

class RNNModel(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, model_name):
        super().__init__(model_name)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
