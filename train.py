import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from config import *
from models import CNNModel, RNNModel
from dataset_loader import SpokenDigitDataset
from audio_feature_extraction import AudioFeatureExtractor

def plot_training_history(train_history, val_history, model_name):
    """Plots and saves the training and validation loss history."""
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
        
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label='Training Loss')
    plt.plot(val_history, label='Validation Loss')
    plt.title(f'Training History for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(RESULTS_PATH, f"{model_name}_training_history.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")

def train_model(model_config, model_class):
    # Setup
    torch.manual_seed(RANDOM_SEED)
    writer = SummaryWriter(log_dir=os.path.join(LOGS_PATH, model_config['model_name']))
    
    # Data
    feature_extractor = AudioFeatureExtractor()
    dataset = SpokenDigitDataset(feature_extractor)
    
    dataset_size = len(dataset)
    test_size = int(TEST_SPLIT_SIZE * dataset_size)
    val_size = int(VALIDATION_SPLIT_SIZE * (dataset_size - test_size))
    train_size = dataset_size - test_size - val_size
    
    train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = model_class(**model_config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    train_history, val_history = [], []

    print(f"--- Starting Training for {model.model_name} ---")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_history.append(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_history.append(avg_val_loss)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_model()
            
        scheduler.step() # Update learning rate

    writer.close()
    print(f"--- Finished Training for {model.model_name} ---")
    plot_training_history(train_history, val_history, model.model_name)


if __name__ == '__main__':
    train_model(CNN_CONFIG, CNNModel)
    train_model(RNN_CONFIG, RNNModel)
