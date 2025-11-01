import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix


# Import from your existing utils file located in the notebooks directory
# We need to add the path for Python to find it
import sys
sys.path.append(os.path.join(os.getcwd(), 'notebooks'))
import utils as fma_utils

# Suppress annoying librosa warnings about audioread
warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')

# --- Configuration ---
class Config:
    # Paths based on your directory structure
    FMA_METADATA_DIR = './data/fma_metadata' 
        # Path to the precomputed MFCCs
    FMA_MFCC_DIR = './data/fma_mfccs_precomputed' 
    FMA_AUDIO_DIR = './data/fma_small' # Main audio directory
    OUTPUT_DIR = './training_outputs' 

    # Audio processing parameters
    SAMPLE_RATE = 22050
    N_MFCC = 20
    # All MFCCs will be padded or truncated to this length in the time dimension
    FIXED_MFCC_LENGTH = 640  

    # Training hyperparameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    NUM_WORKERS = 8 # Number of parallel processes for data loading

# --- Custom PyTorch Dataset ---
# --- Update your Dataset ---
class FmaDataset(Dataset):
    """
    Custom PyTorch Dataset for loading PRECOMPUTED MFCCs from the FMA-small dataset.
    """
    # The audio_dir is now the mfcc_dir
    def __init__(self, metadata_df, mfcc_dir, config):
        self.metadata = metadata_df
        self.mfcc_dir = mfcc_dir
        self.config = config
        self.track_ids = metadata_df.index.tolist()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        
        # Path to the precomputed .npy file
        mfcc_path = os.path.join(self.mfcc_dir, f"{track_id}.npy")
        
        try:
            # 1. LOAD the precomputed MFCC array
            mfcc = np.load(mfcc_path)
            
            # 2. GET the pre-encoded genre label
            label = self.metadata.loc[track_id, 'genre_encoded'].iloc[0] 
            
            # 3. CONVERT to tensors
            # Add a channel dimension for the CNN (batch, channel, height, width)
            mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return mfcc_tensor, label_tensor

        except Exception as e:
            # print(f"\nError loading MFCC for track {track_id}: {e}")
            # Return a dummy sample to avoid crashing
            return torch.zeros(1, self.config.N_MFCC, self.config.FIXED_MFCC_LENGTH), torch.tensor(0, dtype=torch.long)

# --- CNN Model Architecture ---
class CNN(nn.Module):
    def __init__(self, num_classes=8):
        super(CNN, self).__init__()
        # Input shape: (batch_size, 1, N_MFCC, FIXED_MFCC_LENGTH)
        self.conv_stack = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # <-- Increased filters from 32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # <-- Increased filters from 64
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2), # <-- Changed MaxPool from 4 to 2
            nn.Dropout(0.25),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # <-- Increased filters from 128
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2, 4)), # Final pooling layer to reduce dimensions
            nn.Dropout(0.25),
        )
        self.flatten = nn.Flatten()
        
        # Dummy forward pass to calculate the input size of the fully connected layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, Config.N_MFCC, Config.FIXED_MFCC_LENGTH)
            flattened_size = self.flatten(self.conv_stack(dummy_input)).shape[1]
        
        self.fc_stack = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.fc_stack(x)
        return logits

# --- Main Execution Block ---
if __name__ == '__main__':
    print(f"Using device: {Config.DEVICE} 🚀")

    # 1. Load metadata using your utils.load function
    tracks_path = os.path.join(Config.FMA_METADATA_DIR, 'tracks.csv')
    tracks = fma_utils.load(tracks_path)
    
    # 2. Filter for the 'small' subset and prepare labels
    small_subset = tracks[tracks[('set', 'subset')] == 'small'].copy()
    
    # Use LabelEncoder to convert genre strings to integers
    le = LabelEncoder()
    # We create a new column at the top level of the MultiIndex DataFrame
    small_subset['genre_encoded'] = le.fit_transform(small_subset[('track', 'genre_top')])
    num_classes = len(le.classes_)
    print(f"Found {num_classes} genres: {list(le.classes_)}")

    # 3. Split data into training and validation sets
    train_meta = small_subset[small_subset[('set', 'split')] == 'training']
    val_meta = small_subset[small_subset[('set', 'split')] == 'validation']
    
    print(f"Training set size: {len(train_meta)}")
    print(f"Validation set size: {len(val_meta)}")

    # 4. Create Dataset and DataLoader instances
    train_dataset = FmaDataset(train_meta, Config.FMA_MFCC_DIR, Config)
    val_dataset = FmaDataset(val_meta, Config.FMA_MFCC_DIR, Config)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)

    # 5. Initialize model, optimizer, and loss function
    model = CNN(num_classes=num_classes).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("\nStarting training...")
    # 6. Training and Validation Loop
    for epoch in range(Config.NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]")
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': f'{train_loss/train_total:.4f}', 'acc': f'{train_correct/train_total:.4f}'})

        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Val]")
        
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({'loss': f'{val_loss/val_total:.4f}', 'acc': f'{val_correct/val_total:.4f}'})
        
        print(f"Epoch {epoch+1:02d} Summary | Train Acc: {train_correct/train_total:.4f} | Val Acc: {val_correct/val_total:.4f}\n")

    # --------------------------------------------------------------------------
    # FINAL EVALUATION AND SAVING
    # --------------------------------------------------------------------------
    print("\nTraining complete. Starting final evaluation and saving artifacts...")
    
    # 1. Create a unique directory for this training run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = os.path.join(Config.OUTPUT_DIR, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Saving artifacts to: {run_output_dir}")
    
    # 2. Save training parameters to a JSON file 📝
    params = {
        'learning_rate': Config.LEARNING_RATE,
        'batch_size': Config.BATCH_SIZE,
        'num_epochs': Config.NUM_EPOCHS,
        'n_mfcc': Config.N_MFCC,
        'fixed_mfcc_length': Config.FIXED_MFCC_LENGTH,
        'num_classes': num_classes,
        'class_names': list(le.classes_)
    }
    params_path = os.path.join(run_output_dir, 'params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    # 3. Save the trained model's state dictionary 💾
    model_path = os.path.join(run_output_dir, 'model_weights.pth')
    torch.save(model.state_dict(), model_path)
    
    # 4. Generate and save the Classification Report 📊
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Generating Classification Report"):
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Append batch results to the master lists (move to CPU)
            all_labels.append(labels.cpu())
            all_preds.append(predicted.cpu())
    
    # Concatenate all batches
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    # Generate the report string
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=le.classes_,
        digits=4
    )

    # Generate the Confusion Matrix matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Format the Confusion Matrix for printing
    cm_report = "Confusion Matrix\n"
    cm_report += "================\n\n"
    # Header
    header = f'{"": <15}' + " ".join([f'{label: <10}' for label in le.classes_])
    cm_report += header + "\n"
    cm_report += "-" * len(header) + "\n"
    # Rows
    for i, row in enumerate(cm):
        row_str = f'{le.classes_[i]: <15}'
        for val in row:
            row_str += f'{val: <10}'
        cm_report += row_str + "\n"
    
    print("\n" + cm_report)


    
    # Save the report to a text file
    report_path = os.path.join(run_output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=======================\n\n")
        f.write(report)
    
    print("\nClassification Report:")
    print(report)
    
    print("Training finished! 🎉") # This is your original final line