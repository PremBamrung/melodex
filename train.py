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
    FMA_AUDIO_DIR = './data/fma_small' # Main audio directory
    
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
class FmaDataset(Dataset):
    """
    Custom PyTorch Dataset for the FMA-small dataset.
    It loads audio, computes MFCCs on-the-fly, and returns them with their labels.
    """
    def __init__(self, metadata_df, audio_dir, config):
        self.metadata = metadata_df
        self.audio_dir = audio_dir
        self.config = config
        # The track_id is the index of the metadata dataframe
        self.track_ids = metadata_df.index.tolist()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        
        # Use the get_audio_path function from your utils.py
        audio_path = fma_utils.get_audio_path(self.audio_dir, track_id)
        
        try:
            # Load audio file using librosa
            signal, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE, duration=30)

            # Compute MFCCs
            mfcc = librosa.feature.mfcc(
                y=signal, 
                sr=sr, 
                n_mfcc=self.config.N_MFCC
            )

            # Pad or truncate MFCCs to a fixed length for batching
            if mfcc.shape[1] > self.config.FIXED_MFCC_LENGTH:
                mfcc = mfcc[:, :self.config.FIXED_MFCC_LENGTH]
            else:
                pad_width = self.config.FIXED_MFCC_LENGTH - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
            # Get the pre-encoded genre label
            label = self.metadata.loc[track_id, 'genre_encoded']

            # Add a channel dimension for the CNN (batch, channel, height, width)
            mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
            # FIX: Cast label to a standard Python integer to ensure it's a scalar
            label_tensor = torch.tensor(int(label.iloc[0]), dtype=torch.long)

            
            return mfcc_tensor, label_tensor

        except Exception as e:
            # If a file is corrupt or missing, we print an error and return a dummy sample
            print(f"\nError processing track {track_id} at {audio_path}: {e}")
            # Return a zero tensor and a placeholder label to avoid crashing the training loop
            return torch.zeros(1, self.config.N_MFCC, self.config.FIXED_MFCC_LENGTH), torch.tensor(0, dtype=torch.long)

# --- CNN Model Architecture ---
class CNN(nn.Module):
    def __init__(self, num_classes=8):
        super(CNN, self).__init__()
        # Input shape: (batch_size, 1, N_MFCC, FIXED_MFCC_LENGTH)
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(4),
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
    train_dataset = FmaDataset(train_meta, Config.FMA_AUDIO_DIR, Config)
    val_dataset = FmaDataset(val_meta, Config.FMA_AUDIO_DIR, Config)
    
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

    print("Training finished! 🎉")