import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import json
import csv
import warnings
from pathlib import Path
from typing import List, Tuple, Union, Dict, Optional
from tqdm import tqdm

warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')


# --- Model Architecture (must match training) ---
class CNN(nn.Module):
    def __init__(self, num_classes=8, n_mfcc=20, fixed_mfcc_length=640):
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
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2, 4)),  # Final pooling layer to reduce dimensions
            nn.Dropout(0.25),
        )
        self.flatten = nn.Flatten()
        
        # Dummy forward pass to calculate the input size of the fully connected layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_mfcc, fixed_mfcc_length)
            flattened_size = self.flatten(self.conv_stack(dummy_input)).shape[1]
        
        # FC layers (matching training architecture)
        self.fc_stack = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, return_embedding=False):
        x = self.conv_stack(x)
        x = self.flatten(x)
        
        if return_embedding:
            # Extract embedding from before the final classification layer
            embedding = self.fc_stack[0:3](x)  # Linear + ReLU + Dropout
            logits = self.fc_stack[3](embedding)  # Final Linear layer
            return logits, embedding
        else:
            logits = self.fc_stack(x)
            return logits


# --- Audio Processing ---
def preprocess_audio(audio_path: str, sample_rate: int = 22050, n_mfcc: int = 20, fixed_mfcc_length: int = 640) -> np.ndarray:
    """
    Load an audio file and extract MFCC features with the same parameters used during training.
    
    Args:
        audio_path: Path to the audio file
        sample_rate: Target sample rate (default: 22050)
        n_mfcc: Number of MFCC coefficients (default: 20)
        fixed_mfcc_length: Target length for the time dimension (default: 640)
    
    Returns:
        MFCC array of shape (n_mfcc, fixed_mfcc_length)
    """
    try:
        # Load audio (duration=30 seconds to match training)
        signal, sr = librosa.load(audio_path, sr=sample_rate, duration=30)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate to fixed length
        if mfcc.shape[1] > fixed_mfcc_length:
            mfcc = mfcc[:, :fixed_mfcc_length]
        else:
            pad_width = fixed_mfcc_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        # Add channel dimension for CNN: (1, n_mfcc, fixed_mfcc_length)
        mfcc = np.expand_dims(mfcc, axis=0)
        
        return mfcc
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        # Return a zero array with the correct shape
        return np.zeros((1, n_mfcc, fixed_mfcc_length))


def preprocess_audio_sliding_window(
    audio_path: str, 
    sample_rate: int = 22050, 
    n_mfcc: int = 20, 
    fixed_mfcc_length: int = 640,
    segment_duration: float = 30.0,
    hop_duration: float = 30.0
) -> List[np.ndarray]:
    """
    Load an audio file and extract MFCC features using a sliding window approach.
    This allows processing of audio files longer than the training segment duration.
    
    Args:
        audio_path: Path to the audio file
        sample_rate: Target sample rate (default: 22050)
        n_mfcc: Number of MFCC coefficients (default: 20)
        fixed_mfcc_length: Target length for the time dimension (default: 640)
        segment_duration: Duration of each segment in seconds (default: 30.0)
        hop_duration: Hop size between segments in seconds (default: 30.0, non-overlapping)
    
    Returns:
        List of MFCC arrays, each of shape (n_mfcc, fixed_mfcc_length)
    """
    try:
        # Load full audio file without duration limit
        signal, sr = librosa.load(audio_path, sr=sample_rate, duration=None)
        
        # Calculate segment and hop lengths in samples
        segment_length = int(segment_duration * sr)
        hop_length = int(hop_duration * sr)
        
        # Calculate total duration and number of segments
        total_duration = len(signal) / sr
        
        # Split audio into segments
        segments = []
        start = 0
        
        while start < len(signal):
            end = min(start + segment_length, len(signal))
            segment = signal[start:end]
            
            # Only process segments that are at least 5 seconds long
            if len(segment) >= int(5 * sr):
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
                
                # Pad or truncate to fixed length
                if mfcc.shape[1] > fixed_mfcc_length:
                    mfcc = mfcc[:, :fixed_mfcc_length]
                else:
                    pad_width = fixed_mfcc_length - mfcc.shape[1]
                    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
                
                # Add channel dimension for CNN: (1, n_mfcc, fixed_mfcc_length)
                mfcc = np.expand_dims(mfcc, axis=0)
                segments.append(mfcc)
            
            start += hop_length
            
            # Break if we've reached the end
            if end >= len(signal):
                break
        
        if not segments:
            # If no valid segments, return a single segment from the beginning
            return [preprocess_audio(audio_path, sample_rate, n_mfcc, fixed_mfcc_length)]
        
        tqdm.write(f"  Split into {len(segments)} segments (total duration: {total_duration:.1f}s)")
        return segments
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        # Return a single zero array
        return [np.zeros((1, n_mfcc, fixed_mfcc_length))]


# --- PyTorch Dataset for Streaming Audio Files ---
class AudioSegmentDataset(Dataset):
    """
    PyTorch Dataset for streaming audio segments.
    Processes audio files into segments and provides them for batch processing.
    """
    def __init__(
        self,
        audio_files: List[str],
        sample_rate: int = 22050,
        n_mfcc: int = 20,
        fixed_mfcc_length: int = 640,
        segment_duration: float = 30.0,
        hop_duration: float = 30.0,
        full_file: bool = False
    ):
        """
        Args:
            audio_files: List of audio file paths
            sample_rate: Target sample rate
            n_mfcc: Number of MFCC coefficients
            fixed_mfcc_length: Target length for time dimension
            segment_duration: Duration of each segment in seconds
            hop_duration: Hop size between segments in seconds
            full_file: If True, use sliding window for full file
        """
        self.audio_files = audio_files
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.fixed_mfcc_length = fixed_mfcc_length
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.full_file = full_file
        
        # Build index: list of (file_index, segment_index, start_time, end_time)
        self.segment_index = []
        self._build_index()
    
    def _build_index(self):
        """Build index mapping dataset indices to (file, segment, times)."""
        for file_idx, audio_file in enumerate(tqdm(self.audio_files, desc="Building audio index", unit="file")):
            try:
                # Get audio duration without loading full file
                duration = librosa.get_duration(path=audio_file)
                
                if not self.full_file:
                    # Single segment (first 30s)
                    self.segment_index.append((file_idx, 0, 0.0, min(duration, self.segment_duration)))
                else:
                    # Multiple segments with sliding window
                    segment_length = self.segment_duration
                    hop_length = self.hop_duration
                    
                    start = 0.0
                    segment_idx = 0
                    
                    while start < duration:
                        end = min(start + segment_length, duration)
                        
                        # Only add segments that are at least 5 seconds
                        if (end - start) >= 5.0:
                            self.segment_index.append((file_idx, segment_idx, start, end))
                            segment_idx += 1
                        
                        start += hop_length
                        if end >= duration:
                            break
                            
            except Exception as e:
                print(f"Warning: Could not get duration for {audio_file}: {e}")
                # Add at least one segment
                self.segment_index.append((file_idx, 0, 0.0, self.segment_duration))
    
    def __len__(self):
        return len(self.segment_index)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - 'mfcc': tensor of shape (1, n_mfcc, fixed_mfcc_length)
                - 'file_index': index of the audio file
                - 'segment_index': index of the segment within the file
                - 'audio_file': path to audio file
                - 'start_time': start time of segment
                - 'end_time': end time of segment
        """
        file_idx, segment_idx, start_time, end_time = self.segment_index[idx]
        audio_file = self.audio_files[file_idx]
        
        try:
            # Load audio segment
            duration = end_time - start_time
            signal, sr = librosa.load(
                audio_file,
                sr=self.sample_rate,
                offset=start_time,
                duration=duration
            )
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=self.n_mfcc)
            
            # Pad or truncate to fixed length
            if mfcc.shape[1] > self.fixed_mfcc_length:
                mfcc = mfcc[:, :self.fixed_mfcc_length]
            else:
                pad_width = self.fixed_mfcc_length - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
            # Add channel dimension: (1, n_mfcc, fixed_mfcc_length)
            mfcc = np.expand_dims(mfcc, axis=0)
            
        except Exception as e:
            print(f"Error processing segment {segment_idx} of {audio_file}: {e}")
            mfcc = np.zeros((1, self.n_mfcc, self.fixed_mfcc_length))
        
        return {
            'mfcc': torch.tensor(mfcc, dtype=torch.float32),
            'file_index': file_idx,
            'segment_index': segment_idx,
            'audio_file': audio_file,
            'start_time': start_time,
            'end_time': end_time
        }


# --- Inference Function ---
def run_inference(
    model_path: str,
    audio_files: List[str],
    params: Dict = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    return_embeddings: bool = True,
    full_file: bool = False,
    top_k: int = 1
) -> Tuple[Union[List[str], List[List[str]]], np.ndarray, np.ndarray]:
    """
    Run inference on a list of audio files using a trained model.
    
    Args:
        model_path: Path to the trained model weights (.pth file) or directory containing model
        audio_files: List of paths to audio files
        params: Dictionary containing model parameters (loaded from params.json)
                If None, will be inferred from the model directory
        device: Device to run inference on ('cuda' or 'cpu')
        return_embeddings: Whether to return embeddings
        full_file: If True, process entire audio file using sliding window and average predictions
        top_k: Number of top predictions to return (default: 1)
    
    Returns:
        labels: List of predicted genre labels (or list of top-k labels if top_k > 1)
        probabilities: Array of prediction probabilities for each class (or top-k probabilities)
        embeddings: Array of embeddings (if return_embeddings=True, else None)
    """
    # Handle directory path (convert to .pth file path)
    if os.path.isdir(model_path):
        model_dir = model_path
        model_path = os.path.join(model_path, 'model_weights.pth')
    else:
        model_dir = os.path.dirname(model_path)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 1. Load model parameters
    if params is None:
        params_path = os.path.join(model_dir, 'params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
        else:
            raise FileNotFoundError(f"Could not find params.json at {params_path}")
    
    # 2. Initialize model with saved parameters
    model = CNN(
        num_classes=params['num_classes'],
        n_mfcc=params.get('n_mfcc', 20),
        fixed_mfcc_length=params.get('fixed_mfcc_length', 640)
    )
    
    # 3. Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 4. Get class names
    class_names = params.get('class_names', [f'Class_{i}' for i in range(params['num_classes'])])
    
    # 5. Process audio files
    all_labels = []
    all_probs = []
    all_embeddings = []
    
    mode_str = "full-file (sliding window)" if full_file else "first 30s only"
    print(f"Processing {len(audio_files)} audio files (mode: {mode_str})...")
    
    with torch.no_grad():
        for i, audio_file in enumerate(tqdm(audio_files, desc="Processing audio files", unit="file")):
            tqdm.write(f"[{i+1}/{len(audio_files)}] Processing: {audio_file}")
            
            if full_file:
                # Full-file mode: sliding window approach
                segments = preprocess_audio_sliding_window(
                    audio_file,
                    sample_rate=params.get('sample_rate', 22050),
                    n_mfcc=params.get('n_mfcc', 20),
                    fixed_mfcc_length=params.get('fixed_mfcc_length', 640)
                )
                
                # Batch process all segments
                segment_probs = []
                segment_embeddings = []
                
                for segment_mfcc in tqdm(segments, desc="  Processing segments", unit="seg", leave=False):
                    # Convert to tensor
                    input_tensor = torch.tensor(segment_mfcc, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Forward pass
                    if return_embeddings:
                        logits, embedding = model(input_tensor, return_embedding=True)
                        embedding = embedding.cpu().numpy().flatten()
                        segment_embeddings.append(embedding)
                    else:
                        logits = model(input_tensor, return_embedding=False)
                    
                    # Get probabilities
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    segment_probs.append(probs)
                
                # Average probabilities across all segments
                avg_probs = np.mean(segment_probs, axis=0)
                
                # Get top-k predictions
                top_k_indices = np.argsort(avg_probs)[-top_k:][::-1]
                top_k_labels = [class_names[idx] for idx in top_k_indices]
                top_k_probs = avg_probs[top_k_indices]
                
                # Store results
                if top_k == 1:
                    all_labels.append(top_k_labels[0])
                    all_probs.append(avg_probs)
                else:
                    all_labels.append(top_k_labels)
                    all_probs.append(np.column_stack([top_k_indices, top_k_probs]))
                
                # Average embeddings if requested
                if return_embeddings and segment_embeddings:
                    avg_embedding = np.mean(segment_embeddings, axis=0)
                    all_embeddings.append(avg_embedding)
                    
            else:
                # Standard mode: first 30s only
                mfcc = preprocess_audio(
                    audio_file,
                    sample_rate=params.get('sample_rate', 22050),
                    n_mfcc=params.get('n_mfcc', 20),
                    fixed_mfcc_length=params.get('fixed_mfcc_length', 640)
                )
                
                # Convert to tensor
                input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Forward pass
                if return_embeddings:
                    logits, embedding = model(input_tensor, return_embedding=True)
                    embedding = embedding.cpu().numpy().flatten()
                    all_embeddings.append(embedding)
                else:
                    logits = model(input_tensor, return_embedding=False)
                
                # Get probabilities
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                # Get top-k predictions
                top_k_indices = np.argsort(probs)[-top_k:][::-1]
                top_k_labels = [class_names[idx] for idx in top_k_indices]
                top_k_probs = probs[top_k_indices]
                
                # Store results
                if top_k == 1:
                    all_labels.append(top_k_labels[0])
                    all_probs.append(probs)
                else:
                    all_labels.append(top_k_labels)
                    all_probs.append(np.column_stack([top_k_indices, top_k_probs]))
    
    # Convert to numpy arrays
    if top_k == 1:
        all_probs = np.array(all_probs)
    all_embeddings = np.array(all_embeddings) if return_embeddings and all_embeddings else None
    
    return all_labels, all_probs, all_embeddings


def run_inference_streaming(
    model_path: str,
    audio_files: List[str],
    params: Dict = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    return_embeddings: bool = True,
    full_file: bool = False,
    batch_size: int = 8,
    num_workers: int = 4
) -> List[Dict]:
    """
    Run inference using PyTorch DataLoader for efficient streaming.
    
    Args:
        model_path: Path to the trained model weights (.pth file) or directory containing model
        audio_files: List of paths to audio files
        params: Dictionary containing model parameters (loaded from params.json)
        device: Device to run inference on ('cuda' or 'cpu')
        return_embeddings: Whether to return embeddings
        full_file: If True, process entire audio file using sliding window
        batch_size: Number of segments to process in parallel
        num_workers: Number of worker processes for data loading
    
    Returns:
        List of dictionaries, one per audio file, containing:
            - 'audio_file': path to file
            - 'segments': list of per-segment predictions
            - 'global_prediction': averaged prediction across all segments
            - 'embedding': averaged embedding (if return_embeddings=True)
    """
    # Handle directory path (convert to .pth file path)
    if os.path.isdir(model_path):
        model_dir = model_path
        model_path = os.path.join(model_path, 'model_weights.pth')
    else:
        model_dir = os.path.dirname(model_path)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 1. Load model parameters
    if params is None:
        params_path = os.path.join(model_dir, 'params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
        else:
            raise FileNotFoundError(f"Could not find params.json at {params_path}")
    
    # 2. Initialize model
    model = CNN(
        num_classes=params['num_classes'],
        n_mfcc=params.get('n_mfcc', 20),
        fixed_mfcc_length=params.get('fixed_mfcc_length', 640)
    )
    
    # 3. Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 4. Get class names
    class_names = params.get('class_names', [f'Class_{i}' for i in range(params['num_classes'])])
    
    # 5. Create dataset and dataloader
    dataset = AudioSegmentDataset(
        audio_files=audio_files,
        sample_rate=params.get('sample_rate', 22050),
        n_mfcc=params.get('n_mfcc', 20),
        fixed_mfcc_length=params.get('fixed_mfcc_length', 640),
        segment_duration=30.0,
        hop_duration=30.0,
        full_file=full_file
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    mode_str = "full-file (sliding window)" if full_file else "first 30s only"
    print(f"Processing {len(audio_files)} audio files (mode: {mode_str})...")
    print(f"Total segments: {len(dataset)}, Batch size: {batch_size}")
    
    # 6. Process all segments
    segment_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches", unit="batch")):
            mfcc = batch['mfcc'].to(device)
            
            # Forward pass
            if return_embeddings:
                logits, embeddings = model(mfcc, return_embedding=True)
                embeddings = embeddings.cpu().numpy()
            else:
                logits = model(mfcc, return_embedding=False)
                embeddings = None
            
            # Get probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            # Store results for each sample in batch
            batch_size_actual = mfcc.shape[0]
            for i in range(batch_size_actual):
                result = {
                    'file_index': batch['file_index'][i].item(),
                    'segment_index': batch['segment_index'][i].item(),
                    'audio_file': batch['audio_file'][i],
                    'start_time': batch['start_time'][i].item(),
                    'end_time': batch['end_time'][i].item(),
                    'probabilities': probs[i],
                    'predictions': {class_names[j]: float(probs[i][j]) for j in range(len(class_names))},
                    'embedding': embeddings[i] if embeddings is not None else None
                }
                segment_results.append(result)
    
    # 7. Aggregate results by file
    file_results = []
    for file_idx, audio_file in enumerate(tqdm(audio_files, desc="Aggregating results", unit="file")):
        # Get all segments for this file
        file_segments = [s for s in segment_results if s['file_index'] == file_idx]
        
        if not file_segments:
            tqdm.write(f"Warning: No segments found for {audio_file}")
            continue
        
        # Sort by segment index
        file_segments.sort(key=lambda x: x['segment_index'])
        
        # Compute global prediction (average probabilities)
        all_probs = np.array([s['probabilities'] for s in file_segments])
        avg_probs = np.mean(all_probs, axis=0)
        
        # Get top-3 predictions
        top_3_indices = np.argsort(avg_probs)[-3:][::-1]
        top_3_labels = [class_names[i] for i in top_3_indices]
        top_3_probs = avg_probs[top_3_indices]
        
        # Compute global embedding (average)
        global_embedding = None
        if return_embeddings:
            embeddings_list = [s['embedding'] for s in file_segments if s['embedding'] is not None]
            if embeddings_list:
                global_embedding = np.mean(embeddings_list, axis=0).tolist()
        
        # Build result
        result = {
            'audio_file': audio_file,
            'mode': 'full-file' if full_file else 'first-30s',
            'total_segments': len(file_segments),
            'segments': [
                {
                    'segment_index': s['segment_index'],
                    'start_time': s['start_time'],
                    'end_time': s['end_time'],
                    'predictions': s['predictions']
                }
                for s in file_segments
            ],
            'global_prediction': {
                'top_labels': top_3_labels,
                'probabilities': top_3_probs.tolist(),
                'all_probabilities': {class_names[i]: float(avg_probs[i]) for i in range(len(class_names))}
            }
        }
        
        if global_embedding is not None:
            result['embedding'] = global_embedding
        
        file_results.append(result)
        
        tqdm.write(f"\n[{file_idx+1}/{len(audio_files)}] {audio_file}")
        tqdm.write(f"  Segments: {len(file_segments)}")
        tqdm.write(f"  Top prediction: {top_3_labels[0]} ({top_3_probs[0]:.4f})")
    
    return file_results


# --- Result Saving Functions ---
def save_results_json(
    output_path: str,
    audio_files: List[str],
    labels: Union[List[str], List[List[str]]],
    probabilities: np.ndarray,
    embeddings: np.ndarray = None,
    full_file: bool = False,
    class_names: List[str] = None
):
    """
    Save inference results to a JSON file.
    
    Args:
        output_path: Path to save the JSON file
        audio_files: List of audio file paths
        labels: List of predicted labels (single or top-k)
        probabilities: Array of probabilities
        embeddings: Array of embeddings (optional)
        full_file: Whether full-file mode was used
        class_names: List of class names for probability mapping
    """
    results = {
        'audio_files': [str(f) for f in audio_files],
        'mode': 'full-file' if full_file else 'first-30s',
    }
    
    if full_file:
        # Save top-k predictions with their probabilities
        results['predictions'] = []
        for i, (top_labels, prob_data) in enumerate(zip(labels, probabilities)):
            file_predictions = [
                {'label': label, 'probability': float(prob_data[j][1])} 
                for j, label in enumerate(top_labels)
            ]
            results['predictions'].append({
                'audio_file': str(audio_files[i]),
                'top_predictions': file_predictions
            })
    else:
        # Save single prediction with all class probabilities
        results['predictions'] = []
        for i, (audio_file, label, probs) in enumerate(zip(audio_files, labels, probabilities)):
            pred_result = {
                'audio_file': str(audio_file),
                'predicted_label': label,
                'confidence': float(probs.max())
            }
            
            # Add all class probabilities if class names are available
            if class_names:
                pred_result['all_probabilities'] = {
                    class_name: float(prob) 
                    for class_name, prob in zip(class_names, probs)
                }
            
            results['predictions'].append(pred_result)
    
    # Add embeddings if available
    if embeddings is not None:
        results['embeddings'] = embeddings.tolist()
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def save_results_csv(
    output_path: str,
    audio_files: List[str],
    labels: Union[List[str], List[List[str]]],
    probabilities: np.ndarray,
    embeddings: np.ndarray = None,
    full_file: bool = False
):
    """
    Save inference results to a CSV file.
    
    Args:
        output_path: Path to save the CSV file
        audio_files: List of audio file paths
        labels: List of predicted labels (single or top-k)
        probabilities: Array of probabilities
        embeddings: Array of embeddings (optional)
        full_file: Whether full-file mode was used
    """
    with open(output_path, 'w', newline='') as csvfile:
        if full_file:
            # Full-file mode: show top-k predictions
            fieldnames = ['audio_file', 'rank', 'predicted_label', 'probability']
            if embeddings is not None:
                fieldnames.append('embedding')
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, (audio_file, top_labels, prob_data) in enumerate(zip(audio_files, labels, probabilities)):
                for j, (label, prob_row) in enumerate(zip(top_labels, prob_data)):
                    prob = prob_row[1]  # Second column is the probability
                    row = {
                        'audio_file': str(audio_file),
                        'rank': j + 1,
                        'predicted_label': label,
                        'probability': f"{prob:.6f}"
                    }
                    
                    # Add embedding only for the top prediction
                    if embeddings is not None and j == 0:
                        embedding_str = ','.join([f"{x:.6f}" for x in embeddings[i]])
                        row['embedding'] = f"[{embedding_str}]"
                    elif embeddings is not None:
                        row['embedding'] = ''
                    
                    writer.writerow(row)
        else:
            # Standard mode: single prediction
            fieldnames = ['audio_file', 'predicted_label', 'confidence']
            if embeddings is not None:
                fieldnames.append('embedding')
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, (audio_file, label, probs) in enumerate(zip(audio_files, labels, probabilities)):
                row = {
                    'audio_file': str(audio_file),
                    'predicted_label': label,
                    'confidence': f"{probs.max():.6f}"
                }
                
                if embeddings is not None:
                    embedding_str = ','.join([f"{x:.6f}" for x in embeddings[i]])
                    row['embedding'] = f"[{embedding_str}]"
                
                writer.writerow(row)
    
    print(f"\nResults saved to: {output_path}")


def save_results_jsonl(
    output_path: str,
    results: List[Dict],
    include_embeddings: bool = True
):
    """
    Save inference results to a JSONL file (one JSON object per line).
    Each line contains all information for one audio file.
    
    Args:
        output_path: Path to save the JSONL file
        results: List of result dictionaries from run_inference_streaming
        include_embeddings: Whether to include embeddings in output
    """
    with open(output_path, 'w') as f:
        for result in results:
            # Create a copy to avoid modifying original
            output_result = result.copy()
            
            # Remove embedding if not requested
            if not include_embeddings and 'embedding' in output_result:
                del output_result['embedding']
            
            # Write as single-line JSON
            f.write(json.dumps(output_result) + '\n')
    
    print(f"\nResults saved to: {output_path} ({len(results)} files)")


# --- Main Execution ---
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on audio files using a trained model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model weights (.pth file) or directory containing latest model')
    parser.add_argument('--audio', nargs='+', required=True,
                        help='Path(s) to audio file(s) or directory containing audio files')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu), defaults to auto-detect')
    parser.add_argument('--no-embeddings', action='store_true',
                        help='Do not return embeddings')
    parser.add_argument('--full', action='store_true',
                        help='Process entire audio file using sliding window (default: first 30s only)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results file (optional). Format is auto-detected from extension (.json, .jsonl, or .csv)')
    parser.add_argument('--format', type=str, choices=['json', 'jsonl', 'csv'], default=None,
                        help='Output format (json/jsonl/csv). If not specified, will be inferred from --output extension')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for processing segments (default: 8)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for data loading (default: 4)')
    parser.add_argument('--use-legacy', action='store_true',
                        help='Use legacy inference (slower, not recommended)')
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Handle model path (allow specifying model directory)
    model_path = args.model
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, 'model_weights.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Collect audio files
    audio_files = []
    for path in args.audio:
        path_obj = Path(path)
        if path_obj.is_file():
            audio_files.append(str(path_obj.absolute()))
        elif path_obj.is_dir():
            # Find all audio files in directory
            print(f"Scanning directory: {path}")
            for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.ogg']:
                audio_files.extend([str(p.absolute()) for p in path_obj.glob(ext)])
                audio_files.extend([str(p.absolute()) for p in path_obj.glob(ext.upper())])
            # Also search recursively
            for ext in ['**/*.mp3', '**/*.wav', '**/*.flac', '**/*.m4a', '**/*.ogg']:
                audio_files.extend([str(p.absolute()) for p in path_obj.glob(ext)])
                audio_files.extend([str(p.absolute()) for p in path_obj.glob(ext.upper())])
        else:
            print(f"Warning: {path} does not exist, skipping")
    
    # Remove duplicates while preserving order
    audio_files = list(dict.fromkeys(audio_files))
    
    if not audio_files:
        raise ValueError("No valid audio files found!")
    
    print(f"Found {len(audio_files)} audio files\n")
    
    # Run inference using streaming DataLoader (new default) or legacy method
    if args.use_legacy:
        print("Using legacy inference method...")
        # Determine top_k based on --full flag
        top_k = 3 if args.full else 1
        
        labels, probabilities, embeddings = run_inference(
            model_path=model_path,
            audio_files=audio_files,
            device=args.device,
            return_embeddings=not args.no_embeddings,
            full_file=args.full,
            top_k=top_k
        )
        
        # Display results (legacy format)
        print("\n" + "="*80)
        print("INFERENCE RESULTS")
        print("="*80)
        
        if args.full:
            for i, audio_file in enumerate(audio_files):
                print(f"\n[{i+1}/{len(audio_files)}] {audio_file}")
                print(f"  Top 3 Predictions:")
                top_labels = labels[i]
                top_probs_data = probabilities[i]
                for j, (label, prob_row) in enumerate(zip(top_labels, top_probs_data)):
                    prob = prob_row[1]
                    print(f"    {j+1}. {label}: {prob:.4f}")
                if embeddings is not None:
                    print(f"  Embedding Shape: {embeddings[i].shape}")
        else:
            for i, (audio_file, label, probs) in enumerate(zip(audio_files, labels, probabilities)):
                print(f"\n[{i+1}/{len(audio_files)}] {audio_file}")
                print(f"  Predicted Genre: {label}")
                print(f"  Confidence: {probs.max():.4f}")
        
        # For legacy mode, only support old JSON/CSV formats
        if args.output:
            output_format = args.format or ('csv' if args.output.endswith('.csv') else 'json')
            model_dir = os.path.dirname(model_path)
            params_path = os.path.join(model_dir, 'params.json')
            class_names = None
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params_data = json.load(f)
                    class_names = params_data.get('class_names')
            
            if output_format == 'csv':
                save_results_csv(args.output, audio_files, labels, probabilities,
                               embeddings if not args.no_embeddings else None, args.full)
            else:
                save_results_json(args.output, audio_files, labels, probabilities,
                                embeddings if not args.no_embeddings else None, args.full, class_names)
    else:
        # New streaming inference (default)
        print("Using streaming inference with DataLoader...")
        results = run_inference_streaming(
            model_path=model_path,
            audio_files=audio_files,
            device=args.device,
            return_embeddings=not args.no_embeddings,
            full_file=args.full,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Display results summary
        print("\n" + "="*80)
        print("INFERENCE RESULTS SUMMARY")
        print("="*80)
        for i, result in enumerate(results):
            print(f"\n[{i+1}/{len(results)}] {Path(result['audio_file']).name}")
            print(f"  Top Genre: {result['global_prediction']['top_labels'][0]} "
                  f"({result['global_prediction']['probabilities'][0]:.4f})")
            if len(result['global_prediction']['top_labels']) > 1:
                print(f"  2nd: {result['global_prediction']['top_labels'][1]} "
                      f"({result['global_prediction']['probabilities'][1]:.4f})")
                print(f"  3rd: {result['global_prediction']['top_labels'][2]} "
                      f"({result['global_prediction']['probabilities'][2]:.4f})")
            print(f"  Segments: {result['total_segments']}")
        
        # Save results if requested
        if args.output:
            # Determine output format
            output_format = args.format
            if output_format is None:
                # Infer from file extension
                if args.output.lower().endswith('.jsonl'):
                    output_format = 'jsonl'
                elif args.output.lower().endswith('.json'):
                    output_format = 'json'
                elif args.output.lower().endswith('.csv'):
                    output_format = 'csv'
                else:
                    # Default to JSONL for streaming
                    output_format = 'jsonl'
                    print(f"Warning: Could not infer format from extension, defaulting to JSONL")
            
            # Save based on format
            if output_format == 'jsonl':
                save_results_jsonl(
                    args.output,
                    results,
                    include_embeddings=not args.no_embeddings
                )
            elif output_format == 'json':
                # Convert streaming results to old JSON format
                with open(args.output, 'w') as f:
                    output_data = {
                        'audio_files': [r['audio_file'] for r in results],
                        'mode': results[0]['mode'] if results else 'unknown',
                        'results': results
                    }
                    json.dump(output_data, f, indent=2)
                print(f"\nResults saved to: {args.output}")
            elif output_format == 'csv':
                # Convert to CSV format - simplified version with global predictions only
                with open(args.output, 'w', newline='') as csvfile:
                    fieldnames = ['audio_file', 'segments', 'top_genre', 'probability', 
                                'second_genre', 'second_prob', 'third_genre', 'third_prob']
                    if not args.no_embeddings:
                        fieldnames.append('embedding')
                    
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for result in results:
                        gp = result['global_prediction']
                        row = {
                            'audio_file': result['audio_file'],
                            'segments': result['total_segments'],
                            'top_genre': gp['top_labels'][0],
                            'probability': f"{gp['probabilities'][0]:.6f}",
                            'second_genre': gp['top_labels'][1] if len(gp['top_labels']) > 1 else '',
                            'second_prob': f"{gp['probabilities'][1]:.6f}" if len(gp['probabilities']) > 1 else '',
                            'third_genre': gp['top_labels'][2] if len(gp['top_labels']) > 2 else '',
                            'third_prob': f"{gp['probabilities'][2]:.6f}" if len(gp['probabilities']) > 2 else '',
                        }
                        
                        if not args.no_embeddings and 'embedding' in result:
                            emb = result['embedding']
                            embedding_str = ','.join([f"{x:.6f}" for x in emb])
                            row['embedding'] = f"[{embedding_str}]"
                        
                        writer.writerow(row)
                print(f"\nResults saved to: {args.output}")
    
    print("\n" + "="*80)

