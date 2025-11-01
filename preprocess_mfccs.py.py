# preprocess_mfccs_multiprocessing.py
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import multiprocessing  # Import the multiprocessing library

# Add notebooks directory to path to import utils
sys.path.append(os.path.join(os.getcwd(), 'notebooks'))
import utils as fma_utils

# --- Configuration (remains the same) ---
FMA_METADATA_DIR = './data/fma_metadata'
FMA_AUDIO_DIR = './data/fma_small'
MFCC_OUTPUT_DIR = './data/fma_mfccs_precomputed' 

SAMPLE_RATE = 22050
N_MFCC = 20
FIXED_MFCC_LENGTH = 640

# 1. DEFINE THE WORKER FUNCTION
# This function contains the logic to process ONE audio file.
# It will be executed by each worker process in the pool.
def process_track(track_id):
    """
    Loads an audio file, computes its MFCC, and saves it to a .npy file.
    Takes a track_id as input.
    """
    try:
        audio_path = fma_utils.get_audio_path(FMA_AUDIO_DIR, track_id)
        mfcc_path = os.path.join(MFCC_OUTPUT_DIR, f"{track_id}.npy")

        # Avoid re-computing if it already exists
        if os.path.exists(mfcc_path):
            return f"Skipped {track_id}"

        signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=30)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

        # Pad or truncate
        if mfcc.shape[1] > FIXED_MFCC_LENGTH:
            mfcc = mfcc[:, :FIXED_MFCC_LENGTH]
        else:
            pad_width = FIXED_MFCC_LENGTH - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # Save the result
        np.save(mfcc_path, mfcc)
        return f"Processed {track_id}"
        
    except Exception as e:
        # Return an error message if something goes wrong
        return f"ERROR processing {track_id}: {e}"

# 2. USE THE `if __name__ == '__main__':` GUARD
# This is essential for multiprocessing to work correctly on all platforms.
if __name__ == '__main__':
    # Create the output directory if it doesn't exist
    os.makedirs(MFCC_OUTPUT_DIR, exist_ok=True)

    # Load metadata to get all track IDs for the small subset
    tracks_path = os.path.join(FMA_METADATA_DIR, 'tracks.csv')
    tracks = fma_utils.load(tracks_path)
    track_ids_to_process = tracks[tracks[('set', 'subset')] == 'small'].index.tolist()
    
    total_tracks = len(track_ids_to_process)
    print(f"Found {total_tracks} tracks in the small subset to process.")
    print(f"Saving MFCCs to: {MFCC_OUTPUT_DIR}")

    # 3. SETUP AND RUN THE MULTIPROCESSING POOL ⚙️
    # Use most available CPU cores, but leave one free for system stability
    num_workers = max(1, multiprocessing.cpu_count()//2)
    print(f"Using {num_workers} worker processes.")

    # The 'with' statement ensures the pool is properly closed
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use pool.imap_unordered for efficiency and to get a live progress bar.
        # It processes the items and yields results as they complete.
        results = list(tqdm(pool.imap_unordered(process_track, track_ids_to_process), total=total_tracks))

    print("\nPreprocessing complete! 🚀")