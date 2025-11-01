import torchaudio
import os
import sys

# Add notebooks directory to path to find your utils
sys.path.append(os.path.join(os.getcwd(), 'notebooks'))
import utils as fma_utils

# Path to the second audio file in the dataset (track ID 5)
# Using ID 5 because it's typically a standard, clean file.
audio_path = fma_utils.get_audio_path('./data/fma_small', 5)

print(f"Attempting to load: {audio_path}")

if not os.path.exists(audio_path):
    print("---" * 10)
    print("ERROR: Audio file not found. Please check the path in the script.")
    print("---" * 10)
else:
    try:
        waveform, sr = torchaudio.load(audio_path)
        print("---" * 10)
        print("✅ SUCCESS! Audio loaded without crashing.")
        print(f"Waveform shape: {waveform.shape}")
        print(f"Sample rate: {sr}")
        print("---" * 10)
    except Exception as e:
        print(f"Caught a Python exception: {e}")