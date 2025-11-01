#!/usr/bin/env python
"""
Quick test script to verify inference.py works correctly.
This script uses an audio file from the FMA-small dataset to test inference.
"""

import os
import sys
from inference import run_inference
import numpy as np

def test_inference():
    # Find the most recent model
    training_outputs_dir = './training_outputs'
    
    if not os.path.exists(training_outputs_dir):
        print("Error: training_outputs directory not found")
        return
    
    # Get all subdirectories (timestamps)
    subdirs = [d for d in os.listdir(training_outputs_dir) if os.path.isdir(os.path.join(training_outputs_dir, d))]
    if not subdirs:
        print("Error: No trained models found in training_outputs")
        return
    
    # Use the most recent one
    latest_model = sorted(subdirs)[-1]
    model_path = os.path.join(training_outputs_dir, latest_model)
    
    print(f"Using model from: {model_path}")
    
    # Find an audio file to test with
    # Try to find any MP3 file in the fma_small dataset
    audio_file = None
    fma_audio_dir = './data/fma_small'
    
    if os.path.exists(fma_audio_dir):
        # Look for any MP3 file
        for root, dirs, files in os.walk(fma_audio_dir):
            for file in files:
                if file.endswith('.mp3'):
                    audio_file = os.path.join(root, file)
                    break
            if audio_file:
                break
    
    if not audio_file:
        print("Error: Could not find any audio files in ./data/fma_small")
        print("\nTo test inference, you need at least one audio file.")
        print("Usage example:")
        print('  python inference.py --model training_outputs/2025-10-22_22-38-20 --audio path/to/song.mp3')
        return
    
    print(f"Testing with audio file: {audio_file}")
    print("\n" + "="*80)
    print("RUNNING INFERENCE TEST")
    print("="*80 + "\n")
    
    # Run inference
    try:
        labels, probabilities, embeddings = run_inference(
            model_path=model_path,
            audio_files=[audio_file],
            return_embeddings=True
        )
        
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"Audio file: {audio_file}")
        print(f"Predicted genre: {labels[0]}")
        print(f"Confidence: {probabilities[0].max():.4f}")
        print(f"Embedding shape: {embeddings[0].shape}")
        print(f"Embedding norm: {np.linalg.norm(embeddings[0]):.4f}")
        print("\nAll class probabilities:")
        for i, prob in enumerate(probabilities[0]):
            print(f"  Class {i}: {prob:.4f}")
        print("\nTest completed successfully! ✅")
        
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_inference()

