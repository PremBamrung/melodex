# Melodex

A deep learning-based system for analyzing and visualizing music. Melodex helps you study relationships between tracks, analyze the evolution of music from artists, and explore musical patterns through 2D UMAP visualizations, with genre classification as a supporting feature. Currently uses the FMA (Free Music Archive) dataset as one example, with plans to support additional datasets.

> **Name Origin:** Melodex combines "Melody" + "Pokedex" - like a Pokédex catalogs and analyzes Pokémon, Melodex helps catalog and analyze music.

## 📋 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)

## 🎵 Overview

Melodex is a system designed to help you analyze and understand music through interactive visualizations. The primary goal is to study relationships between tracks, analyze the evolution of music from artists over time, and explore musical patterns through 2D UMAP visualizations.

### Key Features:
- **2D UMAP Visualization:** Plot music in a 2D space to analyze relationships, clusters, and patterns between tracks and artists
- **Music Analysis:** Study how music evolves over time, compare tracks from the same artist, and identify musical relationships
- **Genre Classification:** Automatically classify music genres as a supporting feature for analysis (currently supporting 8 genres)
- **MFCC (Mel-Frequency Cepstral Coefficients)** features for audio representation
- **Convolutional Neural Network (CNN)** for extracting meaningful music embeddings
- **Multi-Dataset Support:** Currently demonstrates with FMA dataset, designed to work with additional datasets

## 🔧 Installation

### Requirements
```bash
pip install torch torchvision torchaudio
pip install librosa numpy pandas
pip install scikit-learn matplotlib
```

### Clone and Setup
```bash
git clone <repository-url>
cd melodex
```

## 📁 Project Structure

```
melodex/
├── data/                          # Dataset directory
│   ├── fma_small/                 # Audio files
│   ├── fma_metadata/              # Metadata and labels
│   └── fma_mfccs_precomputed/     # Preprocessed MFCC features
├── training_outputs/              # Model checkpoints and results
│   └── YYYY-MM-DD_HH-MM-SS/      # Timestamped training runs
│       ├── model_weights.pth      # Trained model weights
│       ├── params.json            # Model parameters
│       └── classification_report.txt
├── notebooks/                     # Jupyter notebooks for analysis
├── docs/                          # Research papers and documentation
├── train.py                       # Training script
├── inference.py                   # Inference script (main)
├── preprocess_mfccs.py.py        # MFCC preprocessing
└── README.md                      # This file
```

## 🚀 Usage

### Training

Train a new model on your dataset (example using FMA):

```bash
python train.py --data_dir data/fma_small --epochs 50 --batch_size 32
```

Training outputs will be saved to `training_outputs/` with timestamp.

### Inference

The inference script provides powerful capabilities for classifying audio files.

#### Basic Usage

**Classify a single audio file:**
```bash
python inference.py --model training_outputs/latest --audio song.mp3
```

**Classify multiple files:**
```bash
python inference.py --model training_outputs/latest --audio song1.mp3 song2.mp3 song3.wav
```

**Classify all audio files in a directory (recursive):**
```bash
python inference.py --model training_outputs/latest --audio /path/to/music_folder/
```

#### Saving Results

**Save results as JSONL (recommended for large datasets):**
```bash
python inference.py --model training_outputs/latest --audio music/ --output results.jsonl
```

**Save results as JSON:**
```bash
python inference.py --model training_outputs/latest --audio music/ --output results.json
```

**Save results as CSV:**
```bash
python inference.py --model training_outputs/latest --audio music/ --output results.csv
```

**Explicitly specify format:**
```bash
python inference.py --model training_outputs/latest --audio music/ --output results --format jsonl
```

#### Advanced Options

**Process full audio file with sliding window (instead of first 30s):**
```bash
python inference.py --model training_outputs/latest --audio music/ --full --output results.json
```

**Exclude embeddings from output:**
```bash
python inference.py --model training_outputs/latest --audio music/ --no-embeddings --output results.csv
```

**Specify GPU/CPU:**
```bash
python inference.py --model training_outputs/latest --audio music/ --device cuda
# or
python inference.py --model training_outputs/latest --audio music/ --device cpu
```

**Performance tuning (batch size and workers):**
```bash
# For GPU with lots of memory
python inference.py --model model --audio music/ --batch-size 16 --num-workers 8

# For CPU or limited memory
python inference.py --model model --audio music/ --batch-size 4 --num-workers 2
```

#### Complete Example

```bash
# Process an entire directory, use full-file analysis, save to CSV with embeddings
python inference.py \
  --model training_outputs/2025-10-22_22-38-20 \
  --audio data/fma_small/000/ \
  --full \
  --output genre_predictions.csv \
  --device cuda
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to model weights (.pth) or directory | **Required** |
| `--audio` | Path(s) to audio file(s) or directory | **Required** |
| `--device` | Device to use (cuda/cpu) | Auto-detect |
| `--output` | Path to save results file | None |
| `--format` | Output format (json/jsonl/csv) | Auto from extension |
| `--full` | Process full file with sliding window | False (first 30s) |
| `--no-embeddings` | Don't include embeddings in output | False (include) |
| `--batch-size` | Number of segments to process in parallel | 8 |
| `--num-workers` | Number of data loading workers | 4 |
| `--use-legacy` | Use old inference method (slower) | False |

### Supported Audio Formats

- MP3 (`.mp3`)
- WAV (`.wav`)
- FLAC (`.flac`)
- M4A (`.m4a`)
- OGG (`.ogg`)

Both uppercase and lowercase extensions are supported.

## 📊 Output Formats

### JSONL Output (Recommended - New!)

**JSONL (JSON Lines)** format provides detailed per-segment predictions. Each line is a complete JSON object for one audio file.

```json
{"audio_file": "/path/to/song.mp3", "mode": "full-file", "total_segments": 8, "segments": [{"segment_index": 0, "start_time": 0.0, "end_time": 30.0, "predictions": {"Pop": 0.65, "Rock": 0.20, ...}}, ...], "global_prediction": {"top_labels": ["Pop", "Rock", "Electronic"], "probabilities": [0.62, 0.18, 0.12], "all_probabilities": {...}}, "embedding": [0.0, 4.97, ...]}
```

**Benefits:**
- ✅ Per-segment predictions (see genre changes over time)
- ✅ Memory-efficient streaming for large datasets
- ✅ Parallel processing with PyTorch DataLoader
- ✅ Easy to append new results
- ✅ One file = one line (easy to process)

**See [JSONL_FORMAT_GUIDE.md](JSONL_FORMAT_GUIDE.md) for detailed documentation.**

### JSON Output (Standard Mode)

```json
{
  "audio_files": ["song1.mp3", "song2.mp3"],
  "mode": "first-30s",
  "predictions": [
    {
      "audio_file": "song1.mp3",
      "predicted_label": "rock",
      "confidence": 0.8532,
      "all_probabilities": {
        "rock": 0.8532,
        "metal": 0.0821,
        "pop": 0.0412,
        "electronic": 0.0235,
        ...
      }
    }
  ],
  "embeddings": [
    [0.234, 0.567, 0.123, ...],
    [0.345, 0.678, 0.234, ...]
  ]
}
```

### JSON Output (Full-File Mode)

```json
{
  "audio_files": ["long_song.mp3"],
  "mode": "full-file",
  "predictions": [
    {
      "audio_file": "long_song.mp3",
      "top_predictions": [
        {"label": "rock", "probability": 0.7234},
        {"label": "metal", "probability": 0.1523},
        {"label": "alternative", "probability": 0.0821}
      ]
    }
  ],
  "embeddings": [[0.234, 0.567, ...]]
}
```

### CSV Output (Standard Mode)

| audio_file | predicted_label | confidence | embedding |
|------------|----------------|------------|-----------|
| song1.mp3 | rock | 0.8532 | [0.234,0.567,0.123,...] |
| song2.mp3 | pop | 0.9123 | [0.345,0.678,0.234,...] |

### CSV Output (Full-File Mode)

| audio_file | rank | predicted_label | probability | embedding |
|------------|------|----------------|-------------|-----------|
| song1.mp3 | 1 | rock | 0.7234 | [0.234,0.567,...] |
| song1.mp3 | 2 | metal | 0.1523 | |
| song1.mp3 | 3 | alternative | 0.0821 | |

**Note:** In full-file mode CSV, embeddings are only included in the first (top) prediction row.

## 🏗️ Model Architecture

The CNN model extracts meaningful music embeddings that can be used for both visualization and classification:

**Convolutional Blocks:**
- 4 Conv2D blocks with increasing filters (32 → 64 → 128 → 256)
- Each block has: Conv2D → ReLU → BatchNorm → MaxPool → Dropout(0.25)
- Final pooling: MaxPool2d(2, 4)

**Fully Connected Layers:**
- Flatten layer
- Linear(flattened_size → 512) → ReLU → Dropout(0.5)  # **512-dimensional embeddings for UMAP visualization**
- Linear(512 → num_classes)  # Classification head (genre classification as subtask)

**Input:** MFCC features (1, 20, 640)
- 1 channel
- 20 MFCC coefficients  
- 640 time frames (~30 seconds at 22050 Hz)

**Output:** The model produces 512-dimensional embeddings that can be visualized in 2D using UMAP, enabling analysis of the music space and study of relationships between tracks, artists, and temporal evolution.

## 📦 Dataset

Melodex is designed to work with multiple music datasets. Currently, we demonstrate the system using the [FMA (Free Music Archive) dataset](https://github.com/mdeff/fma) as one example:

- **FMA Small:** 8,000 tracks of 30s, 8 balanced genres
- **Genres:** Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock

The project architecture supports integrating additional datasets for broader music analysis and relationship studies.

### FMA Data Loading

When using the FMA dataset, organize it as:
```
data/
├── fma_small/           # Audio files
│   ├── 000/
│   ├── 001/
│   └── ...
└── fma_metadata/        # Metadata
    ├── tracks.csv
    └── genres.csv
```

## 🎯 Features

### Core Capabilities
- ✅ **2D UMAP Visualization:** Plot music embeddings in 2D space to analyze relationships, clusters, and patterns between tracks and artists
- ✅ **Music Embeddings:** Extract 512-dimensional embeddings for each track for similarity analysis, relationship studies, and visualization
- ✅ **Evolution Analysis:** Study how music evolves from artists over time by comparing tracks chronologically
- ✅ **Relationship Analysis:** Identify musical relationships and patterns through embedding space analysis
- ✅ **Genre Classification:** Classify music genres as a supporting feature for analysis (currently 8 genres)
- ✅ **Multi-Dataset Support:** Architecture designed to work with multiple music datasets

### Technical Features
- ✅ **Streaming Inference:** Efficient PyTorch DataLoader with configurable batch size and workers
- ✅ **JSONL Format:** Detailed per-segment predictions with temporal analysis
- ✅ **Batch Inference:** Process multiple files or entire directories
- ✅ **Recursive Directory Search:** Automatically find audio files in subdirectories
- ✅ **Multiple Output Formats:** Save results as JSONL, JSON, or CSV
- ✅ **Full-File Analysis:** Process songs longer than 30s using sliding window
- ✅ **GPU Acceleration:** Automatic CUDA detection and support
- ✅ **Flexible Input:** Single files, multiple files, or directories
- ✅ **Memory Efficient:** Stream large datasets without loading everything into memory

## 📈 Performance

Model performance metrics are saved in each training run under:
```
training_outputs/YYYY-MM-DD_HH-MM-SS/classification_report.txt
```

## 🔬 Research

See the `docs/papers/` directory for research papers on music genre classification, audio embeddings, and music visualization topics.

## 💡 Tips

1. **For Analysis:** Extract embeddings and use UMAP to create 2D plots for analyzing relationships and patterns in your music collection
2. **For Artist Evolution Studies:** Use `--full` mode with JSONL format to analyze how tracks from the same artist evolve over time
3. **For Speed:** Use standard mode (first 30s) with larger batch sizes
4. **For Relationship Analysis:** Enable embeddings and use JSONL to track genre changes over time, or visualize with UMAP to study musical relationships
5. **For Large Datasets:** Use JSONL format for memory-efficient streaming when analyzing large collections
6. **For Performance:** Tune `--batch-size` and `--num-workers` based on your hardware
7. **For Small Files:** CSV format is simplest for quick analysis in Excel/pandas

## 🐛 Troubleshooting

**CUDA Out of Memory:**
```bash
# Use CPU instead
python inference.py --model model --audio music/ --device cpu
```

**Audio Loading Warnings:**
```bash
# Install ffmpeg for better audio format support
sudo apt-get install ffmpeg  # Linux
brew install ffmpeg          # macOS
```

**PySoundFile Warning:**
The warning "PySoundFile failed. Trying audioread instead" is suppressed by default and doesn't affect functionality.

