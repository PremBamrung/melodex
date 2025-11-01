# JSONL Format Guide - Enhanced Inference Output

## Overview

The new **JSONL (JSON Lines)** format provides a more detailed and scalable way to save inference results. Each line in the file is a complete JSON object containing all information for one audio file.

## Key Improvements

✅ **Per-segment predictions** - See how genre classification changes throughout the song  
✅ **Streaming-friendly** - Process one file at a time without loading everything into memory  
✅ **Appendable** - Add new results without rewriting the entire file  
✅ **Parallel processing** - Uses PyTorch DataLoader with configurable batch size and workers  
✅ **Better for large datasets** - Each line is independent and can be processed separately

---

## JSONL Structure

Each line contains a JSON object with the following structure:

```json
{
  "audio_file": "/path/to/song.mp3",
  "mode": "full-file",
  "total_segments": 8,
  "segments": [
    {
      "segment_index": 0,
      "start_time": 0.0,
      "end_time": 30.0,
      "predictions": {
        "Electronic": 0.0234,
        "Experimental": 0.0156,
        "Folk": 0.0423,
        "Hip-Hop": 0.0089,
        "Instrumental": 0.0167,
        "International": 0.0199,
        "Pop": 0.6532,
        "Rock": 0.2200
      }
    },
    {
      "segment_index": 1,
      "start_time": 30.0,
      "end_time": 60.0,
      "predictions": {
        "Electronic": 0.0412,
        "Experimental": 0.0234,
        "Folk": 0.0521,
        "Hip-Hop": 0.0156,
        "Instrumental": 0.0289,
        "International": 0.0331,
        "Pop": 0.7234,
        "Rock": 0.0823
      }
    }
    // ... more segments
  ],
  "global_prediction": {
    "top_labels": ["Pop", "Rock", "Electronic"],
    "probabilities": [0.6234, 0.1823, 0.0823],
    "all_probabilities": {
      "Electronic": 0.0823,
      "Experimental": 0.0195,
      "Folk": 0.0472,
      "Hip-Hop": 0.0122,
      "Instrumental": 0.0228,
      "International": 0.0265,
      "Pop": 0.6234,
      "Rock": 0.1823
    }
  },
  "embedding": [0.0, 0.0, 4.97, 0.0, ..., 1.53, 0.0]
}
```

---

## Field Descriptions

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `audio_file` | string | Full path to the audio file |
| `mode` | string | Processing mode: `"full-file"` or `"first-30s"` |
| `total_segments` | integer | Number of segments processed for this file |
| `segments` | array | Array of segment objects (see below) |
| `global_prediction` | object | Averaged prediction across all segments |
| `embedding` | array | 512-dimensional feature vector (optional) |

### Segment Object

| Field | Type | Description |
|-------|------|-------------|
| `segment_index` | integer | Index of this segment (0-based) |
| `start_time` | float | Start time in seconds |
| `end_time` | float | End time in seconds |
| `predictions` | object | Genre probabilities for this segment |

**Note:** `predictions` is a dictionary mapping genre names to their probabilities (0.0 to 1.0). All probabilities sum to 1.0.

### Global Prediction Object

| Field | Type | Description |
|-------|------|-------------|
| `top_labels` | array | Top 3 predicted genres (highest probability first) |
| `probabilities` | array | Probabilities for the top 3 genres |
| `all_probabilities` | object | Probabilities for all genres (averaged across segments) |

---

## Usage Examples

### Generate JSONL Output

```bash
# Default output format (JSONL)
python inference.py \
  --model training_outputs/latest \
  --audio music/ \
  --full \
  --output results.jsonl

# Explicitly specify JSONL format
python inference.py \
  --model training_outputs/latest \
  --audio music/ \
  --full \
  --output results.txt \
  --format jsonl

# Without embeddings (smaller file size)
python inference.py \
  --model training_outputs/latest \
  --audio music/ \
  --full \
  --no-embeddings \
  --output results.jsonl

# Adjust batch size and workers for performance
python inference.py \
  --model training_outputs/latest \
  --audio music/ \
  --full \
  --batch-size 16 \
  --num-workers 8 \
  --output results.jsonl
```

### Reading JSONL in Python

```python
import json

# Read JSONL file
results = []
with open('results.jsonl', 'r') as f:
    for line in f:
        result = json.loads(line)
        results.append(result)

# Access data for first file
first_result = results[0]
print(f"File: {first_result['audio_file']}")
print(f"Top genre: {first_result['global_prediction']['top_labels'][0]}")
print(f"Segments: {first_result['total_segments']}")

# Iterate through segments
for segment in first_result['segments']:
    print(f"Segment {segment['segment_index']}: "
          f"{segment['start_time']:.1f}s - {segment['end_time']:.1f}s")
    # Get top prediction for this segment
    top_genre = max(segment['predictions'].items(), key=lambda x: x[1])
    print(f"  Top genre: {top_genre[0]} ({top_genre[1]:.4f})")

# Extract embeddings
if 'embedding' in first_result:
    import numpy as np
    embedding = np.array(first_result['embedding'])
    print(f"Embedding shape: {embedding.shape}")
```

### Streaming Processing (Memory Efficient)

```python
import json

# Process one file at a time without loading all into memory
with open('results.jsonl', 'r') as f:
    for line in f:
        result = json.loads(line)
        
        # Process this file's results
        audio_file = result['audio_file']
        top_genre = result['global_prediction']['top_labels'][0]
        confidence = result['global_prediction']['probabilities'][0]
        
        print(f"{audio_file}: {top_genre} ({confidence:.2%})")
```

### Convert JSONL to Pandas DataFrame

```python
import json
import pandas as pd

# Load JSONL
results = []
with open('results.jsonl', 'r') as f:
    for line in f:
        results.append(json.loads(line))

# Create DataFrame with global predictions
df = pd.DataFrame([
    {
        'audio_file': r['audio_file'],
        'segments': r['total_segments'],
        'top_genre': r['global_prediction']['top_labels'][0],
        'confidence': r['global_prediction']['probabilities'][0],
        'second_genre': r['global_prediction']['top_labels'][1],
        'third_genre': r['global_prediction']['top_labels'][2],
    }
    for r in results
])

print(df.head())
```

### Analyze Genre Changes Over Time

```python
import json
import matplotlib.pyplot as plt

# Load a single song's results
with open('results.jsonl', 'r') as f:
    song = json.loads(f.readline())

# Extract genre probabilities over time
segments = song['segments']
genres = list(segments[0]['predictions'].keys())

# Create time series data
times = [(s['start_time'] + s['end_time']) / 2 for s in segments]
genre_series = {genre: [] for genre in genres}

for segment in segments:
    for genre in genres:
        genre_series[genre].append(segment['predictions'][genre])

# Plot
plt.figure(figsize=(12, 6))
for genre in genres:
    plt.plot(times, genre_series[genre], label=genre, marker='o')

plt.xlabel('Time (seconds)')
plt.ylabel('Probability')
plt.title(f'Genre Classification Over Time: {song["audio_file"]}')
plt.legend()
plt.grid(True)
plt.savefig('genre_timeline.png')
```

---

## Command-Line Arguments

### New Arguments for Streaming Inference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--format` | choice | auto | Output format: `json`, `jsonl`, or `csv` |
| `--batch-size` | int | 8 | Number of segments to process in parallel |
| `--num-workers` | int | 4 | Number of worker processes for data loading |
| `--use-legacy` | flag | False | Use old inference method (not recommended) |

### Performance Tuning

**For CPU:**
```bash
--batch-size 4 --num-workers 2
```

**For GPU with lots of RAM:**
```bash
--batch-size 32 --num-workers 8
```

**For very large datasets:**
```bash
--batch-size 16 --num-workers 6
```

---

## File Size Comparison

For 1000 songs with full-file analysis:

| Format | Without Embeddings | With Embeddings |
|--------|-------------------|-----------------|
| **JSONL** | ~2-5 MB | ~40-50 MB |
| **JSON** | ~2-5 MB | ~40-50 MB |
| **CSV** | ~500 KB | ~40 MB |

**Note:** JSONL and JSON have similar file sizes, but JSONL is much easier to stream and append to.

---

## Benefits Over Previous Formats

### 1. Detailed Temporal Analysis
See how genre predictions change throughout the song:
```python
# Find songs where genre changes significantly
for result in results:
    segment_preds = [s['predictions'] for s in result['segments']]
    # Analyze variance in predictions...
```

### 2. Memory Efficient Processing
Process massive datasets without loading everything:
```python
# Process 100,000 songs without memory issues
with open('huge_dataset.jsonl', 'r') as f:
    for line in f:
        result = json.loads(line)
        # Process one at a time
```

### 3. Easy Appending
Add new results without rewriting:
```bash
# Day 1
python inference.py --audio batch1/ --output results.jsonl

# Day 2 - just append!
python inference.py --audio batch2/ --output results.jsonl --append
```

### 4. Parallel Processing Friendly
Each line is independent:
```python
from multiprocessing import Pool

def process_result(line):
    result = json.loads(line)
    # Do something with result
    return analyze(result)

with open('results.jsonl', 'r') as f:
    lines = f.readlines()

with Pool(8) as p:
    analyses = p.map(process_result, lines)
```

---

## Migration from Old Format

If you have existing JSON files, you can convert them to JSONL:

```python
import json

# Read old format
with open('old_results.json', 'r') as f:
    old_data = json.load(f)

# Write to JSONL
with open('new_results.jsonl', 'w') as f:
    for result in old_data['results']:
        f.write(json.dumps(result) + '\n')
```

---

## Best Practices

1. **Use JSONL for large datasets** (>100 files)
2. **Include embeddings only if needed** (use `--no-embeddings` to save space)
3. **Adjust batch-size based on your hardware**
4. **Use multiple workers for I/O bound tasks**
5. **Stream processing for very large files** (don't load everything into memory)

---

## Troubleshooting

**Out of memory error:**
```bash
# Reduce batch size
--batch-size 4

# Or use legacy mode
--use-legacy
```

**Slow processing:**
```bash
# Increase batch size (if you have GPU memory)
--batch-size 16

# Increase workers (for I/O bottleneck)
--num-workers 8
```

**Want old format:**
```bash
# Use legacy inference
--use-legacy --output results.json

# Or convert JSONL to old JSON format
python convert_to_old_format.py results.jsonl results.json
```

---

## Summary

JSONL format is the **recommended default** for all new workflows because it:
- ✅ Provides detailed per-segment information
- ✅ Scales to massive datasets
- ✅ Supports streaming processing
- ✅ Uses PyTorch DataLoader for efficiency
- ✅ Enables temporal analysis of genre changes

For quick analysis or small datasets, JSON or CSV formats are still available using `--format json` or `--format csv`.


