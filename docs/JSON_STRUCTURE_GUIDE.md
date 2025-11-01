# JSON Output Structure with Embeddings - Detailed Guide

## Overview

The JSON output file from the inference script contains three main sections:
1. **Audio Files List** - All processed audio files
2. **Mode** - Processing mode used
3. **Predictions** - Genre predictions for each file
4. **Embeddings** - 512-dimensional feature vectors (when enabled)

---

## Full Structure Breakdown

### 📁 Top-Level Structure

```json
{
  "audio_files": [ ... ],      // List of all audio file paths processed
  "mode": "full-file",          // or "first-30s"
  "predictions": [ ... ],       // Array of prediction objects
  "embeddings": [ ... ]         // Array of embedding vectors (if enabled)
}
```

---

## 1. Audio Files List

**Type:** Array of strings  
**Description:** Complete paths to all audio files that were processed

```json
"audio_files": [
  "/home/user/music/song1.mp3",
  "/home/user/music/song2.mp3",
  "/home/user/music/song3.mp3"
]
```

**Note:** The order matches the order of predictions and embeddings (index 0 = first file, index 1 = second file, etc.)

---

## 2. Mode

**Type:** String  
**Values:** `"full-file"` or `"first-30s"`

```json
"mode": "full-file"
```

- **"first-30s"**: Standard mode - only the first 30 seconds analyzed
- **"full-file"**: Sliding window mode - entire audio file analyzed in segments

---

## 3. Predictions

### A) Full-File Mode (with `--full` flag)

**Structure:** Array of objects, each containing top-3 predictions

```json
"predictions": [
  {
    "audio_file": "/path/to/song1.mp3",
    "top_predictions": [
      {
        "label": "Pop",
        "probability": 0.5440947413444519
      },
      {
        "label": "Rock",
        "probability": 0.15398797392845154
      },
      {
        "label": "Experimental",
        "probability": 0.12234155088663101
      }
    ]
  },
  {
    "audio_file": "/path/to/song2.mp3",
    "top_predictions": [
      {
        "label": "Folk",
        "probability": 0.34676727652549744
      },
      {
        "label": "Pop",
        "probability": 0.34505388140678406
      },
      {
        "label": "Instrumental",
        "probability": 0.09730793535709381
      }
    ]
  }
]
```

**Key Points:**
- Each audio file has 3 predictions ranked by probability
- Probabilities are averaged across all segments (sliding windows)
- Top prediction is the most likely genre

### B) Standard Mode (first 30s only)

**Structure:** Array of objects with single prediction + all class probabilities

```json
"predictions": [
  {
    "audio_file": "/path/to/song1.mp3",
    "predicted_label": "Pop",
    "confidence": 0.8532,
    "all_probabilities": {
      "Electronic": 0.0234,
      "Experimental": 0.0156,
      "Folk": 0.0423,
      "Hip-Hop": 0.0089,
      "Instrumental": 0.0167,
      "International": 0.0199,
      "Pop": 0.8532,
      "Rock": 0.0200
    }
  },
  {
    "audio_file": "/path/to/song2.mp3",
    "predicted_label": "Rock",
    "confidence": 0.7234,
    "all_probabilities": {
      "Electronic": 0.0412,
      "Experimental": 0.0523,
      "Folk": 0.0821,
      "Hip-Hop": 0.0156,
      "Instrumental": 0.0289,
      "International": 0.0331,
      "Pop": 0.0234,
      "Rock": 0.7234
    }
  }
]
```

**Key Points:**
- Single prediction per file
- `confidence` is the highest probability value
- `all_probabilities` contains probabilities for all 8 genres
- All probabilities sum to 1.0

---

## 4. Embeddings

**Type:** Array of arrays (2D array)  
**Dimensions:** `[number_of_files, 512]`  
**Description:** 512-dimensional feature vectors extracted from the CNN

```json
"embeddings": [
  [
    0.0,
    0.0,
    0.0,
    4.970742225646973,
    0.0,
    2.3451234567890123,
    ...  // 506 more values
    1.5375181436538696,
    0.0,
    0.0
  ],
  [
    0.0,
    0.7667840123176575,
    0.0,
    3.2341234567890123,
    ...  // 508 more values
    0.0,
    2.1234567890123456
  ]
  // ... one embedding per audio file
]
```

### Understanding Embeddings

**What are they?**
- 512-dimensional numerical vectors
- Extracted from the second-to-last layer of the CNN
- Represent the "learned features" of the audio

**What can you do with them?**

1. **Music Similarity Analysis**
   ```python
   import numpy as np
   from sklearn.metrics.pairwise import cosine_similarity
   
   # Load embeddings
   embeddings = np.array(results['embeddings'])
   
   # Calculate similarity between first two songs
   similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
   print(f"Similarity: {similarity:.4f}")
   ```

2. **Clustering Similar Songs**
   ```python
   from sklearn.cluster import KMeans
   
   # Cluster songs into 5 groups
   kmeans = KMeans(n_clusters=5)
   clusters = kmeans.fit_predict(embeddings)
   ```

3. **Build Recommendation System**
   ```python
   # Find 5 most similar songs to a given song
   from sklearn.neighbors import NearestNeighbors
   
   knn = NearestNeighbors(n_neighbors=5)
   knn.fit(embeddings)
   
   # Find similar to song at index 0
   distances, indices = knn.kneighbors([embeddings[0]])
   ```

4. **Dimensionality Reduction for Visualization**
   ```python
   from sklearn.manifold import TSNE
   import matplotlib.pyplot as plt
   
   # Reduce to 2D for plotting
   tsne = TSNE(n_components=2)
   embeddings_2d = tsne.fit_transform(embeddings)
   
   plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
   plt.show()
   ```

**Characteristics:**
- Sparse (many zeros due to ReLU activation)
- Each dimension captures different audio characteristics
- Similar songs have embeddings close in vector space
- Distance metrics (cosine, euclidean) measure similarity

---

## Complete Example

Here's what a complete JSON file looks like (3 files, full-file mode, with embeddings):

```json
{
  "audio_files": [
    "/music/song1.mp3",
    "/music/song2.mp3",
    "/music/song3.mp3"
  ],
  "mode": "full-file",
  "predictions": [
    {
      "audio_file": "/music/song1.mp3",
      "top_predictions": [
        {"label": "Pop", "probability": 0.65},
        {"label": "Rock", "probability": 0.20},
        {"label": "Electronic", "probability": 0.08}
      ]
    },
    {
      "audio_file": "/music/song2.mp3",
      "top_predictions": [
        {"label": "Folk", "probability": 0.45},
        {"label": "Pop", "probability": 0.32},
        {"label": "Instrumental", "probability": 0.12}
      ]
    },
    {
      "audio_file": "/music/song3.mp3",
      "top_predictions": [
        {"label": "Hip-Hop", "probability": 0.78},
        {"label": "Electronic", "probability": 0.11},
        {"label": "International", "probability": 0.05}
      ]
    }
  ],
  "embeddings": [
    [0.0, 0.0, 4.97, 0.0, ..., 1.53, 0.0],  // 512 values for song1
    [0.0, 0.76, 0.0, 3.23, ..., 0.0, 2.12],  // 512 values for song2
    [1.23, 0.0, 2.45, 0.0, ..., 3.45, 0.0]   // 512 values for song3
  ]
}
```

---

## Index Mapping

**Critical:** All arrays use the same index ordering!

```
Index 0:
  - audio_files[0] = "/music/song1.mp3"
  - predictions[0] = predictions for song1
  - embeddings[0] = embedding for song1

Index 1:
  - audio_files[1] = "/music/song2.mp3"
  - predictions[1] = predictions for song2
  - embeddings[1] = embedding for song2

Index N:
  - audio_files[N] = path to N-th song
  - predictions[N] = predictions for N-th song
  - embeddings[N] = embedding for N-th song
```

---

## File Size Considerations

**Embeddings Impact:**
- Each embedding = 512 floats ≈ 4 KB (in JSON)
- 100 songs with embeddings ≈ 400 KB
- 1,000 songs with embeddings ≈ 4 MB
- 10,000 songs with embeddings ≈ 40 MB

**Without embeddings:**
- Predictions only ≈ 0.5 KB per song
- Much smaller file size
- Use `--no-embeddings` if you don't need similarity analysis

---

## Loading in Python

```python
import json
import numpy as np

# Load the JSON file
with open('results.json', 'r') as f:
    results = json.load(f)

# Access components
audio_files = results['audio_files']
mode = results['mode']
predictions = results['predictions']
embeddings = np.array(results['embeddings'])  # Convert to numpy array

# Example: Get prediction for first file
print(f"File: {audio_files[0]}")
if mode == 'full-file':
    top_pred = predictions[0]['top_predictions'][0]
    print(f"Top Genre: {top_pred['label']} ({top_pred['probability']:.2%})")
else:
    pred = predictions[0]
    print(f"Genre: {pred['predicted_label']} ({pred['confidence']:.2%})")

# Example: Work with embeddings
print(f"Embedding shape: {embeddings.shape}")  # (num_files, 512)
print(f"First embedding: {embeddings[0][:10]}...")  # First 10 values
```

---

## Summary

| Component | Type | Size | Purpose |
|-----------|------|------|---------|
| `audio_files` | Array[String] | N paths | File identifiers |
| `mode` | String | 1 value | Processing mode |
| `predictions` | Array[Object] | N objects | Genre predictions |
| `embeddings` | Array[Array[Float]] | N × 512 | Feature vectors |

Where N = number of audio files processed.

