#!/usr/bin/env python3
"""
UMAP Visualization Script for Music Embeddings
Reads a JSONL results file and creates a 2D UMAP plot colored by main genre.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import umap
from pathlib import Path
import argparse


def load_results(jsonl_file):
    """
    Load embeddings and genres from JSONL results file.
    
    Args:
        jsonl_file: Path to the JSONL file
        
    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        genres: list of main genre labels
        audio_files: list of audio file paths
    """
    embeddings = []
    genres = []
    audio_files = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Extract embedding
            embedding = data.get('embedding', [])
            if embedding:
                embeddings.append(embedding)
                
                # Extract main genre (first top label)
                main_genre = data.get('global_prediction', {}).get('top_labels', ['Unknown'])[0]
                genres.append(main_genre)
                
                # Extract audio file name
                audio_file = Path(data.get('audio_file', '')).stem
                audio_files.append(audio_file)
    
    return np.array(embeddings), genres, audio_files


def plot_umap(embeddings, genres, audio_files, output_file='umap_plot.png', 
              n_neighbors=15, min_dist=0.1, metric='cosine'):
    """
    Create and save UMAP visualization.
    
    Args:
        embeddings: numpy array of embeddings
        genres: list of genre labels
        audio_files: list of audio file names
        output_file: path to save the plot
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        metric: distance metric for UMAP
    """
    print(f"Processing {len(embeddings)} embeddings...")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Get unique genres and assign colors
    unique_genres = sorted(list(set(genres)))
    print(f"\nGenres found: {unique_genres}")
    
    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_genres)))
    genre_to_color = {genre: colors[i] for i, genre in enumerate(unique_genres)}
    
    # Fit UMAP
    print(f"\nFitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        n_components=2,
        n_jobs=-1
    )
    
    embedding_2d = reducer.fit_transform(embeddings)
    print("UMAP fitting complete!")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot each genre separately for legend
    for genre in unique_genres:
        mask = np.array(genres) == genre
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[genre_to_color[genre]],
            label=genre,
            alpha=0.7,
            s=100,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title('UMAP Projection of Music Embeddings\nColored by Main Genre', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Genre', loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Also show the plot
    plt.show()
    
    return embedding_2d, unique_genres


def print_genre_statistics(genres):
    """Print statistics about genre distribution."""
    from collections import Counter
    
    genre_counts = Counter(genres)
    print("\n" + "="*50)
    print("GENRE DISTRIBUTION")
    print("="*50)
    for genre, count in genre_counts.most_common():
        print(f"{genre:20s}: {count:3d} ({count/len(genres)*100:5.1f}%)")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description='Create UMAP visualization of music embeddings colored by genre'
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        default='results.jsonl',
        help='Path to JSONL results file (default: results.jsonl)'
    )
    parser.add_argument(
        '-o', '--output',
        default='umap_plot.png',
        help='Output file path for the plot (default: umap_plot.png)'
    )
    parser.add_argument(
        '-n', '--n-neighbors',
        type=int,
        default=15,
        help='UMAP n_neighbors parameter (default: 15)'
    )
    parser.add_argument(
        '-d', '--min-dist',
        type=float,
        default=0.1,
        help='UMAP min_dist parameter (default: 0.1)'
    )
    parser.add_argument(
        '-m', '--metric',
        default='cosine',
        help='Distance metric for UMAP (default: cosine)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: File '{args.input_file}' not found!")
        return
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    embeddings, genres, audio_files = load_results(args.input_file)
    
    if len(embeddings) == 0:
        print("Error: No embeddings found in the file!")
        return
    
    # Print statistics
    print_genre_statistics(genres)
    
    # Create UMAP plot
    plot_umap(
        embeddings, 
        genres, 
        audio_files,
        output_file=args.output,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric
    )


if __name__ == '__main__':
    main()

