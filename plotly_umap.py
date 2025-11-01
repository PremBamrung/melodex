#!/usr/bin/env python3
"""
Interactive UMAP Visualization Script for Music Embeddings
Reads a JSONL results file and creates an interactive 2D UMAP plot colored by genre.
"""

import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import umap
import pandas as pd
from pathlib import Path
import argparse


def load_results(jsonl_file):
    """
    Load embeddings and metadata from JSONL results file.
    Each music file should have a global prediction with the most probable genre.
    
    Args:
        jsonl_file: Path to the JSONL file
        
    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        genres: list of genre labels (from global prediction)
        titles: list of song titles
        audio_files: list of audio file paths (stems)
    """
    embeddings = []
    genres = []
    titles = []
    audio_files = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Extract embedding (already averaged per file)
            embedding = data.get('embedding', [])
            if embedding:
                # Extract audio file name
                audio_file = data.get('audio_file', '')
                file_stem = Path(audio_file).stem
                
                # Get the most probable genre from global_prediction
                global_pred = data.get('global_prediction', {})
                top_labels = global_pred.get('top_labels', [])
                genre = top_labels[0] if top_labels else 'Unknown'
                
                # Try to get title from data, otherwise use file stem
                title = data.get('title', data.get('song', file_stem))
                
                embeddings.append(embedding)
                genres.append(genre)
                titles.append(title)
                audio_files.append(file_stem)
    
    return np.array(embeddings), genres, titles, audio_files


def plot_umap(embeddings, genres, titles, audio_files, output_file='umap_plot.html', 
              n_neighbors=15, min_dist=0.1, metric='cosine'):
    """
    Create and save interactive UMAP visualization using Plotly.
    
    Args:
        embeddings: numpy array of embeddings
        genres: list of genre labels
        titles: list of song titles
        audio_files: list of audio file names (stems)
        output_file: path to save the plot (HTML file)
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        metric: distance metric for UMAP
    """
    print(f"Processing {len(embeddings)} embeddings...")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Get unique genres
    unique_genres = sorted(list(set(genres)))
    print(f"\nGenres found: {len(unique_genres)}")
    for genre in unique_genres:
        print(f"  - {genre}")
    
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
    
    # Create DataFrame for plotly
    df = pd.DataFrame({
        'UMAP-1': embedding_2d[:, 0],
        'UMAP-2': embedding_2d[:, 1],
        'Genre': genres,
        'Title': titles,
        'Filename': audio_files
    })
    
    # Create interactive scatter plot
    fig = px.scatter(
        df,
        x='UMAP-1',
        y='UMAP-2',
        color='Genre',
        hover_data={
            'Title': True,
            'Genre': True,
            'Filename': True,
            'UMAP-1': ':.3f',
            'UMAP-2': ':.3f'
        },
        title='Interactive UMAP Projection of Music Embeddings<br><sub>Colored by Genre - Hover for Details</sub>',
        color_discrete_sequence=px.colors.qualitative.Dark24 if len(unique_genres) > 10 else px.colors.qualitative.Plotly,
        width=1200,
        height=900
    )
    
    # Update markers
    fig.update_traces(
        marker=dict(
            size=10,
            line=dict(width=0.5, color='white'),
            opacity=0.8
        ),
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                      'Genre: %{customdata[1]}<br>' +
                      'File: %{customdata[2]}<br>' +
                      'UMAP-1: %{x:.3f}<br>' +
                      'UMAP-2: %{y:.3f}<br>' +
                      '<extra></extra>'
    )
    
    # Update layout
    fig.update_layout(
        font=dict(size=12),
        legend=dict(
            title=dict(text='Genre', font=dict(size=14)),
            font=dict(size=10),
            itemsizing='constant',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        ),
        xaxis=dict(
            title='UMAP Dimension 1',
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            title='UMAP Dimension 2',
            gridcolor='lightgray',
            showgrid=True
        ),
        plot_bgcolor='white',
        hovermode='closest'
    )
    
    # Save the plot
    fig.write_html(output_file)
    print(f"\nInteractive plot saved to: {output_file}")
    print(f"Open this file in your browser to explore the visualization!")
    
    # Also show the plot in browser
    # fig.show()
    
    return embedding_2d, unique_genres


def print_genre_statistics(genres):
    """Print statistics about genre distribution."""
    from collections import Counter
    
    genre_counts = Counter(genres)
    print("\n" + "="*70)
    print("GENRE DISTRIBUTION")
    print("="*70)
    for genre, count in genre_counts.most_common():
        print(f"{genre:50s}: {count:3d} ({count/len(genres)*100:5.1f}%)")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Create interactive UMAP visualization of music embeddings colored by genre'
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        default='results.jsonl',
        help='Path to JSONL results file (default: results.jsonl)'
    )
    parser.add_argument(
        '-o', '--output',
        default='umap_plot.html',
        help='Output file path for the plot (default: umap_plot.html)'
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
    print(f"Loading embeddings from {args.input_file}...")
    embeddings, genres, titles, audio_files = load_results(args.input_file)
    
    if len(embeddings) == 0:
        print("Error: No embeddings found in the file!")
        return
    
    # Print statistics
    print_genre_statistics(genres)
    
    # Create interactive UMAP plot
    plot_umap(
        embeddings, 
        genres,
        titles,
        audio_files,
        output_file=args.output,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric
    )


if __name__ == '__main__':
    main()

