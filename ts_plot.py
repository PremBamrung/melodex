#!/usr/bin/env python3
"""
Interactive UMAP Visualization Script for Taylor Swift Song Embeddings
Reads a JSONL results file and creates an interactive 2D UMAP plot colored by album.
Uses matched_songs.csv to map songs to albums.
"""

import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import umap
import pandas as pd
from pathlib import Path
import argparse


def load_album_mapping(csv_file):
    """
    Load the CSV file and create a mapping from filename to album and song title.
    
    Args:
        csv_file: Path to the matched_songs.csv file
        
    Returns:
        tuple: (album_map, title_map) - dictionaries mapping filename to album/title
    """
    df = pd.read_csv(csv_file)
    # Create mappings from filename to album and title
    album_map = {}
    title_map = {}
    
    for _, row in df.iterrows():
        filename = row['Filename']
        album = row['Album']
        title = row.get('Song', row.get('Title', ''))  # Try different column names
        
        if pd.notna(filename):
            if pd.notna(album):
                album_map[filename] = album
            if pd.notna(title):
                title_map[filename] = title
    
    return album_map, title_map


def load_results(jsonl_file, album_map, title_map):
    """
    Load embeddings and metadata from JSONL results file.
    
    Args:
        jsonl_file: Path to the JSONL file
        album_map: Dictionary mapping filenames to albums
        title_map: Dictionary mapping filenames to song titles
        
    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        albums: list of album labels
        titles: list of song titles
        audio_files: list of audio file paths (stems)
    """
    embeddings = []
    albums = []
    titles = []
    audio_files = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Extract embedding
            embedding = data.get('embedding', [])
            if embedding:
                # Extract audio file name
                audio_file = data.get('audio_file', '')
                filename = Path(audio_file).name
                file_stem = Path(audio_file).stem
                
                # Get album and title from mappings
                album = album_map.get(filename, 'Unknown')
                title = title_map.get(filename, file_stem)
                
                embeddings.append(embedding)
                albums.append(album)
                titles.append(title)
                audio_files.append(file_stem)
    
    return np.array(embeddings), albums, titles, audio_files


def plot_umap(embeddings, albums, titles, audio_files, output_file='ts_umap_plot.html', 
              n_neighbors=15, min_dist=0.1, metric='cosine'):
    """
    Create and save interactive UMAP visualization using Plotly.
    
    Args:
        embeddings: numpy array of embeddings
        albums: list of album labels
        titles: list of song titles
        audio_files: list of audio file names (stems)
        output_file: path to save the plot (HTML file)
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        metric: distance metric for UMAP
    """
    print(f"Processing {len(embeddings)} embeddings...")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Get unique albums
    unique_albums = sorted(list(set(albums)))
    print(f"\nAlbums found: {len(unique_albums)}")
    for album in unique_albums:
        print(f"  - {album}")
    
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
        'Album': albums,
        'Title': titles,
        'Filename': audio_files
    })
    
    # Create interactive scatter plot
    fig = px.scatter(
        df,
        x='UMAP-1',
        y='UMAP-2',
        color='Album',
        hover_data={
            'Title': True,
            'Album': True,
            'Filename': True,
            'UMAP-1': ':.3f',
            'UMAP-2': ':.3f'
        },
        title='Interactive UMAP Projection of Taylor Swift Song Embeddings<br><sub>Colored by Album - Hover for Details</sub>',
        color_discrete_sequence=px.colors.qualitative.Dark24 if len(unique_albums) > 10 else px.colors.qualitative.Plotly,
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
                      'Album: %{customdata[1]}<br>' +
                      'File: %{customdata[2]}<br>' +
                      'UMAP-1: %{x:.3f}<br>' +
                      'UMAP-2: %{y:.3f}<br>' +
                      '<extra></extra>'
    )
    
    # Update layout
    fig.update_layout(
        font=dict(size=12),
        legend=dict(
            title=dict(text='Album', font=dict(size=14)),
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
    
    return embedding_2d, unique_albums


def print_album_statistics(albums):
    """Print statistics about album distribution."""
    from collections import Counter
    
    album_counts = Counter(albums)
    print("\n" + "="*70)
    print("ALBUM DISTRIBUTION")
    print("="*70)
    for album, count in album_counts.most_common():
        print(f"{album:50s}: {count:3d} ({count/len(albums)*100:5.1f}%)")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Create interactive UMAP visualization of Taylor Swift song embeddings colored by album'
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        default='results.jsonl',
        help='Path to JSONL results file (default: results.jsonl)'
    )
    parser.add_argument(
        '-c', '--csv',
        default='matched_songs.csv',
        help='Path to matched songs CSV file (default: matched_songs.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        default='ts_umap_plot.html',
        help='Output file path for the plot (default: ts_umap_plot.html)'
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
    
    # Check if input files exist
    if not Path(args.input_file).exists():
        print(f"Error: File '{args.input_file}' not found!")
        return
    
    if not Path(args.csv).exists():
        print(f"Error: CSV file '{args.csv}' not found!")
        return
    
    # Load album and title mappings from CSV
    print(f"Loading metadata from {args.csv}...")
    album_map, title_map = load_album_mapping(args.csv)
    print(f"Loaded {len(album_map)} album mappings and {len(title_map)} title mappings")
    
    # Load data
    print(f"\nLoading embeddings from {args.input_file}...")
    embeddings, albums, titles, audio_files = load_results(args.input_file, album_map, title_map)
    
    if len(embeddings) == 0:
        print("Error: No embeddings found in the file!")
        return
    
    # Print statistics
    print_album_statistics(albums)
    
    # Create interactive UMAP plot
    plot_umap(
        embeddings, 
        albums,
        titles,
        audio_files,
        output_file=args.output,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric
    )


if __name__ == '__main__':
    main()

