import os
import csv
from pathlib import Path
from difflib import SequenceMatcher
import re

def normalize_album_name(album_str):
    """Normalize album names to canonical versions"""
    if not album_str:
        return "Unknown Album"
    
    # Canonical album names
    album_map = {
        'taylor swift': 'Taylor Swift (2006)',
        'fearless': 'Fearless (2008)',
        'speak now': 'Speak Now (2010)',
        'red': 'Red (2012)',
        '1989': '1989 (2014)',
        'reputation': 'Reputation (2017)',
        'lover': 'Lover (2019)',
        'folklore': 'Folklore (2020)',
        'evermore': 'Evermore (2020)',
        'midnights': 'Midnights (2022)',
        'the tortured poets department': 'The Tortured Poets Department (2024)',
        'the life of a showgirl': 'The Life of a Showgirl (2025)',
    }
    
    # Clean the album string - remove version info, deluxe, etc.
    album_lower = album_str.lower()
    
    # Check each canonical album name
    for key, canonical in album_map.items():
        if key in album_lower:
            return canonical
    
    # If it's a non-album song, collaboration, or soundtrack
    if 'non-album' in album_lower:
        return 'Non-album song'
    
    # Return original if no match found
    return album_str

def clean_song_name(name):
    """Remove quotes, special characters, and normalize the song name"""
    # Remove triple quotes and regular quotes
    name = name.replace('"""', '').replace('"', '').replace("'", '')
    # Remove file extension if present
    name = re.sub(r'\.(mp3|MP3)$', '', name)
    # Remove common patterns like (feat. Artist), [Remix], etc.
    name = re.sub(r'\(.*?\)|\[.*?\]', '', name)
    # Remove extra whitespace
    name = ' '.join(name.split())
    # Convert to lowercase for comparison
    return name.lower().strip()

def similarity_score(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a, b).ratio()

def load_csv_metadata(csv_path):
    """Load song metadata from CSV file"""
    songs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Song']:  # Skip empty rows
                songs.append(row)
    return songs

def find_mp3_files(directory):
    """Find all MP3 files in the given directory"""
    mp3_files = []
    for file in Path(directory).rglob('*.mp3'):
        mp3_files.append(file)
    return mp3_files

def match_mp3_to_metadata(mp3_path, metadata, threshold=0.6):
    """Match an MP3 file to metadata from CSV"""
    mp3_name = mp3_path.stem  # Get filename without extension
    cleaned_mp3 = clean_song_name(mp3_name)
    
    best_match = None
    best_score = 0
    
    for song_data in metadata:
        song_name = clean_song_name(song_data['Song'])
        score = similarity_score(cleaned_mp3, song_name)
        
        if score > best_score:
            best_score = score
            best_match = song_data
    
    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, best_score

def main():
    # Configuration
    csv_path = 'List_of_songs_by_Taylor_Swift.csv'
    mp3_directory = 'samples/taylor_swift'  # Change this to your MP3 directory
    similarity_threshold = 0.6  # Adjust threshold as needed (0.0 to 1.0)
    
    print("Loading metadata from CSV...")
    metadata = load_csv_metadata(csv_path)
    print(f"Loaded {len(metadata)} songs from CSV\n")
    
    print("Scanning for MP3 files...")
    mp3_files = find_mp3_files(mp3_directory)
    print(f"Found {len(mp3_files)} MP3 files\n")
    
    print("Matching MP3 files to metadata...")
    print("=" * 80)
    
    matched = []
    unmatched = []
    
    for mp3_file in mp3_files:
        match, score = match_mp3_to_metadata(mp3_file, metadata, similarity_threshold)
        
        if match:
            # Normalize the album name
            normalized_album = normalize_album_name(match['Album(s)'])
            
            matched.append({
                'file': mp3_file,
                'match': match,
                'score': score,
                'normalized_album': normalized_album
            })
            print(f"✓ {mp3_file.name}")
            print(f"  → Matched: {match['Song']}")
            print(f"  → Album: {normalized_album}")
            print(f"  → Year: {match['Release year']}")
            print(f"  → Writer(s): {match['Writer(s)']}")
            print(f"  → Confidence: {score:.2%}\n")
        else:
            unmatched.append({
                'file': mp3_file,
                'best_score': score
            })
            print(f"✗ {mp3_file.name}")
            print(f"  → No match found (best score: {score:.2%})\n")
    
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Matched: {len(matched)}/{len(mp3_files)}")
    print(f"  Unmatched: {len(unmatched)}/{len(mp3_files)}")
    
    # Optionally, save results to a file
    if matched:
        print("\nSaving matched results to 'matched_songs.csv'...")
        with open('matched_songs.csv', 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Filename', 'Song', 'Artist(s)', 'Writer(s)', 'Album', 
                         'Release year', 'Confidence']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in matched:
                writer.writerow({
                    'Filename': item['file'].name,
                    'Song': item['match']['Song'],
                    'Artist(s)': item['match']['Artist(s)'],
                    'Writer(s)': item['match']['Writer(s)'],
                    'Album': item['normalized_album'],
                    'Release year': item['match']['Release year'],
                    'Confidence': f"{item['score']:.2%}"
                })
        print("Results saved successfully!")

if __name__ == "__main__":
    main()

