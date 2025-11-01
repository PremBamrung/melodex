import argparse
import zipfile
import os

def unzip(zip_file, extract_to):
    # Create the extraction directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)

    # Open the zip file for reading
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Extract all files into the specified directory
        zip_ref.extractall(extract_to)

    print("Extraction complete.")

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Unzip files.")

    # Add arguments
    parser.add_argument("zip_file", help="Path to the zip file")
    parser.add_argument("extract_to", help="Directory to extract files to")

    # Parse the arguments
    args = parser.parse_args()

    # Unzip the files
    unzip(args.zip_file, args.extract_to)

