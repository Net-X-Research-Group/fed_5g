#!/usr/bin/env python3
"""
Script to rename folders by Run ID based on metadata from a CSV file.

Usage:
    python rename_folders.py <csv_file> <directory>
    
Example:
    python rename_folders.py run_metadata.csv /path/to/folders
"""

import os
import sys
import re
import csv
from pathlib import Path


def clean_value(value):
    """Remove leading/trailing spaces from a value."""
    return value.strip() if value else ""


def parse_tdd_split(tdd_split):
    """
    Parse TDD Split column to format like '3-1'.
    Expected formats: '3/1', '3-1', '3:1', etc.
    """
    tdd_split = clean_value(tdd_split)
    # Replace common separators with dash
    tdd_split = re.sub(r'[/:]', '-', tdd_split)
    return tdd_split


def parse_mimo(num_in_out):
    """
    Parse NumIn NumOut column to determine MIMO2x2 or SISO.
    Examples: '2 2', '2x2', '1 1', '1x1'
    """
    num_in_out = clean_value(num_in_out)
    # Extract numbers from the string
    numbers = re.findall(r'\d+', num_in_out)
    
    if len(numbers) >= 2:
        num_in = int(numbers[0])
        num_out = int(numbers[1])
        if num_in == 2 and num_out == 2:
            return "MIMO2x2"
        elif num_in == 1 and num_out == 1:
            return "SISO"
        else:
            return f"MIMO{num_in}x{num_out}"
    
    return "UNKNOWN"


def is_run_id(folder_name):
    """Check if a folder name looks like a Run ID (string of ~20 digits)."""
    # Check if it's purely digits and approximately 15-25 characters long
    return bool(re.match(r'^\d{15,25}$', folder_name))


def is_already_renamed(folder_name):
    """
    Check if a folder has already been renamed.
    A renamed folder should match the pattern: RunID_XN_YMHz_..._..._...
    """
    # Pattern: digits_XN_YMHz_..._..._...
    pattern = r'^\d{15,25}_\d+N_\d+MHz_.+_.+_.+$'
    return bool(re.match(pattern, folder_name))


def build_new_name(run_id, metadata):
    """
    Build the new folder name from Run ID and metadata.
    Format: [Run ID]_[Number of nodes]N_[Bandwidth]MHz_[TDD]_[MIMO]_[Distribution]
    """
    num_nodes = clean_value(metadata['Num nodes'])
    bandwidth = clean_value(metadata['Bandwidth (MHz)'])
    tdd_split = parse_tdd_split(metadata['TDD Split (D/U)'])
    mimo = parse_mimo(metadata['NumIn NumOut'])
    distribution = clean_value(metadata['Distribution'])
    
    new_name = f"{run_id}_{num_nodes}N_{bandwidth}MHz_{tdd_split}_{mimo}_{distribution}"
    return new_name


def load_metadata(csv_file):
    """
    Load metadata from CSV file into a dictionary keyed by Run ID.
    Returns: dict mapping Run ID -> metadata dict
    """
    metadata_dict = {}
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Verify required columns exist
            required_columns = [
                'Num nodes', 
                'Bandwidth (MHz)', 
                'TDD Split (D/U)', 
                'NumIn NumOut', 
                'Distribution'
            ]
            
            # Check for column name variations (with/without extra spaces)
            fieldnames = [field.strip() for field in reader.fieldnames]
            
            for col in required_columns:
                if col not in fieldnames:
                    print(f"Error: Required column '{col}' not found in CSV.")
                    print(f"Available columns: {fieldnames}")
                    sys.exit(1)
            
            # Process each row
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                # Clean up column names in case they have extra spaces
                clean_row = {k.strip(): v for k, v in row.items()}
                
                # Find the Run ID column (look for a column with ~20 digit values)
                run_id = None
                for key, value in clean_row.items():
                    value_clean = clean_value(value)
                    if re.match(r'^\d{15,25}$', value_clean):
                        run_id = value_clean
                        break
                
                if not run_id:
                    print(f"Warning: No Run ID found in row {row_num}, skipping")
                    continue
                
                # Store metadata for this Run ID
                metadata_dict[run_id] = {
                    'Num nodes': clean_row.get('Num nodes', ''),
                    'Bandwidth (MHz)': clean_row.get('Bandwidth (MHz)', ''),
                    'TDD Split (D/U)': clean_row.get('TDD Split (D/U)', ''),
                    'NumIn NumOut': clean_row.get('NumIn NumOut', ''),
                    'Distribution': clean_row.get('Distribution', '')
                }
    
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    return metadata_dict


def rename_folders(csv_file, directory):
    """
    Main function to rename folders based on CSV metadata.
    """
    # Load metadata from CSV
    print(f"Loading metadata from {csv_file}...")
    metadata_dict = load_metadata(csv_file)
    print(f"Loaded metadata for {len(metadata_dict)} Run IDs")
    
    # Get all folders in the directory
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
    
    folders = [f for f in dir_path.iterdir() if f.is_dir()]
    print(f"Found {len(folders)} folders in {directory}")
    
    # Track statistics
    renamed_count = 0
    skipped_already_renamed = 0
    skipped_no_metadata = 0
    skipped_not_run_id = 0
    
    # Process each folder
    for folder in folders:
        folder_name = folder.name
        
        # Check if it's a Run ID folder
        if not is_run_id(folder_name):
            skipped_not_run_id += 1
            continue
        
        # Check if already renamed
        if is_already_renamed(folder_name):
            print(f"Skipping (already renamed): {folder_name}")
            skipped_already_renamed += 1
            continue
        
        # Check if we have metadata for this Run ID
        if folder_name not in metadata_dict:
            print(f"Skipping (no metadata): {folder_name}")
            skipped_no_metadata += 1
            continue
        
        # Build new name
        try:
            new_name = build_new_name(folder_name, metadata_dict[folder_name])
            new_path = folder.parent / new_name
            
            # Check if target already exists
            if new_path.exists():
                print(f"Warning: Target already exists, skipping: {new_name}")
                continue
            
            # Rename the folder
            print(f"Renaming: {folder_name}")
            print(f"      -> {new_name}")
            folder.rename(new_path)
            renamed_count += 1
            
        except Exception as e:
            print(f"Error renaming {folder_name}: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total folders processed: {len(folders)}")
    print(f"  Renamed: {renamed_count}")
    print(f"  Skipped (already renamed): {skipped_already_renamed}")
    print(f"  Skipped (no metadata): {skipped_no_metadata}")
    print(f"  Skipped (not Run ID): {skipped_not_run_id}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rename_folders.py <csv_file> <directory>")
        print("\nExample:")
        print("  python rename_folders.py run_metadata.csv /path/to/folders")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    directory = sys.argv[2]
    
    rename_folders(csv_file, directory)