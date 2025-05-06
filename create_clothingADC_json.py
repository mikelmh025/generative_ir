"""
Script to create corpus and queries JSON files for ClothingADC dataset.
Modified to support creating multiple corpus files of different sizes, 
with one set of queries that is a subset of all corpus files.
"""

import os
import json
import argparse
import random
from glob import glob
from tqdm import tqdm


def find_all_clothingadc_images(clothingadc_dir):
    """
    Find all PNG images in the ClothingADC directory structure.
    
    Args:
        clothingadc_dir: Root directory containing ClothingADC images
        
    Returns:
        List of relative paths to all images
    """
    print(f"Finding all images in {clothingadc_dir}...")
    
    all_images = []
    # Use glob to find all PNG files recursively
    for image_path in tqdm(glob(os.path.join(clothingadc_dir, "**/*.png"), recursive=True)):
        # Convert to relative path from clothingadc_dir
        rel_path = os.path.relpath(image_path, clothingadc_dir)
        all_images.append(rel_path)
    
    print(f"Found {len(all_images)} total images")
    return all_images


def create_clothingadc_json_files(clothingadc_dir, output_dir, corpus_sizes, num_query_images=1000, seed=42):
    """
    Create JSON files for ClothingADC corpus and queries, where queries are a subset of all corpus sizes.
    
    Args:
        clothingadc_dir: Root directory containing ClothingADC images
        output_dir: Directory to save the JSON files
        corpus_sizes: List of sizes for different corpus files (None means use all available)
        num_query_images: Number of images to include as queries
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (corpus_file_paths, queries_file_path)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all image paths
    all_image_paths = find_all_clothingadc_images(clothingadc_dir)
    total_images = len(all_image_paths)
    
    # Process corpus sizes
    processed_corpus_sizes = []
    for size in corpus_sizes:
        if size is None or size == "all":
            processed_corpus_sizes.append(total_images)
        else:
            size = int(size)
            if size > total_images:
                print(f"Warning: Requested corpus size {size} exceeds total images {total_images}. Using all available images.")
                processed_corpus_sizes.append(total_images)
            else:
                processed_corpus_sizes.append(size)
    
    # Sort corpus sizes in ascending order
    processed_corpus_sizes.sort()
    
    # Ensure we have at least one corpus size
    if not processed_corpus_sizes:
        processed_corpus_sizes = [total_images]
        print(f"No valid corpus sizes provided. Using all {total_images} images.")
    
    # Ensure the largest corpus size has enough images for queries
    largest_corpus_size = processed_corpus_sizes[-1]
    if num_query_images > largest_corpus_size:
        print(f"Warning: num_query_images ({num_query_images}) is greater than largest corpus size ({largest_corpus_size}).")
        print(f"Setting num_query_images to {largest_corpus_size}")
        num_query_images = largest_corpus_size
    
    # Shuffle all images
    random.shuffle(all_image_paths)
    
    # Create corpus files for each size
    corpus_files = []
    for size in processed_corpus_sizes:
        corpus_paths = all_image_paths[:size]
        
        # Define output filename
        corpus_filename = 'clothingadc_corpus_all.json' if size == total_images else f'clothingadc_corpus_{size}.json'
        corpus_file = os.path.join(output_dir, corpus_filename)
        
        # Save corpus JSON - just a list of image paths
        with open(corpus_file, 'w') as f:
            json.dump(corpus_paths, f)
        print(f"Created corpus file with {len(corpus_paths)} images: {corpus_file}")
        corpus_files.append(corpus_file)
    
    # Select query images from the smallest corpus to ensure they're in all corpora
    smallest_corpus_size = processed_corpus_sizes[0]
    if num_query_images > smallest_corpus_size:
        print(f"Warning: num_query_images ({num_query_images}) is greater than smallest corpus size ({smallest_corpus_size}).")
        print(f"Setting num_query_images to {smallest_corpus_size}")
        num_query_images = smallest_corpus_size
    
    query_indices = random.sample(range(smallest_corpus_size), num_query_images)
    query_paths = [all_image_paths[i] for i in query_indices]
    
    # Create query JSON - for each query image, with empty dialog
    queries = []
    for path in query_paths:
        query = {
            'img': path,
            'dialog': []  # Empty dialog list as required
        }
        queries.append(query)
    
    queries_file = os.path.join(output_dir, f'clothingadc_queries_{num_query_images}.json')
    with open(queries_file, 'w') as f:
        json.dump(queries, f)
    print(f"Created queries file with {len(queries)} images: {queries_file}")
    
    return corpus_files, queries_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create JSON files for ClothingADC dataset")
    
    parser.add_argument("--clothingadc-dir", type=str, default="/data/cloth1m/v4/cloth1m_data_v4/images/", 
                        help="Root directory containing ClothingADC images")
    parser.add_argument("--output-dir", type=str, default="./clothingadc_data", 
                        help="Directory to save the JSON files")
    parser.add_argument("--corpus-sizes", type=str, default="5000,10000,20000,40000,60000,100000,500000,all", 
                        help="Comma-separated list of corpus sizes (use 'all' for all images)")
    parser.add_argument("--query-size", type=int, default=2000, 
                        help="Number of images to include as queries")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--external-text-dir", type=str, 
                        default="./results_genir/self_dialogue_caption_clothingadc/captions", 
                        help="Directory containing external text captions (for validation)")
    parser.add_argument("--check-captions", action="store_true", 
                        help="Check which query images have corresponding captions")
    parser.add_argument("--include-metadata", action="store_true",
                        help="Include metadata from metadata.json if available")
    
    args = parser.parse_args()
    
    # Parse corpus sizes
    corpus_sizes = []
    for size in args.corpus_sizes.split(','):
        size = size.strip()
        if size.lower() == 'all' or size.lower() == 'none':
            corpus_sizes.append(None)
        else:
            try:
                corpus_sizes.append(int(size))
            except ValueError:
                print(f"Warning: Invalid corpus size '{size}'. Skipping.")
    
    # Create the JSON files
    corpus_files, queries_file = create_clothingadc_json_files(
        args.clothingadc_dir,
        args.output_dir,
        corpus_sizes,
        args.query_size,
        args.seed
    )