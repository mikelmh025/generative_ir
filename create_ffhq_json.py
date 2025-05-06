#!/usr/bin/env python3
"""
Script to create corpus and queries JSON files for FFHQ dataset.
"""

import os
import json
import argparse
import random
from glob import glob
from tqdm import tqdm


def find_all_ffhq_images(ffhq_dir):
    """
    Find all PNG images in the FFHQ directory structure.
    
    Args:
        ffhq_dir: Root directory containing FFHQ images
        
    Returns:
        List of relative paths to all images
    """
    print(f"Finding all images in {ffhq_dir}...")
    
    all_images = []
    # Use glob to find all PNG files recursively
    for image_path in tqdm(glob(os.path.join(ffhq_dir, "**/*.png"), recursive=True)):
        # Convert to relative path from ffhq_dir
        rel_path = os.path.relpath(image_path, ffhq_dir)
        all_images.append(rel_path)
    
    print(f"Found {len(all_images)} total images")
    return all_images


def create_ffhq_json_files(ffhq_dir, output_dir, num_corpus_images=None, num_query_images=1000, seed=42):
    """
    Create JSON files for FFHQ corpus and queries, where queries are a subset of corpus.
    
    Args:
        ffhq_dir: Root directory containing FFHQ images
        output_dir: Directory to save the JSON files
        num_corpus_images: Number of images to include in the corpus (None means use all available)
        num_query_images: Number of images to include as queries (must be <= num_corpus_images)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (corpus_file_path, queries_file_path)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all image paths
    all_image_paths = find_all_ffhq_images(ffhq_dir)
    total_images = len(all_image_paths)
    
    # Handle the case when num_corpus_images is None (use all images)
    if num_corpus_images is None:
        num_corpus_images = total_images
        print(f"Using all {num_corpus_images} images for corpus")
    else:
        # Check if we have enough images
        if total_images < num_corpus_images:
            print(f"Warning: Found only {total_images} images, but requested {num_corpus_images}.")
            num_corpus_images = total_images
    
    # Ensure num_query_images is not greater than num_corpus_images
    if num_query_images > num_corpus_images:
        print(f"Warning: num_query_images ({num_query_images}) is greater than num_corpus_images ({num_corpus_images}).")
        print(f"Setting num_query_images to {num_corpus_images}")
        num_query_images = num_corpus_images
    
    # Shuffle and select corpus images
    random.shuffle(all_image_paths)
    corpus_paths = all_image_paths[:num_corpus_images]
    
    # Randomly select a subset of corpus images as query images
    query_indices = random.sample(range(num_corpus_images), num_query_images)
    query_paths = [corpus_paths[i] for i in query_indices]
    
    # Define output filenames
    corpus_filename = 'ffhq_corpus_all.json' if num_corpus_images == total_images else f'ffhq_corpus_{num_corpus_images}.json'
    corpus_file = os.path.join(output_dir, corpus_filename)
    
    # Save corpus JSON - just a list of image paths
    with open(corpus_file, 'w') as f:
        json.dump(corpus_paths, f)
    print(f"Created corpus file with {len(corpus_paths)} images: {corpus_file}")
    
    # Create query JSON - for each query image, with empty dialog
    queries = []
    for path in query_paths:
        query = {
            'img': path,
            'dialog': []  # Empty dialog list as required
        }
        queries.append(query)
    
    queries_file = os.path.join(output_dir, f'ffhq_queries_{num_query_images}.json')
    with open(queries_file, 'w') as f:
        json.dump(queries, f)
    print(f"Created queries file with {len(queries)} images: {queries_file}")
    
    return corpus_file, queries_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create JSON files for FFHQ dataset")
    
    parser.add_argument("--ffhq-dir", type=str, default="/data/skin_gen_data/ffhq/images", 
                        help="Root directory containing FFHQ images")
    parser.add_argument("--output-dir", type=str, default="./ffhq_data", 
                        help="Directory to save the JSON files")
    parser.add_argument("--corpus-size", type=lambda x: None if x == 'None' else int(x), default=50000, 
                        help="Number of images to include in the corpus")
    parser.add_argument("--query-size", type=int, default=2000, 
                        help="Number of images to include as queries")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--external-text-dir", type=str, 
                        default="./results_genir/self_dialogue_caption_ffhq-4b/captions", 
                        help="Directory containing external text captions (for validation)")
    parser.add_argument("--check-captions", action="store_true", 
                        help="Check which query images have corresponding captions")
    
    args = parser.parse_args()
    
    # Create the JSON files
    corpus_file, queries_file = create_ffhq_json_files(
        args.ffhq_dir,
        args.output_dir,
        args.corpus_size,
        args.query_size,
        args.seed
    )
    
