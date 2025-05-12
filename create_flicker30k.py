#!/usr/bin/env python3
"""
Script to create corpus and queries JSON files for Flickr30k dataset.
"""

import os
import json
import argparse
import random
from glob import glob
from tqdm import tqdm


def find_all_flickr30k_images(flickr30k_dir):
    """
    Find all JPG images in the Flickr30k directory structure.
    
    Args:
        flickr30k_dir: Root directory containing Flickr30k images
        
    Returns:
        List of relative paths to all images
    """
    print(f"Finding all images in {flickr30k_dir}...")
    
    all_images = []
    # Use glob to find all JPG files (Flickr30k uses JPG format)
    for image_path in tqdm(glob(os.path.join(flickr30k_dir, "*.jpg"))):
        # Convert to relative path from flickr30k_dir
        rel_path = os.path.relpath(image_path, flickr30k_dir)
        all_images.append(rel_path)
    
    print(f"Found {len(all_images)} total images")
    return all_images


def create_flickr30k_json_files(flickr30k_dir, output_dir, num_corpus_images=None, num_query_images=1000, seed=42):
    """
    Create JSON files for Flickr30k corpus and queries, where queries are a subset of corpus.
    
    Args:
        flickr30k_dir: Root directory containing Flickr30k images
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
    all_image_paths = find_all_flickr30k_images(flickr30k_dir)
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
    corpus_filename = 'flickr30k_corpus_all.json' if num_corpus_images == total_images else f'flickr30k_corpus_{num_corpus_images}.json'
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
    
    queries_file = os.path.join(output_dir, f'flickr30k_queries_{num_query_images}.json')
    with open(queries_file, 'w') as f:
        json.dump(queries, f)
    print(f"Created queries file with {len(queries)} images: {queries_file}")
    
    return corpus_file, queries_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create JSON files for Flickr30k dataset")
    
    parser.add_argument("--flickr30k-dir", type=str, default="/data/flickr30k/flickr30k-images", 
                        help="Root directory containing Flickr30k images")
    parser.add_argument("--output-dir", type=str, default="./flickr30k_data", 
                        help="Directory to save the JSON files")
    parser.add_argument("--corpus-size", type=lambda x: None if x == 'None' else int(x), default=None, 
                        help="Number of images to include in the corpus (default: all available)")
    parser.add_argument("--query-size", type=int, default=2000, 
                        help="Number of images to include as queries")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--external-text-dir", type=str, 
                        default="./results_genir/self_dialogue_caption_flickr30k/captions", 
                        help="Directory containing external text captions (for validation)")
    parser.add_argument("--check-captions", action="store_true", 
                        help="Check which query images have corresponding captions")
    
    args = parser.parse_args()
    
    # Create the JSON files
    corpus_file, queries_file = create_flickr30k_json_files(
        args.flickr30k_dir,
        args.output_dir,
        args.corpus_size,
        args.query_size,
        args.seed
    )
    
    # Check for captions if requested
    if args.check_captions and os.path.exists(args.external_text_dir):
        print(f"Checking for captions in {args.external_text_dir}...")
        
        # Load the queries
        with open(queries_file, 'r') as f:
            queries = json.load(f)
        
        # Count how many query images have captions
        have_captions = 0
        for query in queries:
            img_path = query['img']
            # Get the image filename without extension
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Check if a caption file exists (assuming same naming convention)
            caption_file = os.path.join(args.external_text_dir, f"{img_name}.txt")
            if os.path.exists(caption_file):
                have_captions += 1
        
        print(f"Found captions for {have_captions} out of {len(queries)} query images")