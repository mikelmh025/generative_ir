#!/usr/bin/env python3
"""
Script to create corpus and queries JSON files for ClothingADC dataset.
Modified to output corpus JSON as a simple list of image paths.
"""

import os
import json
import argparse
import random
from glob import glob
from tqdm import tqdm


def find_all_clothing_images(clothing_dir):
    """
    Find all JPG images in the ClothingADC directory structure.
    
    Args:
        clothing_dir: Root directory containing ClothingADC images
        
    Returns:
        List of relative paths to all images
    """
    print(f"Finding all images in {clothing_dir}...")
    
    all_images = []
    
    # Get all clothing categories
    clothing_categories = [d for d in os.listdir(clothing_dir) if os.path.isdir(os.path.join(clothing_dir, d))]
    print(f"Found {len(clothing_categories)} clothing categories: {', '.join(clothing_categories)}")
    
    # Process each category
    for category in clothing_categories:
        category_dir = os.path.join(clothing_dir, category)
        print(f"Processing category: {category}")
        
        # Get all subcategories (e.g., Black_Wool_Floral)
        subcategories = [d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))]
        
        # Process each subcategory
        for subcategory in tqdm(subcategories, desc=f"Finding images in {category}"):
            subcategory_path = os.path.join(category_dir, subcategory)
            
            # Find all JPG files in this subcategory
            for image_path in glob(os.path.join(subcategory_path, "*.jpg")):
                # Convert to relative path from clothing_dir
                rel_path = os.path.relpath(image_path, clothing_dir)
                all_images.append(rel_path)
    
    print(f"Found {len(all_images)} total images")
    return all_images


def parse_clothing_metadata(image_path):
    """
    Parse metadata from the image path.
    
    Args:
        image_path: Path to image file (e.g., 'Dress/Black_Wool_Floral/000001.jpg')
        
    Returns:
        Dictionary with metadata (category, color, material, pattern)
    """
    parts = image_path.split('/')
    category = parts[0]
    if len(parts) >= 2 and '_' in parts[1]:
        style_parts = parts[1].split('_')
        if len(style_parts) >= 3:
            color = style_parts[0]
            material = style_parts[1]
            pattern = style_parts[2]
            return {
                "category": category,
                "color": color,
                "material": material,
                "pattern": pattern
            }
    
    # Default return if parsing fails
    return {"category": category}


def create_clothing_json_files(clothing_dir, output_dir, num_corpus_images=None, num_query_images=1000, seed=42):
    """
    Create JSON files for ClothingADC corpus and queries, where queries are a subset of corpus.
    Corpus is now a simple list of image paths.
    
    Args:
        clothing_dir: Root directory containing ClothingADC images
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
    all_image_paths = find_all_clothing_images(clothing_dir)
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
    
    # Define output filenames
    corpus_filename = 'clothing_corpus_all.json' if num_corpus_images == total_images else f'clothing_corpus_{num_corpus_images}.json'
    corpus_file = os.path.join(output_dir, corpus_filename)
    
    # Save corpus JSON as a simple list of paths
    with open(corpus_file, 'w') as f:
        json.dump(corpus_paths, f)
    print(f"Created corpus file with {len(corpus_paths)} image paths: {corpus_file}")
    
    # For queries, we still need the metadata and dialog structure
    query_indices = random.sample(range(num_corpus_images), num_query_images)
    query_paths = [corpus_paths[i] for i in query_indices]
    
    # Process queries and add metadata and empty dialog field
    queries = []
    for path in query_paths:
        metadata = parse_clothing_metadata(path)
        query = {
            "img": path,
            "metadata": metadata,
            "dialog": []  # Empty dialog list as required
        }
        queries.append(query)
    
    queries_file = os.path.join(output_dir, f'clothing_queries_{num_query_images}.json')
    with open(queries_file, 'w') as f:
        json.dump(queries, f)
    print(f"Created queries file with {len(queries)} images: {queries_file}")
    
    return corpus_file, queries_file


def create_clothing_category_statistics(clothing_dir, output_dir):
    """
    Create a JSON file with statistics about the ClothingADC dataset categories.
    
    Args:
        clothing_dir: Root directory containing ClothingADC images
        output_dir: Directory to save the JSON files
    
    Returns:
        Path to the statistics file
    """
    print("Generating category statistics...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image paths
    all_image_paths = find_all_clothing_images(clothing_dir)
    
    # Initialize counters
    stats = {
        "total_images": len(all_image_paths),
        "categories": {},
        "colors": {},
        "materials": {},
        "patterns": {}
    }
    
    # Count occurrences of each category, color, material, and pattern
    for path in all_image_paths:
        metadata = parse_clothing_metadata(path)
        
        # Update category count
        category = metadata.get("category")
        if category:
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
        
        # Update color count
        color = metadata.get("color")
        if color:
            stats["colors"][color] = stats["colors"].get(color, 0) + 1
        
        # Update material count
        material = metadata.get("material")
        if material:
            stats["materials"][material] = stats["materials"].get(material, 0) + 1
        
        # Update pattern count
        pattern = metadata.get("pattern")
        if pattern:
            stats["patterns"][pattern] = stats["patterns"].get(pattern, 0) + 1
    
    # Save statistics to JSON file
    stats_file = os.path.join(output_dir, 'clothing_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Created statistics file: {stats_file}")
    return stats_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create JSON files for ClothingADC dataset")
    
    parser.add_argument("--clothing-dir", type=str, default="/data/clothing/images", 
                        help="Root directory containing ClothingADC images")
    parser.add_argument("--output-dir", type=str, default="./clothing_data", 
                        help="Directory to save the JSON files")
    parser.add_argument("--corpus-size", type=lambda x: None if x == 'None' else int(x), default=None, 
                        help="Number of images to include in the corpus (None means use all available)")
    parser.add_argument("--query-size", type=int, default=1000, 
                        help="Number of images to include as queries")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--generate-stats", action="store_true", 
                        help="Generate statistics about the dataset categories")
    parser.add_argument("--filter-category", type=str, default=None,
                        help="Filter corpus to include only a specific category (e.g., 'Dress')")
    parser.add_argument("--filter-color", type=str, default=None,
                        help="Filter corpus to include only a specific color (e.g., 'Black')")
    parser.add_argument("--filter-material", type=str, default=None,
                        help="Filter corpus to include only a specific material (e.g., 'Wool')")
    parser.add_argument("--filter-pattern", type=str, default=None,
                        help="Filter corpus to include only a specific pattern (e.g., 'Floral')")
    
    args = parser.parse_args()
    
    # Create the JSON files
    corpus_file, queries_file = create_clothing_json_files(
        args.clothing_dir,
        args.output_dir,
        args.corpus_size,
        args.query_size,
        args.seed
    )
    
    # Generate statistics if requested
    if args.generate_stats:
        stats_file = create_clothing_category_statistics(args.clothing_dir, args.output_dir)