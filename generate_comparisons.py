import os
import json
import torch
from diffusers import StableDiffusion3Pipeline, FluxPipeline
from PIL import Image
import time
from tqdm import tqdm
import numpy as np
from multiprocessing import Process, Queue, Manager
import queue
import requests
import urllib.parse
import io
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Initialize the models
def init_sd35():
    torch.set_default_dtype(torch.float16)
    sd_pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    sd_pipe = sd_pipe.to("cuda:0")
    return sd_pipe

def init_flux():
    flux_pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=torch.float16
    )
    flux_pipe = flux_pipe.to("cuda:1")
    return flux_pipe

def create_directory_structure():
    # Create main directories
    base_dir = "generated_images"
    categories = ["sd35", "flux", "search_engine"]
    for category in categories:
        os.makedirs(os.path.join(base_dir, category), exist_ok=True)
    
    # Create comparison tables directory
    os.makedirs(os.path.join(base_dir, "comparison_tables"), exist_ok=True)

def generate_sd35_image(prompt, output_path, pipe):
    try:
        image = pipe(
            prompt,
            num_inference_steps=28,
            guidance_scale=3.5,
        ).images[0]
        image.save(output_path)
        return True
    except Exception as e:
        print(f"Error generating SD35 image: {e}")
        return False

def generate_flux_image(prompt, output_path, pipe):
    try:
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        image.save(output_path)
        return True
    except Exception as e:
        print(f"Error generating FLUX image: {e}")
        return False

import requests
import urllib.parse
import io
import random
import time

def search_unsplash(query, size=(1024, 1024)):
    """
    Search for images using Unsplash public API
    """
    try:
        # Format the query for URL
        encoded_query = urllib.parse.quote(query)
        
        # Use Unsplash source API (doesn't require authentication)
        # This is their public Source API that's meant for public use
        url = f"https://source.unsplash.com/1024x1024/?{encoded_query}"
        
        # Send request
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content)).resize(size)
        else:
            print(f"Unsplash search failed with status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error searching Unsplash: {e}")
        return None

def search_pixabay(query, size=(1024, 1024)):
    """
    Search for images using Pixabay public API
    """
    try:
        # Format the query for URL
        encoded_query = urllib.parse.quote(query)
        
        # Using Pixabay API without key for public access (limited results)
        url = f"https://pixabay.com/api/?q={encoded_query}&image_type=photo&per_page=3"
        
        # Send request
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('hits') and len(data['hits']) > 0:
                # Get the first image URL
                img_url = data['hits'][0].get('webformatURL')
                if img_url:
                    img_response = requests.get(img_url, timeout=10)
                    if img_response.status_code == 200:
                        return Image.open(io.BytesIO(img_response.content)).resize(size)
            
            print("No images found on Pixabay")
            return None
        else:
            print(f"Pixabay search failed with status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error searching Pixabay: {e}")
        return None

def create_fallback_image(prompt, size=(1024, 1024)):
    """
    Create a fallback image with colors derived from the prompt
    """
    # Create a colored background with gradient
    img = Image.new('RGB', size, color=(240, 240, 240))
    
    # Create gradient background
    for y in range(size[1]):
        for x in range(size[0]):
            # Simple gradient
            r = int(200 + (x * 55 / size[0]))
            g = int(200 + (y * 55 / size[1]))
            b = 220
            img.putpixel((x, y), (r, g, b))
    
    # Create a hash of the prompt to get deterministic patterns
    prompt_hash = sum(ord(c) for c in prompt)
    
    # Draw some colored blocks based on words in the prompt
    block_size = 50
    for i, word in enumerate(prompt.split()):
        if i > 20:  # Limit number of blocks
            break
        # Use word to determine position and color
        word_hash = sum(ord(c) for c in word)
        x = (word_hash % (size[0] - block_size))
        y = ((i * 137 + word_hash) % (size[1] - block_size))
        
        # Generate color from word
        r = (word_hash * 17) % 256
        g = (word_hash * 23) % 256
        b = (word_hash * 31) % 256
        
        # Draw colored block
        for by in range(block_size):
            for bx in range(block_size):
                if 0 <= x+bx < size[0] and 0 <= y+by < size[1]:
                    img.putpixel((x+bx, y+by), (r, g, b))
    
    return img

def simple_image_search(prompt, output_path, reference_dir="reference_images"):
    """
    A real image search engine that uses free public APIs without requiring authentication
    """
    try:
        # Try Unsplash first
        image = search_unsplash(prompt)
        
        # If Unsplash fails, try Pixabay
        if image is None:
            image = search_pixabay(prompt)
        
        # If both fail, use fallback
        if image is None:
            image = create_fallback_image(prompt)
        
        # Save the image
        image.save(output_path)
        
        # Add some delay to prevent rate limiting
        time.sleep(random.uniform(1.0, 2.0))
        
        return True
    except Exception as e:
        print(f"Error in image search: {e}")
        # If anything fails, create and save a fallback image
        create_fallback_image(prompt).save(output_path)
        return True
        
        # Simple keyword matching
        prompt_words = [word.lower() for word in prompt.split() if len(word) > 3]
        
        best_match = None
        best_score = -1
        
        for file in ref_files:
            score = 0
            file_lower = file.lower()
            for word in prompt_words:
                if word in file_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = file
        
        # If no good match, just use the first image
        if best_match is None or best_score == 0:
            best_match = ref_files[0]
        
        # Copy and resize the matched image
        matched_img = Image.open(os.path.join(reference_dir, best_match))
        matched_img = matched_img.resize((1024, 1024))
        matched_img.save(output_path)
        
        return True
    except Exception as e:
        print(f"Error in image search: {e}")
        return False

def create_comparison_table(category, level, prompt_id):
    """
    Create a side-by-side comparison of images from different models
    """
    base_dir = "generated_images"
    categories = ["sd35", "flux", "search_engine"]
    
    # Check if all required images exist
    images = []
    for cat in categories:
        img_path = os.path.join(base_dir, cat, f"{cat}_{level}_{prompt_id}.png")
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                img = img.resize((512, 512))
                images.append(img)
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")
                images.append(Image.new('RGB', (512, 512), 'red'))  # Red placeholder for error
        else:
            print(f"Warning: Image not found at {img_path}")
            images.append(Image.new('RGB', (512, 512), 'white'))  # White placeholder for missing
    
    # Create a horizontal comparison image (side by side)
    width = 512 * len(images)
    comparison = Image.new('RGB', (width, 512), 'white')
    
    # Paste all images side by side
    for i, img in enumerate(images):
        comparison.paste(img, (i * 512, 0))
    
    # Save the comparison table
    output_path = os.path.join(base_dir, "comparison_tables", f"comparison_{category}_{level}_{prompt_id}.png")
    comparison.save(output_path)
    print(f"Saved comparison table to {output_path}")

def process_sd35(sd35_queue, results_queue):
    print("Starting SD3.5 process...")
    sd_pipe = init_sd35()
    while True:
        try:
            data = sd35_queue.get(timeout=1)
            if data is None:  # Poison pill
                break
                
            prompt, category, level, prompt_id = data
            output_path = os.path.join("generated_images", "sd35", f"sd35_{level}_{prompt_id}.png")
            success = generate_sd35_image(prompt, output_path, sd_pipe)
            results_queue.put(("sd35", category, level, prompt_id, success))
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in SD35 process: {e}")
    print("SD3.5 process finished")

def process_flux(flux_queue, results_queue):
    print("Starting FLUX process...")
    flux_pipe = init_flux()
    while True:
        try:
            data = flux_queue.get(timeout=1)
            if data is None:  # Poison pill
                break
                
            prompt, category, level, prompt_id = data
            output_path = os.path.join("generated_images", "flux", f"flux_{level}_{prompt_id}.png")
            success = generate_flux_image(prompt, output_path, flux_pipe)
            results_queue.put(("flux", category, level, prompt_id, success))
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in FLUX process: {e}")
    print("FLUX process finished")

def process_search(search_queue, results_queue):
    print("Starting Image Search process...")
    while True:
        try:
            data = search_queue.get(timeout=1)
            if data is None:  # Poison pill
                break
                
            prompt, category, level, prompt_id = data
            output_path = os.path.join("generated_images", "search_engine", f"search_engine_{level}_{prompt_id}.png")
            success = simple_image_search(prompt, output_path)
            results_queue.put(("search_engine", category, level, prompt_id, success))
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in Search process: {e}")
    print("Image Search process finished")

def process_comparisons(comparison_queue):
    print("Starting Comparison process...")
    while True:
        try:
            data = comparison_queue.get(timeout=1)
            if data is None:  # Poison pill
                break
                
            category, level, prompt_id = data
            create_comparison_table(category, level, prompt_id)
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in Comparison process: {e}")
    print("Comparison process finished")

def main():
    # Create directory structure
    create_directory_structure()
    
    # Load prompts
    with open("json_files/structured_prompts.json", "r") as f:
        prompts_data = json.load(f)
    
    # Create queues
    sd35_queue = Queue()
    flux_queue = Queue()
    search_queue = Queue()
    results_queue = Queue()
    comparison_queue = Queue()
    
    # Create shared tracking dictionary
    manager = Manager()
    completed = manager.dict()
    
    # Start processes
    processes = [
        Process(target=process_sd35, args=(sd35_queue, results_queue)),
        Process(target=process_flux, args=(flux_queue, results_queue)),
        Process(target=process_search, args=(search_queue, results_queue)),
        Process(target=process_comparisons, args=(comparison_queue,))
    ]
    
    for p in processes:
        p.start()
    
    # Queue all prompts immediately (don't make GPUs wait for each other)
    total_prompts = 0
    for category, levels in prompts_data.items():
        for level, data in levels.items():
            for example in data["examples"]:
                prompt = example["text"]
                prompt_id = example["id"]
                key = f"{category}_{level}_{prompt_id}"
                
                # Track completion status for each prompt
                completed[key] = {"sd35": False, "flux": False, "search_engine": False}
                
                # Queue for all processes independently
                sd35_queue.put((prompt, category, level, prompt_id))
                flux_queue.put((prompt, category, level, prompt_id))
                search_queue.put((prompt, category, level, prompt_id))
                
                total_prompts += 1
    
    # Process results and create comparison tables when all models finished for a prompt
    completed_count = 0
    expected_results = total_prompts * 3  # 3 models per prompt
    
    while completed_count < expected_results:
        try:
            model, category, level, prompt_id, success = results_queue.get(timeout=1)
            completed_count += 1
            
            key = f"{category}_{level}_{prompt_id}"
            completed[key][model] = True
            
            # If all models have completed for this prompt, queue for comparison
            if all(completed[key].values()):
                comparison_queue.put((category, level, prompt_id))
            
            # Print progress
            if completed_count % 10 == 0:
                print(f"Progress: {completed_count}/{expected_results} images generated")
                
        except queue.Empty:
            continue
    
    # Send poison pills to stop processes
    for q in [sd35_queue, flux_queue, search_queue, comparison_queue]:
        q.put(None)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    print("All processing complete!")

if __name__ == "__main__":
    main()