import os
import json
import shutil
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import concurrent.futures

# Source and destination directories
src_root = "/data/cloth1m/v3/cloth1m_data_v3/images"
dst_root = "/data/cloth1m/v4/cloth1m_data_v4"
images_root = os.path.join(dst_root, "images")
json_file = os.path.join(dst_root, "metadata.json")

# Number of parallel workers for image conversion
NUM_WORKERS = 128

# Create destination directories if they don't exist
os.makedirs(dst_root, exist_ok=True)
os.makedirs(images_root, exist_ok=True)

# Function to process a single image
def process_image(data):
    # Group by the first 3 digits of the ID
    group_id = data["global_id"][:3]
    
    # Create destination directory
    dst_dir = os.path.join(images_root, group_id)
    os.makedirs(dst_dir, exist_ok=True)
    
    # Destination file path
    dst_file = os.path.join(dst_dir, f"{data['global_id']}.png")
    
    # Copy and convert the file from JPG to PNG
    try:
        img = Image.open(data["src_path"])
        img.save(dst_file, format="PNG")
    except Exception as e:
        print(f"Error processing {data['src_path']}: {e}")
        # Fallback to simple copy if conversion fails
        jpg_dst = os.path.splitext(dst_file)[0] + ".jpg"
        shutil.copyfile(data["src_path"], jpg_dst)
        # Update the destination file to reflect the actual file created
        dst_file = jpg_dst
    
    # Return metadata for this image
    return data["global_id"], {
        "id": data["global_id"],
        "class": data["class"],
        "color": data["color"],
        "material": data["material"],
        "pattern": data["pattern"],
        "path": os.path.relpath(dst_file, dst_root),
        "original_path": data["src_path"]
    }

# Collect all image information
print("Scanning source directory...")
image_data = []
global_counter = 0

for clothing_type in sorted(os.listdir(src_root)):
    clothing_type_path = os.path.join(src_root, clothing_type)
    if not os.path.isdir(clothing_type_path):
        continue
    
    for color_material_pattern in sorted(os.listdir(clothing_type_path)):
        cmp_path = os.path.join(clothing_type_path, color_material_pattern)
        if not os.path.isdir(cmp_path):
            continue
        
        # Parse color, material, and pattern
        parts = color_material_pattern.split('_')
        if len(parts) >= 3:
            color = parts[0]
            material = parts[1]
            pattern = '_'.join(parts[2:])  # In case pattern has underscores
        else:
            print(f"Skipping folder with unexpected format: {color_material_pattern}")
            continue
        
        for image_file in sorted(os.listdir(cmp_path)):
            if not image_file.endswith('.jpg'):
                continue
            
            # Full path to the image
            image_path = os.path.join(cmp_path, image_file)
            
            # Generate a new globally unique ID
            global_counter += 1
            global_id = f"{global_counter:07d}"  # 7-digit ID with leading zeros
            
            # Store information for processing
            image_data.append({
                "src_path": image_path,
                "global_id": global_id,
                "class": clothing_type,
                "color": color,
                "material": material,
                "pattern": pattern
            })

print(f"Found {len(image_data)} images. Processing...")

# Process images in parallel
metadata = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    future_to_data = {executor.submit(process_image, data): data for data in image_data}
    
    # Show progress
    for future in tqdm(concurrent.futures.as_completed(future_to_data), total=len(image_data)):
        try:
            global_id, meta = future.result()
            metadata[global_id] = meta
        except Exception as e:
            data = future_to_data[future]
            print(f"Error processing image {data['src_path']}: {e}")

# Save metadata to JSON file
print(f"Saving metadata to {json_file}...")
with open(json_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Reorganization complete. {len(metadata)} images processed. Metadata saved to {json_file}")