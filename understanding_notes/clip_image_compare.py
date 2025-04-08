import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import numpy as np

# Load CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def get_image_embedding(image_path):
    """Get the CLIP embedding for an image"""
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    
    # Normalize embedding
    embedding = outputs.detach().numpy()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def plot_similarity_matrix(similarity_matrix, image_paths):
    """Plot similarity matrix as a heatmap, without showing the diagonal"""
    # Get image names from paths for labels
    image_names = [os.path.basename(path) for path in image_paths]
    
    # Create a mask for the diagonal
    mask = np.eye(len(image_paths), dtype=bool)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix, 
        annot=True, 
        xticklabels=image_names,
        yticklabels=image_names,
        cmap='viridis',
        vmin=0, 
        vmax=1,
        mask=mask  # This masks out the diagonal elements
    )
    plt.title('CLIP Image Embedding Similarity (Diagonal Hidden)')
    plt.tight_layout()
    # plt.show()
    plt.savefig("understanding_notes/similarity_matrix.png")

# Example usage
def compare_images(image_paths):
    """Compare multiple images using CLIP embeddings"""
    if len(image_paths) < 2:
        print("Please provide at least 2 images to compare")
        return
    
    # Get embeddings for all images
    embeddings = []
    for path in image_paths:
        embedding = get_image_embedding(path)
        embeddings.append(embedding)
    
    # Stack all embeddings
    embeddings_matrix = np.vstack(embeddings)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings_matrix)
    
    # Plot similarity
    plot_similarity_matrix(similarity_matrix, image_paths)
    
    # Return top similar pairs
    return similarity_matrix

gt_image_path = '/data/mscoco/unlabeled2017/000000483410.jpg'
fake_image_dir = "understanding_notes/sd35"
fake_image_paths = [os.path.join(fake_image_dir, img) for img in os.listdir(fake_image_dir)]
fake_image_paths.sort
image_paths = [gt_image_path] + fake_image_paths

# Run comparison
similarity_matrix = compare_images(image_paths)
print(f"Similarity matrix shape: {similarity_matrix.shape}")

# You can also get the embedding for just one image for further processing
example_embedding = get_image_embedding(image_paths[0])
print(f"Image embedding shape: {example_embedding.shape}")