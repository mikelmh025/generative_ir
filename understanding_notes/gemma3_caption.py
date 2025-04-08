from transformers import pipeline
import torch

# Create the pipeline
pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-27b-it",
    device="cuda",  # Use "cpu" if no GPU is available
    torch_dtype=torch.bfloat16
)

# Prepare message format using chat template for instruction-tuned model
def caption_image_with_pipeline(image_path_or_url, prompt="Describe this image in detail."):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path_or_url},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    output = pipe(text=messages, max_new_tokens=300)
    return output[0]["generated_text"][-1]["content"]

# Example usage
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
caption = caption_image_with_pipeline(image_url)
print(f"Caption: {caption}")

# Get a more specific description
specific_caption = caption_image_with_pipeline(
    image_url, 
    "What objects are visible in this image? List them all."
)
print(f"Objects in image: {specific_caption}")