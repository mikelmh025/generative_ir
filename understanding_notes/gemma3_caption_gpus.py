from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import os

def setup_captioning_model(model_id="google/gemma-3-27b-it", gpu_ids="0,1"):
    """
    Set up the Gemma 3 model for image captioning.
    
    Args:
        model_id (str): HuggingFace model ID
        gpu_ids (str): Comma-separated GPU IDs to use
        
    Returns:
        tuple: (model, processor) ready for inference
    """
    # Set visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    # Load model with automatic device mapping
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, 
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)
    
    return model, processor

def get_detailed_caption(
    image_path_or_url, 
    model, 
    processor, 
    prompt="Generate a detailed, descriptive caption of this image for use as a prompt in a diffusion model. \
        Focus on visual elements, style, composition, colors, and details. Generate ONLY a detailed caption \
        for this image. DO NOT include any introductory text or explanations. The output should be the \
        raw caption text ready for a diffusion model.", 
    max_tokens=77  # Set your desired maximum here
):
    """
    Generate a detailed caption for an image, optimized for diffusion model input.
    
    Args:
        image_path_or_url (str): Path or URL to the image
        model: The loaded Gemma 3 model
        processor: The loaded processor
        prompt (str): The instruction prompt to get the desired caption style
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: The generated caption text
    """
    # Update the system message to encourage brevity
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": f"You are an image captioning assistant that creates concise but detailed descriptions for use in text-to-image diffusion models. Keep your description under {max_tokens} tokens."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path_or_url},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Process with chat template
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True, 
        return_tensors="pt"
    )
    
    # Move inputs to the same device as the first layer of the model
    first_device = next(iter(model.hf_device_map.values())) if hasattr(model, "hf_device_map") else model.device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}
    
    # Track input length for extraction
    input_len = inputs["input_ids"].shape[-1]
    
    # Generate text with strict max_new_tokens
    with torch.inference_mode():
        generation = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,  # This is critical - it won't generate more than this
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            # Add early stopping to encourage shorter completions
            early_stopping=True,
            # Using smaller penalty for repetition to help find natural endpoints
            repetition_penalty=1.1  
        )
        generation = generation[0][input_len:]
    
    # Decode the output
    caption = processor.decode(generation, skip_special_tokens=True)
    
    return caption.strip()

# Example usage
if __name__ == "__main__":
    # Initialize model once
    model, processor = setup_captioning_model(model_id="google/gemma-3-12b-it", gpu_ids="0")
    
    # Example image URL
    # image_url = "https://example.com/your-image.jpg"
    image_path = "/data/mscoco/unlabeled2017/000000483410.jpg"
    
    # Get diffusion-ready caption
    caption = get_detailed_caption(image_path, model, processor, max_tokens=77)
    
    print("CAPTION FOR DIFFUSION MODEL:")
    print("-" * 50)
    print(caption)
    print("-" * 50)