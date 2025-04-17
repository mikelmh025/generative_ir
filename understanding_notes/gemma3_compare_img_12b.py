from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import os

def setup_comparison_model(model_id="google/gemma-3-27b-it", gpu_ids="0,1"):
    """
    Set up the Gemma 3 model for image comparison.
    
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

def compare_images(
    image1_path_or_url, 
    image2_path_or_url,
    model, 
    processor, 
    prompt="Compare these two images in detail. Describe their similarities and differences in terms of visual elements, style, composition, colors, and objects present. Focus on meaningful distinctions.", 
    max_tokens=200  # Increased for comparison which requires more detail
):
    """
    Compare two images using the Gemma 3 model.
    
    Args:
        image1_path_or_url (str): Path or URL to the first image
        image2_path_or_url (str): Path or URL to the second image
        model: The loaded Gemma 3 model
        processor: The loaded processor
        prompt (str): The instruction prompt for image comparison
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: The generated comparison text
    """
    # Update the system message for comparison task
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an image analysis assistant that compares images in detail, highlighting similarities and differences. Be thorough but precise."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1_path_or_url},
                {"type": "image", "image": image2_path_or_url},
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
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            early_stopping=True,
            repetition_penalty=1.1  
        )
        generation = generation[0][input_len:]
    
    # Decode the output
    comparison = processor.decode(generation, skip_special_tokens=True)
    
    return comparison.strip()

# Example usage
if __name__ == "__main__":
    # Initialize model once
    model, processor = setup_comparison_model(model_id="google/gemma-3-12b-it", gpu_ids="0")
    
    # Example image paths
    image1_path = "/home/minghao/Documents/diffusion_ir/results_genir/caption_refinement_output/image_000000000648/original.jpg"   # Demo purpose, I am note sure if the fake image here is actually wrong
    image2_path = "/home/minghao/Documents/diffusion_ir/results_genir/caption_refinement_output/image_000000000648/generated_images/generated_0.jpg"  # Another image for comparison
    
    # Original caption
    caption_paht = "/home/minghao/Documents/diffusion_ir/results_genir/caption_refinement_output/image_000000000648/captions/caption_0.txt"
    with open(caption_paht, "r") as f:
        original_caption = f.read()
    print("Original Caption:")
    print("-" * 50)
    print(original_caption)
    print("-" * 50)
    
    
    
    prompt = f"Here is the Origional caption of the first image: {original_caption}. Based on the caption someone drawed the second image. Compare the content difference in between the two images. Only tell me what's the difference. Concise and clear."
    
    diff = compare_images(
        image1_path, 
        image2_path, 
        model, 
        processor,
        prompt=prompt,
        max_tokens=2000
    )
    print("IMAGE COMPARISON RESULT:")
    print("-" * 50)
    print(diff)
    print("-" * 50)
    
    prompt = f"Here is the Origional caption of the first image: {original_caption}. Here is the difference between the two images: {diff}. Please help me to generate a new caption for the image, so that we can generate the original exactly the same. Focus on the content difference in between the two images. No explanation, no other information. Just tell me the new caption."
    new_description = compare_images(
        image1_path, 
        image2_path, 
        model, 
        processor,
        prompt=prompt,
        max_tokens=300
    )
    
    print("Original Caption:")
    print("-" * 50)
    print(original_caption)
    print("-" * 50)
    
    print("NEW CAPTION:")
    print("-" * 50)
    print(new_description)
    print("-" * 50)
    
    # diff = compare_images(
    #     image1_path, 
    #     image2_path, 
    #     model, 
    #     processor,
    #     prompt="I am trying to understand the content difference in between these two images. Help me on this.",
    #     max_tokens=500
    # )
    
    # print("IMAGE COMPARISON RESULT:")
    # print("-" * 50)
    # print(diff)
    # print("-" * 50)
    
    # new_description = compare_images(
    #     image1_path, 
    #     image2_path, 
    #     model, 
    #     processor,
    #     prompt=f"Difference on the images{diff}, Original caption: {original_caption}. Please help me to generate a new caption for the image.",
    #     max_tokens=500
    # )    
    
    # # Get comparison result
    # comparison = compare_images(image1_path, image2_path, model, processor)
    
    # print("IMAGE COMPARISON RESULT:")
    # print("-" * 50)
    # print(comparison)
    # print("-" * 50)
    
    # # You can also customize the comparison prompt for specific use cases
    # similarity_score = compare_images(
    #     image1_path, 
    #     image2_path, 
    #     model, 
    #     processor,
    #     prompt="Compare these two images and rate their visual similarity on a scale from 1 to 10, where 1 means completely different and 10 means nearly identical. Explain your rating.",
    #     max_tokens=150
    # )
    
    # print("\nSIMILARITY SCORE:")
    # print("-" * 50)
    # print(similarity_score)
    # print("-" * 50)