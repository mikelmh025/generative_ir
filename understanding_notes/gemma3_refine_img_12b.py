from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import os

def setup_gemma_model(model_id="google/gemma-3-27b-it", gpu_ids="0,1"):
    """
    Set up the Gemma 3 model for image tasks.
    
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
    max_tokens=77
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
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            early_stopping=True,
            repetition_penalty=1.1  
        )
        generation = generation[0][input_len:]
    
    # Decode the output
    caption = processor.decode(generation, skip_special_tokens=True)
    
    return caption.strip()

def compare_image_with_fake(
    real_image_path_or_url,
    fake_image_path_or_url,
    caption,
    model,
    processor,
    prompt=None,
    max_tokens=200
):
    """
    Compare a real image with its fake/generated counterpart and analyze the caption.
    
    Args:
        real_image_path_or_url (str): Path or URL to the real image
        fake_image_path_or_url (str): Path or URL to the fake/generated image
        caption (str): The caption to analyze
        model: The loaded Gemma 3 model
        processor: The loaded processor
        prompt (str): Custom prompt for analysis (optional)
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: Analysis of differences between images and caption quality
    """
    if prompt is None:
        prompt = f"""I'll show you two images: 
1. The first image is the original real image
2. The second image is a generated fake or modified version

The current caption for these images is:
"{caption}"

Please analyze:
1. What are the key differences between the real and fake images?
2. What elements are present in the real image but missing or altered in the fake?
3. What elements are present in the fake but don't exist or look different in the real?
4. How accurate is the caption for the real image? What's missing or incorrect?
5. What would make this caption better represent the real image?

Be thorough and specific in your analysis."""

    # Set up messages for the comparison task
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert image analyst specializing in comparing real images with AI-generated or modified versions. You provide detailed analyses of differences and caption accuracy."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": real_image_path_or_url},
                {"type": "image", "image": fake_image_path_or_url},
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
    
    # Generate text
    with torch.inference_mode():
        generation = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        generation = generation[0][input_len:]
    
    # Decode the output
    analysis = processor.decode(generation, skip_special_tokens=True)
    
    return analysis.strip()

def refine_caption_with_comparison(
    real_image_path_or_url,
    fake_image_path_or_url,
    original_caption,
    comparison_analysis,
    model,
    processor,
    prompt=None,
    max_tokens=100
):
    """
    Refine an image caption based on comparison analysis between real and fake images.
    
    Args:
        real_image_path_or_url (str): Path or URL to the real image
        fake_image_path_or_url (str): Path or URL to the fake image
        original_caption (str): The original caption to refine
        comparison_analysis (str): Analysis from the comparison function
        model: The loaded Gemma 3 model
        processor: The loaded processor
        prompt (str): Custom prompt for refinement (optional)
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: Refined caption
    """
    if prompt is None:
        prompt = f"""Original caption: "{original_caption}"

Analysis comparing real and fake images: {comparison_analysis}

Please create an improved caption for the REAL image (the first image). The caption should:
1. Accurately describe elements present in the real image but not in the fake
2. Remove or correct descriptions of elements that only appear in the fake image
3. Be detailed, precise, and focus on the real image's style, composition, and important visual elements
4. Be suitable for a text-to-image diffusion model
5. Be concise yet comprehensive

Generate ONLY the improved caption text with no additional explanation."""

    # Set up messages for the refinement task
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": f"You are an expert image captioning assistant that refines and improves image descriptions. Focus on accurately describing the real image, not the fake version. Keep your description under {max_tokens} tokens."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": real_image_path_or_url},
                {"type": "image", "image": fake_image_path_or_url},
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
    
    # Generate text
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
    refined_caption = processor.decode(generation, skip_special_tokens=True)
    
    return refined_caption.strip()

def complete_caption_pipeline_with_fake(
    real_image_path_or_url,
    fake_image_path_or_url,
    model,
    processor,
    initial_caption_prompt=None,
    comparison_prompt=None,
    refinement_prompt=None,
    max_tokens_initial=77,
    max_tokens_comparison=300,
    max_tokens_refined=77
):
    """
    Complete pipeline for generating, comparing with fake, and refining an image caption.
    
    Args:
        real_image_path_or_url (str): Path or URL to the real image
        fake_image_path_or_url (str): Path or URL to the fake/generated image
        model: The loaded Gemma 3 model
        processor: The loaded processor
        initial_caption_prompt (str): Custom prompt for initial caption generation
        comparison_prompt (str): Custom prompt for image comparison
        refinement_prompt (str): Custom prompt for caption refinement
        max_tokens_initial (int): Max tokens for initial caption
        max_tokens_comparison (int): Max tokens for comparison analysis
        max_tokens_refined (int): Max tokens for refined caption
        
    Returns:
        dict: Contains original caption, comparison analysis, and refined caption
    """
    # Step 1: Generate initial caption for the real image
    initial_caption = get_detailed_caption(
        real_image_path_or_url,
        model,
        processor,
        prompt=initial_caption_prompt,
        max_tokens=max_tokens_initial
    )
    
    # Step 2: Compare real and fake images and analyze the caption
    comparison_analysis = compare_image_with_fake(
        real_image_path_or_url,
        fake_image_path_or_url,
        initial_caption,
        model,
        processor,
        prompt=comparison_prompt,
        max_tokens=max_tokens_comparison
    )
    
    # Step 3: Refine caption based on comparison analysis
    refined_caption = refine_caption_with_comparison(
        real_image_path_or_url,
        fake_image_path_or_url,
        initial_caption,
        comparison_analysis,
        model,
        processor,
        prompt=refinement_prompt,
        max_tokens=max_tokens_refined
    )
    
    return {
        "initial_caption": initial_caption,
        "comparison_analysis": comparison_analysis,
        "refined_caption": refined_caption
    }

# Example usage
if __name__ == "__main__":
    # Initialize model once
    model, processor = setup_gemma_model(model_id="google/gemma-3-4b-it", gpu_ids="0")
    
    # Example image paths
    real_image_path = "/home/minghao/Documents/diffusion_ir/genIR_images/sd35/unlabeled2017/000000029869.jpg"
    fake_image_path = "/home/minghao/Documents/diffusion_ir/genIR_images/sd35/unlabeled2017/000000029869_0.jpg"
    
    # Full pipeline with default settings
    result = complete_caption_pipeline_with_fake(real_image_path, fake_image_path, model, processor)
    
    print("INITIAL CAPTION (REAL IMAGE):")
    print("-" * 50)
    print(result["initial_caption"])
    print("-" * 50)
    
    print("\nCOMPARISON ANALYSIS (REAL vs FAKE):")
    print("-" * 50)
    print(result["comparison_analysis"])
    print("-" * 50)
    
    print("\nREFINED CAPTION (OPTIMIZED FOR REAL IMAGE):")
    print("-" * 50)
    print(result["refined_caption"])
    print("-" * 50)
    
    # Optional: Generate a separate caption for the fake image if needed
    """
    fake_image_caption = get_detailed_caption(
        fake_image_path,
        model,
        processor
    )
    
    print("\nFAKE IMAGE CAPTION:")
    print("-" * 50)
    print(fake_image_caption)
    print("-" * 50)
    """