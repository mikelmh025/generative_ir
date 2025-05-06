import torch
import json
import os
import shutil
from pathlib import Path
from diffusers import StableDiffusion3Pipeline, FluxPipeline
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch.multiprocessing as mp
from tqdm import tqdm
from PIL import Image
import logging
from typing import Dict, List, Callable, Union, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CaptionRefinementPipeline:
    """
    Enhanced pipeline that:
    1. Reads original images from a directory
    2. Generates initial captions using Gemma3
    3. Uses captions to generate new images with diffusion models
    4. Compares original and generated images to refine captions
    5. Optionally repeats the generate-refine process for multiple iterations
    6. Saves all intermediate results including captions, images, and comparisons
    """
    
    def __init__(
        self, 
        input_dir: str = "/data/mscoco",
        queries_path: str = "ChatIR/dialogues/VisDial_v1.0_queries_val.json",
        output_dir: str = "caption_refinement_output",
        image_subdirs: List[str] = ["train2017", "val2017", "unlabeled2017"],
        max_images: Optional[int] = None,
        caption_model_id: str = "google/gemma-3-12b-it",
        comparison_model_id: str = "google/gemma-3-12b-it",  # Can be same as caption model
        diffusion_model: str = "sd35",  # "sd35" or "flux"
        caption_gpu_id: int = 0,
        diffusion_gpu_id: int = 1,
        refinement_rounds: int = 2,  # Number of refinement iterations
    ):
        """
        Initialize the pipeline.
        
        Args:
            input_dir: Directory containing original images
            queries_path: Path to JSON file with queries/dialogues
            output_dir: Directory to save generated content
            image_subdirs: Subdirectories in input_dir containing images to process
            max_images: Maximum number of images to process
            caption_model_id: Gemma model ID for captioning
            comparison_model_id: Gemma model ID for image comparison
            diffusion_model: Type of diffusion model to use ('sd35' or 'flux')
            caption_gpu_id: GPU ID for captioning model
            diffusion_gpu_id: GPU ID for diffusion model
            refinement_rounds: Number of caption refinement iterations
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_subdirs = image_subdirs
        self.max_images = max_images
        self.caption_model_id = caption_model_id
        self.comparison_model_id = comparison_model_id
        self.diffusion_model = diffusion_model
        self.caption_gpu_id = caption_gpu_id
        self.diffusion_gpu_id = diffusion_gpu_id
        self.refinement_rounds = refinement_rounds
        
        # Load queries
        self.queries = self._load_json(queries_path)
        
        # Create output directories
        self.original_images_dir = self.output_dir / "original_images"
        self.captions_dir = self.output_dir / "captions"
        self.generated_images_dir = self.output_dir / "generated_images"
        self.comparisons_dir = self.output_dir / "comparisons"
        self.dialogue_dir = self.output_dir / "dialogues"
        
        os.makedirs(self.original_images_dir, exist_ok=True)
        os.makedirs(self.captions_dir, exist_ok=True)
        os.makedirs(self.generated_images_dir, exist_ok=True)
        os.makedirs(self.comparisons_dir, exist_ok=True)
        os.makedirs(self.dialogue_dir, exist_ok=True)
        
        # Initialize models
        self.caption_model = None
        self.caption_processor = None
        self.comparison_model = None
        self.comparison_processor = None
        self.diffusion_pipe = None
        
    def _load_json(self, path: str) -> Dict:
        """Load and parse a JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Could not find queries file at {path}. Will use image directory scanning instead.")
            return {}
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {path}")
    
    def _setup_caption_model(self):
        """Set up the Gemma3 model for image captioning on GPU 0."""
        logger.info(f"Setting up captioning model on GPU {self.caption_gpu_id}")
        
        # Load model with explicit device mapping
        self.caption_model = Gemma3ForConditionalGeneration.from_pretrained(
            self.caption_model_id, 
            device_map=f"cuda:{self.caption_gpu_id}",
            torch_dtype=torch.bfloat16
        ).eval()
        
        # Load processor
        self.caption_processor = AutoProcessor.from_pretrained(self.caption_model_id)
        
        logger.info(f"Captioning model loaded on GPU {self.caption_gpu_id}")
    
    def _setup_comparison_model(self):
        """Set up the Gemma3 model for image comparison on GPU 0."""
        logger.info(f"Setting up comparison model on GPU {self.caption_gpu_id}")
        
        # If using the same model as captioning, reuse it
        if self.comparison_model_id == self.caption_model_id and self.caption_model is not None:
            self.comparison_model = self.caption_model
            self.comparison_processor = self.caption_processor
            logger.info(f"Reusing captioning model for comparison")
        else:
            # Load a separate model
            self.comparison_model = Gemma3ForConditionalGeneration.from_pretrained(
                self.comparison_model_id, 
                device_map=f"cuda:{self.caption_gpu_id}",
                torch_dtype=torch.bfloat16
            ).eval()
            
            # Load processor
            self.comparison_processor = AutoProcessor.from_pretrained(self.comparison_model_id)
            
            logger.info(f"Dedicated comparison model loaded on GPU {self.caption_gpu_id}")
    
    def _setup_diffusion_model(self):
        """Set up the diffusion model on GPU 1."""
        logger.info(f"Setting up diffusion model on GPU {self.diffusion_gpu_id}")
        device = f"cuda:{self.diffusion_gpu_id}"
        
        if self.diffusion_model == "sd35":
            # Set torch compute dtype
            torch.set_default_dtype(torch.float16)
            self.diffusion_pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                torch_dtype=torch.float16,
                variant="fp16"
            )
            self.diffusion_pipe = self.diffusion_pipe.to(device)
        elif self.diffusion_model == "flux":
            self.diffusion_pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", 
                torch_dtype=torch.bfloat16
            )
            self.diffusion_pipe.enable_model_cpu_offload(gpu_id=self.diffusion_gpu_id)
        else:
            raise ValueError(f"Unsupported diffusion model: {self.diffusion_model}. Choose either 'sd35' or 'flux'")
        
        logger.info(f"Diffusion model loaded on GPU {self.diffusion_gpu_id}")
    
    def _generate_caption(self, image_path: str, max_tokens: int = 200, 
                          refinement_round: int = 0, previous_caption: str = None,
                          comparison_feedback: str = None) -> str:
        """
        Generate or refine a caption for an image using Gemma3.
        
        Args:
            image_path: Path to the image
            max_tokens: Maximum number of tokens for caption
            refinement_round: Current refinement iteration (0 = initial caption)
            previous_caption: Previous caption to refine (if refinement_round > 0)
            comparison_feedback: Feedback from image comparison (if refinement_round > 0)
            
        Returns:
            str: Generated or refined caption
        """
        # Different prompts based on whether this is initial caption or refinement
        if refinement_round == 0:
            prompt = (
                "Generate a detailed, descriptive caption of this image for use as a prompt in a diffusion model. "
                "Focus on visual elements, style, composition, colors, and details. Generate ONLY a detailed caption "
                "for this image. DO NOT include any introductory text or explanations. The output should be the "
                "raw caption text ready for a diffusion model."
            )
            
            # System message for initial captioning
            system_message = f"You are an image captioning assistant that creates concise but detailed descriptions for use in text-to-image diffusion models. Keep your description under {max_tokens} tokens."
            # system_message = f"You are an image captioning assistant that creates detailed descriptions for use in text-to-image diffusion models. Keep your description under {max_tokens} tokens. Try to use up the token limit."
            
            # Just the image in the user message for initial captioning
            user_content = [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        else:
            # For refinement, include previous caption and comparison feedback
            prompt = (
                f"I need you to refine the following caption for this image. The caption was used to generate a "
                f"synthetic version of this image, but the result didn't fully match. Here is the comparison feedback: "
                f"\n\n{comparison_feedback}\n\n"
                f"Previous caption: '{previous_caption}'\n\n"
                f"Please provide an improved caption that would help generate an image more similar to the original. "
                f"Focus on correcting the issues identified in the comparison. Generate ONLY the refined caption, "
                f"without any explanations or surrounding text."
            )
            
            # System message for refinement
            system_message = (
                f"You are a caption refinement assistant. Your task is to refine image captions based on feedback "
                f"about the differences between original images and images generated from captions. "
                f"Keep your refined caption under {max_tokens} tokens."
            )
            
            # Include the image in the user message for refinement
            user_content = [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        # Process with chat template
        inputs = self.caption_processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        )
        
        # Move inputs to GPU
        caption_device = f"cuda:{self.caption_gpu_id}"
        inputs = {k: v.to(caption_device) for k, v in inputs.items()}
        
        # Track input length for extraction
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate text
        with torch.inference_mode():
            generation = self.caption_model.generate(
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
        caption = self.caption_processor.decode(generation, skip_special_tokens=True)
        
        return caption.strip()
    
    def _compare_images(self, original_caption: str, original_image_path: str, generated_image_path: str, max_tokens: int = 400) -> str:
        """
        Compare original and generated images using the Gemma3 model.
        
        Args:
            original_image_path: Path to the original image
            generated_image_path: Path to the generated image
            max_tokens: Maximum number of tokens for the comparison text
            
        Returns:
            str: The generated comparison text
        """
        # prompt = (
        #     "Compare these two images in detail. The first image is the original, and the second is a "
        #     "synthetically generated version. Describe their similarities and differences, focusing on: "
        #     "1. Objects and subjects present or missing in each image\n"
        #     "2. Spatial layout and composition differences\n"
        #     "3. Color, lighting, and style variations\n"
        #     "4. Background elements and details\n"
        #     "5. Key aspects of the original image that are not captured in the generated version\n\n"
        #     "Be specific about what would need to be fixed in the caption to make the generated image more "
        #     "similar to the original."
        # )
        prompt = f"Here is the Origional caption of the first image: {original_caption}. Based on the caption someone drawed the second image. Compare the content difference in between the two images. Only tell me what's the difference. Concise and clear."
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an image analysis assistant that compares original images with their synthetically generated versions, highlighting important differences that need to be addressed in caption refinement."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_image_path},
                    {"type": "image", "image": generated_image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process with chat template
        inputs = self.comparison_processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        )
        
        # Move inputs to GPU
        comparison_device = f"cuda:{self.caption_gpu_id}"
        inputs = {k: v.to(comparison_device) for k, v in inputs.items()}
        
        # Track input length for extraction
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate text
        with torch.inference_mode():
            generation = self.comparison_model.generate(
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
        comparison = self.comparison_processor.decode(generation, skip_special_tokens=True)
        
        return comparison.strip()
    
    def _generate_image(self, caption: str) -> Image.Image:
        """
        Generate an image from a caption using the diffusion model.
        
        Args:
            caption: The text caption to use for generation
            
        Returns:
            PIL.Image: Generated image
        """
        if self.diffusion_model == "sd35":
            return self.diffusion_pipe(
                caption,
                num_inference_steps=28,
                guidance_scale=3.5,
            ).images[0]
        else:  # flux
            return self.diffusion_pipe(
                caption,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=5,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]
    
    def _collect_image_paths(self) -> List[str]:
        """
        Collect image paths either from self.queries or by scanning directories.
        
        Returns:
            List[str]: List of image paths
        """
        image_paths = []
        
        # Try to use queries if available
        if self.queries:
            logger.info("Collecting image paths from queries JSON file")
            for query in self.queries:
                mscoco_name = query['img']
                image_path = os.path.join(self.input_dir, mscoco_name)
                
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                else:
                    logger.warning(f"Image path not found: {image_path}")
        
        # If no images from queries or empty queries dict, scan directories
        if not image_paths:
            logger.info("No images found in queries or no queries file. Scanning directories.")
            for subdir in self.image_subdirs:
                directory = self.input_dir / subdir
                if not directory.exists():
                    logger.warning(f"Directory not found: {directory}")
                    continue
                    
                logger.info(f"Scanning directory: {directory}")
                
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    paths = list(directory.glob(ext))
                    logger.info(f"Found {len(paths)} {ext} files in {directory}")
                    image_paths.extend(paths)
        
        # Limit the number of images if specified
        if self.max_images is not None and len(image_paths) > self.max_images:
            logger.info(f"Limiting to {self.max_images} images (from {len(image_paths)} found)")
            image_paths = image_paths[:self.max_images]
        
        return image_paths
    
    def process(self):
        """
        Process all images through the pipeline with caption refinement.
        """
        # Initialize models
        self._setup_caption_model()
        self._setup_comparison_model()
        self._setup_diffusion_model()
        
        # Collect image paths
        image_paths = self._collect_image_paths()
        logger.info(f"Processing {len(image_paths)} images with {self.refinement_rounds} refinement rounds")
        
        # Process each image
        for idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            try:
                # Get relative path for organization
                rel_path = os.path.relpath(str(image_path), str(self.input_dir))
                image_id = f"{Path(rel_path).stem}"
                
                # Create a directory to store all rounds for this image
                image_output_dir = self.output_dir / f"image_{image_id}"
                os.makedirs(image_output_dir, exist_ok=True)
                
                # Create subdirectories within the image directory
                image_captions_dir = image_output_dir / "captions"
                image_generated_dir = image_output_dir / "generated_images"
                image_comparisons_dir = image_output_dir / "comparisons"
                os.makedirs(image_captions_dir, exist_ok=True)
                os.makedirs(image_generated_dir, exist_ok=True)
                os.makedirs(image_comparisons_dir, exist_ok=True)
                
                # Copy original image to both the main original images directory and image-specific directory
                original_out_path = self.original_images_dir / f"{image_id}_original.jpg"
                image_original_path = image_output_dir / f"original.jpg"
                shutil.copy2(image_path, original_out_path)
                shutil.copy2(image_path, image_original_path)
                
                # Initialize dialogue tracking
                dialogue = {
                    "image_id": image_id,
                    "original_image_path": str(original_out_path),
                    "rounds": []
                }
                
                # Initial caption (round 0)
                initial_caption = self._generate_caption(str(image_path), refinement_round=0)
                
                
                
                # Save initial caption - both to main directories and image-specific directory
                caption_0_path = self.captions_dir / f"{image_id}_caption_0.txt"
                image_caption_0_path = image_output_dir / "captions" / f"caption_0.txt"
                with open(caption_0_path, 'w') as f:
                    f.write(initial_caption)
                with open(image_caption_0_path, 'w') as f:
                    f.write(initial_caption)
                
                
                # Generate initial image
                generated_image_0 = self._generate_image(initial_caption)
                # Save generated image - both to main directories and image-specific directory
                generated_image_0_path = self.generated_images_dir / f"{image_id}_0.jpg"
                image_generated_0_path = image_output_dir / "generated_images" / f"generated_0.jpg"
                generated_image_0.save(generated_image_0_path)
                generated_image_0.save(image_generated_0_path)
                
                # Add initial round to dialogue
                dialogue["rounds"].append({
                    "round": 0,
                    "caption": initial_caption,
                    "caption_path": str(caption_0_path),
                    "generated_image_path": str(generated_image_0_path),
                    "comparison": None,
                    "comparison_path": None
                })
                
                # Previous caption for refinement
                previous_caption = initial_caption
                
                # Perform refinement rounds
                for round_idx in range(1, self.refinement_rounds + 1):
                    # Compare original and previously generated image
                    previous_image_path = self.generated_images_dir / f"{image_id}_{round_idx-1}.jpg"
                    comparison = self._compare_images(previous_caption, str(image_path), str(previous_image_path))
                    
                    # Save comparison text - both to main directory and image-specific directory
                    comparison_path = self.comparisons_dir / f"{image_id}_comparison_{round_idx-1}.txt"
                    image_comparison_path = image_output_dir / "comparisons" / f"comparison_{round_idx-1}.txt"
                    with open(comparison_path, 'w') as f:
                        f.write(comparison)
                    with open(image_comparison_path, 'w') as f:
                        f.write(comparison)
                    
                    # Refine caption based on comparison feedback
                    refined_caption = self._generate_caption(
                        str(image_path), 
                        refinement_round=round_idx, 
                        previous_caption=previous_caption,
                        comparison_feedback=comparison
                    )
                    
                    # Generate new image with refined caption
                    generated_image = self._generate_image(refined_caption)
                    
                    # Save refined caption and image - both to main directories and image-specific directory
                    caption_path = self.captions_dir / f"{image_id}_caption_{round_idx}.txt"
                    image_caption_path = image_output_dir / "captions" / f"caption_{round_idx}.txt"
                    with open(caption_path, 'w') as f:
                        f.write(refined_caption)
                    with open(image_caption_path, 'w') as f:
                        f.write(refined_caption)
                    
                    generated_image_path = self.generated_images_dir / f"{image_id}_{round_idx}.jpg"
                    image_generated_path = image_output_dir / "generated_images" / f"generated_{round_idx}.jpg"
                    generated_image.save(generated_image_path)
                    generated_image.save(image_generated_path)
                    
                    # Add round to dialogue
                    dialogue["rounds"].append({
                        "round": round_idx,
                        "caption": refined_caption,
                        "caption_path": str(caption_path),
                        "generated_image_path": str(generated_image_path),
                        "comparison": comparison,
                        "comparison_path": str(comparison_path)
                    })
                    
                    # Update previous caption for next round
                    previous_caption = refined_caption
                
                # Save the dialogue JSON - both to main directory and image-specific directory
                dialogue_path = self.dialogue_dir / f"{image_id}_dialogue.json"
                image_dialogue_path = image_output_dir / f"dialogue.json"
                with open(dialogue_path, 'w') as f:
                    json.dump(dialogue, f, indent=2)
                with open(image_dialogue_path, 'w') as f:
                    json.dump(dialogue, f, indent=2)
                
                logger.info(f"Processed image {idx+1}/{len(image_paths)}: {rel_path} with {self.refinement_rounds} refinement rounds")
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    def run(self):
        """Run the complete pipeline with refinement."""
        logger.info(f"Starting Caption Refinement Pipeline")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Captioning model: {self.caption_model_id} on GPU {self.caption_gpu_id}")
        logger.info(f"Comparison model: {self.comparison_model_id} on GPU {self.caption_gpu_id}")
        logger.info(f"Diffusion model: {self.diffusion_model} on GPU {self.diffusion_gpu_id}")
        logger.info(f"Refinement rounds: {self.refinement_rounds}")
        
        start_time = time.time()
        self.process()
        end_time = time.time()
        
        logger.info(f"Pipeline complete! Processed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # Create and run the pipeline
    pipeline = CaptionRefinementPipeline(
        input_dir="/data/mscoco",
        queries_path="ChatIR/dialogues/VisDial_v1.0_queries_val.json",
        output_dir="caption_refinement_output_12b_200t_concise",
        image_subdirs=["val2017"],  # Process only validation images
        max_images=None,  # Limit to 10 images for testing
        caption_model_id="google/gemma-3-12b-it",  # Use 12B model
        comparison_model_id="google/gemma-3-12b-it",  # Same model for comparison
        # diffusion_model="sd35",  # Use Stable Diffusion 3.5
        diffusion_model="flux",  # Use FLUX model
        caption_gpu_id=0,  # Use GPU 0 for captioning and comparison
        diffusion_gpu_id=1,  # Use GPU 1 for image generation
        refinement_rounds=0  # Perform 2 rounds of refinement
    )
    pipeline.run()