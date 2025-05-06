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


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CaptionToImagePipeline:
    """
    Pipeline that:
    1. Reads original images from a directory
    2. Generates captions using Gemma3 on GPU 0
    3. Uses captions to generate new images with diffusion models on GPU 1
    4. Saves both captions and generated images
    """
    
    def __init__(
        self, 
        input_dir: str = "/data/mscoco",
        queries_path: str = "ChatIR/dialogues/VisDial_v1.0_queries_val.json",
        output_dir: str = "caption_to_image_output",
        image_subdirs: List[str] = ["train2017", "val2017", "unlabeled2017"],
        max_images: Optional[int] = None,
        caption_model_id: str = "google/gemma-3-12b-it",
        diffusion_model: str = "sd35",  # "sd35" or "flux"
        caption_gpu_id: int = 0,
        diffusion_gpu_id: int = 1
    ):
        """
        Initialize the pipeline.
        
        Args:
            input_dir: Directory containing original images
            output_dir: Directory to save generated content
            image_subdirs: Subdirectories in input_dir containing images to process
            max_images: Maximum number of images to process
            caption_model_id: Gemma model ID for captioning
            diffusion_model: Type of diffusion model to use ('sd35' or 'flux')
            caption_gpu_id: GPU ID for captioning model
            diffusion_gpu_id: GPU ID for diffusion model
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_subdirs = image_subdirs
        self.max_images = max_images
        self.caption_model_id = caption_model_id
        self.diffusion_model = diffusion_model
        self.caption_gpu_id = caption_gpu_id
        self.diffusion_gpu_id = diffusion_gpu_id
        
        # Load queries
        self.queries = self._load_json(queries_path)
        
        # Create output directories
        self.original_images_dir = self.output_dir / "original_images"
        self.captions_dir = self.output_dir / "captions"
        self.generated_images_dir = self.output_dir / "generated_images"
        
        os.makedirs(self.original_images_dir, exist_ok=True)
        os.makedirs(self.captions_dir, exist_ok=True)
        os.makedirs(self.generated_images_dir, exist_ok=True)
        
        # Initialize models
        self.caption_model = None
        self.caption_processor = None
        self.diffusion_pipe = None
        
    def _load_json(self, path: str) -> Dict:
        """Load and parse a JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find queries file at {path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {path}")
    
    def _setup_caption_model(self):
        """Set up the Gemma3 model for image captioning on GPU 0."""
        logger.info(f"Setting up captioning model on GPU {self.caption_gpu_id}")
        
        # Load model with explicit device mapping to GPU 0
        self.caption_model = Gemma3ForConditionalGeneration.from_pretrained(
            self.caption_model_id, 
            device_map=f"cuda:{self.caption_gpu_id}",
            torch_dtype=torch.bfloat16
        ).eval()
        
        # Load processor
        self.caption_processor = AutoProcessor.from_pretrained(self.caption_model_id)
        
        logger.info(f"Captioning model loaded on GPU {self.caption_gpu_id}")
    
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
    
    def _generate_caption(self, image_path: str, max_tokens: int = 77) -> str:
        """
        Generate a detailed caption for an image using Gemma3.
        
        Args:
            image_path: Path to the image
            max_tokens: Maximum number of tokens for caption
            
        Returns:
            str: Generated caption
        """
        prompt = (
            "Generate a detailed, descriptive caption of this image for use as a prompt in a diffusion model. "
            "Focus on visual elements, style, composition, colors, and details. Generate ONLY a detailed caption "
            "for this image. DO NOT include any introductory text or explanations. The output should be the "
            "raw caption text ready for a diffusion model."
        )
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": f"You are an image captioning assistant that creates concise but detailed descriptions for use in text-to-image diffusion models. Keep your description under {max_tokens} tokens."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
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
        
        # Move inputs to GPU 0
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
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]
    
    # def _collect_image_paths(self) -> List[str]:
    #     """
    #     Collect all image paths from the input directories.
        
    #     Returns:
    #         List[str]: List of image paths
    #     """
    #     image_paths = []
        
    #     for subdir in self.image_subdirs:
    #         directory = self.input_dir / subdir
    #         if not directory.exists():
    #             logger.warning(f"Directory not found: {directory}")
    #             continue
                
    #         logger.info(f"Scanning directory: {directory}")
            
    #         for ext in ["*.jpg", "*.jpeg", "*.png"]:
    #             paths = list(directory.glob(ext))
    #             logger.info(f"Found {len(paths)} {ext} files in {directory}")
    #             image_paths.extend(paths)
        
    #     # Limit the number of images if specified
    #     if self.max_images is not None and len(image_paths) > self.max_images:
    #         logger.info(f"Limiting to {self.max_images} images (from {len(image_paths)} found)")
    #         image_paths = image_paths[:self.max_images]
        
    #     return image_paths
    
    def _collect_image_paths(self) -> List[str]:
        """
        Collect all image paths self.queries.
        
        Returns:
            List[str]: List of image paths
        """
        image_paths = []
        dialogues   = []
        
        for query in self.queries:
            mscoco_name = query['img']
            dialogue = query['dialog']
            image_path = os.path.join(self.input_dir, mscoco_name)
            
            if os.path.exists(image_path):
                image_paths.append(image_path)
                dialogues.append(dialogue)
            else:
                logger.warning(f"Image path not found: {image_path}")
        
        # Limit the number of images if specified
        if self.max_images is not None and len(image_paths) > self.max_images:
            logger.info(f"Limiting to {self.max_images} images (from {len(image_paths)} found)")
            image_paths = image_paths[:self.max_images]
        
        return image_paths
        
    
    def process(self):
        """
        Process all images through the pipeline:
        1. Copy original images
        2. Generate captions
        3. Generate new images from captions
        """
        # Initialize models
        self._setup_caption_model()
        self._setup_diffusion_model()
        
        # Collect image paths
        image_paths = self._collect_image_paths()
        logger.info(f"Processing {len(image_paths)} images")
        
        # Process each image
        for idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            try:
                # Get relative path for organization
                rel_path = os.path.relpath(image_path, self.input_dir)
                image_id = f"{Path(rel_path).stem}"
                
                # 1. Copy original image to output directory
                original_out_path = self.original_images_dir / f"{image_id}_original.jpg"
                shutil.copy2(image_path, original_out_path)
                
                # 2. Generate caption with Gemma3 on GPU 0
                caption = self._generate_caption(str(image_path))
                
                # Save caption to file
                caption_path = self.captions_dir / f"{image_id}_caption.txt"
                with open(caption_path, 'w') as f:
                    f.write(caption)
                
                # 3. Generate new image from caption on GPU 1
                generated_image = self._generate_image(caption)
                
                # Save generated image
                # Hard coded since this there is no conversation rounds here yet
                generated_image_path = self.generated_images_dir / f"{image_id}_0.jpg" 
                generated_image.save(generated_image_path)
                
                logger.info(f"Processed image {idx+1}/{len(image_paths)}: {rel_path}")
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
    
    def run(self):
        """Run the complete pipeline."""
        logger.info(f"Starting Caption-to-Image Pipeline")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Captioning model: {self.caption_model_id} on GPU {self.caption_gpu_id}")
        logger.info(f"Diffusion model: {self.diffusion_model} on GPU {self.diffusion_gpu_id}")
        
        self.process()
        
        logger.info("Pipeline complete!")


if __name__ == "__main__":
    # Create and run the pipeline
    pipeline = CaptionToImagePipeline(
        input_dir="/data/mscoco",
        output_dir="caption_to_image_output",
        image_subdirs=["val2017"],  # Process only validation images
        max_images=None,  # Limit to 50 images for testing
        caption_model_id="google/gemma-3-12b-it",  # Use 12B model for efficiency
        diffusion_model="sd35",  # Use Stable Diffusion 3.5
        caption_gpu_id=0,  # Use GPU 0 for captioning
        diffusion_gpu_id=1  # Use GPU 1 for image generation
    )
    pipeline.run()