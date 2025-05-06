import torch
import json
import os
import shutil
import sys
from pathlib import Path
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from tqdm import tqdm
from PIL import Image
import logging
from typing import Dict, List, Callable, Union, Optional
import time
import os.path as osp
import random
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SelfDialogueCaptionPipeline:
    """
    Enhanced pipeline that:
    1. Reads original images from a directory
    2. Generates initial captions using Gemma3
    3. Uses different roles of the same LLM to critique and refine the caption
    4. Iteratively improves the caption through multiple rounds of dialogue
    5. Saves all intermediate results including captions and dialogue history
    """
    
    def __init__(
        self, 
        input_dir: str = "/data/mscoco",
        queries_path: str = "ChatIR/dialogues/VisDial_v1.0_queries_val.json",
        output_dir: str = "caption_refinement_output",
        image_subdirs: List[str] = ["train2017", "val2017", "unlabeled2017"],
        max_images: Optional[int] = None,
        model_id: str = "google/gemma-3-12b-it",
        gpu_id: int = 0,
        refinement_rounds: int = 5,  # Number of refinement iterations
        roles: List[str] = ["Critic", "Refiner", "Evaluator"]  # Different roles for the dialogue
    ):
        """
        Initialize the pipeline.
        
        Args:
            input_dir: Directory containing original images
            queries_path: Path to JSON file with queries/dialogues
            output_dir: Directory to save generated content
            image_subdirs: Subdirectories in input_dir containing images to process
            max_images: Maximum number of images to process
            model_id: Gemma model ID for captioning and dialogue
            gpu_id: GPU ID for the model
            refinement_rounds: Number of caption refinement iterations
            roles: Different roles the LLM will take in the dialogue
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_subdirs = image_subdirs
        self.max_images = max_images
        self.model_id = model_id
        self.gpu_id = gpu_id
        self.refinement_rounds = refinement_rounds
        self.roles = roles
        
        # Load queries
        self.queries = self._load_json(queries_path)
        
        # Create output directories
        self.original_images_dir = self.output_dir / "original_images"
        self.captions_dir = self.output_dir / "captions"
        self.dialogue_dir = self.output_dir / "dialogues"
        
        os.makedirs(self.original_images_dir, exist_ok=True)
        os.makedirs(self.captions_dir, exist_ok=True)
        os.makedirs(self.dialogue_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.processor = None
        
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
    
    def _setup_model(self):
        """Set up the Gemma3 model on the specified GPU."""
        logger.info(f"Setting up model on GPU {self.gpu_id}")
        
        # Load model with explicit device mapping
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id, 
            device_map=f"cuda:{self.gpu_id}",
            torch_dtype=torch.bfloat16
        ).eval()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        logger.info(f"Model loaded on GPU {self.gpu_id}")
    
    def _generate_initial_caption(self, image_path: str, max_tokens: int = 200) -> str:
        """
        Generate an initial caption for an image using Gemma3.
        
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
        
        # System message for initial captioning
        system_message = f"You are an image captioning assistant that creates concise but detailed descriptions for use in text-to-image diffusion models. Keep your description under {max_tokens} tokens."
        
        # Just the image in the user message for initial captioning
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
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        )
        
        # Move inputs to GPU
        device = f"cuda:{self.gpu_id}"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Track input length for extraction
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate text
        with torch.inference_mode():
            generation = self.model.generate(
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
        caption = self.processor.decode(generation, skip_special_tokens=True)
        
        return caption.strip()
    
    def _critique_caption(self, image_path: str, caption: str, role: str, max_tokens: int = 200) -> str:
        """
        Generate criticism of a caption from a specific role perspective.
        
        Args:
            image_path: Path to the image
            caption: The caption to critique
            role: The role perspective (e.g., "Critic", "Refiner")
            max_tokens: Maximum number of tokens for the critique
            
        Returns:
            str: Generated critique
        """
        prompt = f"You are playing the role of a '{role}' in a caption refinement process. Review this caption: \"{caption}\". Provide specific feedback on how to improve it. Focus on accuracy, detail, clarity, and usefulness for a text-to-image system. Keep your critique under {max_tokens} tokens. Provide ONLY the critique with constructive feedback."
        
        # System message
        system_message = f"You are a {role.lower()} in an iterative caption refinement process. Your goal is to help improve image captions by identifying issues and suggesting improvements."
        
        # Prepare messages with the image
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
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
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        )
        
        # Move inputs to GPU
        device = f"cuda:{self.gpu_id}"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Track input length for extraction
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate text
        with torch.inference_mode():
            generation = self.model.generate(
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
        critique = self.processor.decode(generation, skip_special_tokens=True)
        
        return critique.strip()
    
    def _refine_caption(self, image_path: str, caption: str, critique: str, max_tokens: int = 200) -> str:
        """
        Refine a caption based on critique.
        
        Args:
            image_path: Path to the image
            caption: The original caption
            critique: Critique of the caption
            max_tokens: Maximum number of tokens for the refined caption
            
        Returns:
            str: Refined caption
        """
        prompt = f"Based on this critique: \"{critique}\", refine the following caption: \"{caption}\". Create an improved version that addresses the issues while staying true to the image content. Provide ONLY the refined caption with no explanations or surrounding text. Keep your response under {max_tokens} tokens."
        
        # System message
        system_message = "You are a caption refinement assistant. Your task is to improve image captions based on critique feedback."
        
        # Prepare messages with the image
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
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
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        )
        
        # Move inputs to GPU
        device = f"cuda:{self.gpu_id}"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Track input length for extraction
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate text
        with torch.inference_mode():
            generation = self.model.generate(
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
        refined_caption = self.processor.decode(generation, skip_special_tokens=True)
        
        return refined_caption.strip()
    
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
            # for subdir in self.image_subdirs:
            #     directory = self.input_dir / subdir
            #     if not directory.exists():
            #         logger.warning(f"Directory not found: {directory}")
            #         continue
                    
            #     logger.info(f"Scanning directory: {directory}")
                
            #     for ext in ["*.jpg", "*.jpeg", "*.png"]:
            #         paths = list(directory.glob(ext))
            #         logger.info(f"Found {len(paths)} {ext} files in {directory}")
            #         image_paths.extend(paths)
            for subdir in sorted(os.listdir(self.input_dir)):
                if not os.path.isdir(os.path.join(self.input_dir, subdir)) or subdir == "LICENSE.txt":
                    continue
                
                directory = self.input_dir / subdir
                logger.info(f"Scanning FFHQ directory: {directory}")
                
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
        Process all images through the pipeline with self-dialogue caption refinement.
        """
        # Initialize model
        self._setup_model()
        
        # Collect image paths
        image_paths = self._collect_image_paths()
        logger.info(f"Processing {len(image_paths)} images with {self.refinement_rounds} refinement rounds")
        
        # Process each image
        for idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            # Get relative path for organization
            rel_path = os.path.relpath(str(image_path), str(self.input_dir))
            image_id = f"{Path(rel_path).stem}"
            
            # Create a directory to store all rounds for this image
            image_output_dir = self.output_dir / f"image_{image_id}"
            os.makedirs(image_output_dir, exist_ok=True)
            
            # Create subdirectories within the image directory
            image_captions_dir = image_output_dir / "captions"
            image_dialogues_dir = image_output_dir / "dialogues"
            os.makedirs(image_captions_dir, exist_ok=True)
            os.makedirs(image_dialogues_dir, exist_ok=True)
            
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
            
            # Generate initial caption (round 0)
            initial_caption = self._generate_initial_caption(str(image_path))
            
            # Save initial caption
            caption_0_path = self.captions_dir / f"{image_id}_caption_0.txt"
            image_caption_0_path = image_captions_dir / f"caption_0.txt"
            with open(caption_0_path, 'w') as f:
                f.write(initial_caption)
            with open(image_caption_0_path, 'w') as f:
                f.write(initial_caption)
            
            # Add initial round to dialogue
            dialogue["rounds"].append({
                "round": 0,
                "role": "Captioner",
                "caption": initial_caption,
                "caption_path": str(caption_0_path),
                "message_type": "caption"
            })
            
            # Track the current caption
            current_caption = initial_caption
            
            # Perform refinement rounds through dialogue
            for round_idx in range(1, self.refinement_rounds + 1):
                
                # Generate critique 
                critique = self._critique_caption(
                    str(image_path), 
                    current_caption, 
                    'Detail Critic'
                )
                
                # Save critique
                critique_path = self.dialogue_dir / f"{image_id}_critique_{round_idx}.txt"
                image_critique_path = image_dialogues_dir / f"critique_{round_idx}.txt"
                with open(critique_path, 'w') as f:
                    f.write(critique)
                with open(image_critique_path, 'w') as f:
                    f.write(critique)
                    
                # Add critique to dialogue
                dialogue["rounds"].append({
                    "round": round_idx,
                    "role": "Detail Critic",
                    "message": critique,
                    "message_path": str(critique_path),
                    "message_type": "critique"
                })
                # Generate refined caption based on critique
                refined_caption = self._refine_caption(
                    str(image_path),
                    current_caption,
                    critique
                )
                
                # Save refined caption
                caption_path = self.captions_dir / f"{image_id}_caption_{round_idx}.txt"
                image_caption_path = image_captions_dir / f"caption_{round_idx}.txt"
                with open(caption_path, 'w') as f:
                    f.write(refined_caption)
                with open(image_caption_path, 'w') as f:
                    f.write(refined_caption)
                    
                # Add refined caption to dialogue
                dialogue["rounds"].append({
                    "round": round_idx,
                    "role": "Refiner",
                    "caption": refined_caption,
                    "caption_path": str(caption_path),
                    "message_type": "refined_caption"
                })
                
                # Update current caption for next round
                current_caption = refined_caption
                
                
                
            # Save the dialogue JSON
            dialogue_path = self.dialogue_dir / f"{image_id}_dialogue.json"
            image_dialogue_path = image_output_dir / f"dialogue.json"
            with open(dialogue_path, 'w') as f:
                json.dump(dialogue, f, indent=2)
            with open(image_dialogue_path, 'w') as f:
                json.dump(dialogue, f, indent=2)
            
            logger.info(f"Processed image {idx+1}/{len(image_paths)}: {rel_path} with {self.refinement_rounds} refinement rounds")
                
    
    def run(self):
        """Run the complete self-dialogue caption refinement pipeline."""
        logger.info(f"Starting Self-Dialogue Caption Refinement Pipeline")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Model: {self.model_id} on GPU {self.gpu_id}")
        logger.info(f"Refinement rounds: {self.refinement_rounds}")
        logger.info(f"Dialogue roles: {self.roles}")
        
        start_time = time.time()
        self.process()
        end_time = time.time()
        
        logger.info(f"Pipeline complete! Processed in {end_time - start_time:.2f} seconds")


def parse_args():
    parser = argparse.ArgumentParser(description="Self Dialogue Caption Pipeline")
    
    # Required arguments
    parser.add_argument("--input_dir", type=str, default="/data/mscoco", 
                        help="Input directory containing the image dataset")
    parser.add_argument("--queries_path", type=str, 
                        default="ChatIR/dialogues/VisDial_v1.0_queries_val.json",
                        help="Path to queries JSON file")
    parser.add_argument("--output_dir", type=str, 
                        default="results_genir/self_dialogue_caption_output",
                        help="Output directory for results")
    
    # Optional arguments
    parser.add_argument("--image_subdirs", nargs="+", default=["val2017"],
                        help="Image subdirectories to process")
    parser.add_argument("--max_images", type=int, default=3000,
                        help="Maximum number of images to process")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-12b-it",
                        help="Model ID to use for the pipeline")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use for processing")
    parser.add_argument("--refinement_rounds", type=int, default=10,
                        help="Number of refinement rounds")
    parser.add_argument("--roles", nargs="+", default=["Detail Critic", "Refiner"],
                        help="Roles for the dialogue")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # args.input_dir = '/data/skin_gen_data/ffhq/images'
    # args.output_dir = './results_genir/self_dialogue_caption_output_dev'
    # args.gpu_id    = 1
    # args.queries_path = ''
    # args.max_images = 1000
    
    # Create and run the pipeline
    pipeline = SelfDialogueCaptionPipeline(
        input_dir=args.input_dir,
        queries_path=args.queries_path,
        output_dir=args.output_dir,
        image_subdirs=args.image_subdirs,
        max_images=args.max_images,
        model_id=args.model_id,
        gpu_id=args.gpu_id,
        refinement_rounds=args.refinement_rounds,
        roles=args.roles
    )
    pipeline.run()