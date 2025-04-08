import torch
import json
import os
import shutil
from pathlib import Path
from diffusers import StableDiffusion3Pipeline, FluxPipeline
from typing import Dict, List, Callable, Union, Optional
import torch.multiprocessing as mp
from tqdm import tqdm


class GenIRImageGenerator:
    """
    Image generation class for GenIR which generates images based on dialogue queries.
    Utilizes multiple GPUs for parallel processing and also copies original images for reference.
    """
    
    def __init__(
        self, 
        save_path: str = 'genIR_images',
        model_type: str = 'sd35',
        max_length: Optional[int] = 11,  # Make max_length optional
        max_dialog_length: int = 11,
        queries_path: str = "ChatIR/dialogues/VisDial_v1.0_queries_val.json",
        gpu_ids: List[int] = [0, 1],
        mscoco_dir: str = "/data/mscoco"
    ):
        """
        Initialize the image generator.
        
        Args:
            save_path: Directory to save generated images
            model_type: Type of model to use ('sd35' or 'flux')
            max_length: Maximum number of dialogues to process
            queries_path: Path to the JSON file containing dialogue queries
            gpu_ids: List of GPU IDs to use for processing
            mscoco_dir: Directory containing original MSCOCO images
        """
        self.save_path = Path(save_path)
        self.save_path = os.path.join(self.save_path, model_type)
        os.makedirs(self.save_path, exist_ok=True)
        
        self.max_length = max_length
        self.max_dialog_length = max_dialog_length
        self.model_type = model_type
        self.gpu_ids = gpu_ids
        self.device_count = len(gpu_ids)
        self.mscoco_dir = Path(mscoco_dir)
        
        # Ensure MSCOCO directory exists
        if not self.mscoco_dir.exists():
            raise FileNotFoundError(f"MSCOCO directory not found: {self.mscoco_dir}")
        
        # Load queries
        self.queries = self._load_json(queries_path)
        
        # Create base save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        
    def _load_json(self, path: str) -> Dict:
        """Load and parse a JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find queries file at {path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {path}")
    
    def _init_model(self, model_type: str, device_id: int) -> Union[StableDiffusion3Pipeline, FluxPipeline]:
        """Initialize the appropriate model pipeline based on model type and assign to specific GPU."""
        device = f"cuda:{device_id}"
        
        if model_type == "sd35":
            # Set torch compute dtype
            torch.set_default_dtype(torch.float16)
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                torch_dtype=torch.float16,
                variant="fp16"
            )
            pipe = pipe.to(device)
            return pipe
        elif model_type == "flux":
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", 
                torch_dtype=torch.bfloat16
            )
            # Either use one of these approaches:
            
            # OPTION 1: Specify the device explicitly in enable_model_cpu_offload
            pipe.enable_model_cpu_offload(gpu_id=device_id)
            
            # OPTION 2: Or, move to device first, then enable offload
            # pipe.to(device)
            # pipe.enable_model_cpu_offload()
            
            return pipe
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Choose either 'sd35' or 'flux'")
    
    def _sd35_inference(self, pipe, prompt: str):
        """Run inference with Stable Diffusion 3.5."""
        return pipe(
            prompt,
            num_inference_steps=28,
            guidance_scale=3.5,
        ).images[0]

    def _flux_inference(self, pipe, prompt: str):
        """Run inference with FLUX model."""
        return pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

    def _get_inference_function(self, model_type: str) -> Callable:
        """Get the appropriate inference function for the selected model type."""
        if model_type == "sd35":
            return self._sd35_inference
        elif model_type == "flux":
            return self._flux_inference
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _copy_original_image(self, mscoco_name: str) -> bool:
        """Copy the original MSCOCO image to the output directory."""
        try:
            # Source path in the MSCOCO directory
            src_path = self.mscoco_dir / mscoco_name
            
            # Destination path in our output directory structure
            dest_dir = os.path.join(self.save_path, os.path.dirname(mscoco_name))
            dest_path = os.path.join(self.save_path, mscoco_name)
            
            # Create destination directory if it doesn't exist
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copy the file if it exists
            if src_path.exists():
                shutil.copy2(src_path, dest_path)
                print(f"Copied original image: {mscoco_name}")
                return True
            else:
                print(f"Original image not found: {src_path}")
                return False
        except Exception as e:
            print(f"Error copying original image {mscoco_name}: {e}")
            return False
    
    def _process_batch(self, gpu_id: int, query_batch: List[Dict]):
        """Process a batch of queries on a specific GPU."""
        device = f"cuda:{gpu_id}"
        print(f"Initializing model on GPU {gpu_id}")
        
        # Initialize model on this GPU
        pipe = self._init_model(self.model_type, gpu_id)
        inference_fn = self._get_inference_function(self.model_type)
        
        # Process each query in the batch
        for query in tqdm(query_batch, desc=f"GPU {gpu_id} processing"):
            mscoco_name = query['img']
            dialogue = query['dialog']
            
            # Copy the original image first
            self._copy_original_image(mscoco_name)
            
            # Generate an image for each dialogue turn
            dialouge_turns = min(len(dialogue), self.max_dialog_length)
            for turn_idx in range(1, dialouge_turns + 1):
                # Build prompt from dialogue history up to current turn
                prompt = " ".join(dialogue[:turn_idx])
                
                # Generate image
                try:
                    
                    # Create save path and ensure directories exist
                    image_filename = mscoco_name.replace(".jpg", f"_{turn_idx-1}.jpg")
                    save_path = self.save_path / Path(image_filename)
                    os.makedirs(save_path.parent, exist_ok=True)
                    
                    if save_path.exists():
                        print(f"Image already exists: {save_path}")
                        continue
                    else:
                        # Run inference                    
                        image = inference_fn(pipe, prompt)
                        # Save the image
                        print(f"GPU {gpu_id}: Saving image to {save_path}")
                        image.save(save_path)
                    
                except Exception as e:
                    print(f"GPU {gpu_id}: Error generating image for {mscoco_name}, turn {turn_idx}: {e}")
    
    def run(self):
        """Process queries in parallel using multiple GPUs."""
        # Use all queries if max_length is None, otherwise limit to max_length
        limited_queries = self.queries if self.max_length is None else self.queries[:self.max_length]
        
        # Distribute queries among GPUs
        batches = [[] for _ in range(self.device_count)]
        for i, query in enumerate(limited_queries):
            batches[i % self.device_count].append(query)
        
        print(f"Distributing {len(limited_queries)} queries across {self.device_count} GPUs")
        for i, batch in enumerate(batches):
            print(f"GPU {self.gpu_ids[i]}: {len(batch)} queries assigned")
        
        # For single GPU processing
        if self.device_count == 1:
            self._process_batch(self.gpu_ids[0], batches[0])
            return
            
        # For multi-GPU processing, use torch multiprocessing
        mp.set_start_method('spawn', force=True)
        processes = []
        
        try:
            # Create processes for each GPU
            for i, gpu_id in enumerate(self.gpu_ids):
                if len(batches[i]) > 0:  # Only create process if there are queries to process
                    p = mp.Process(
                        target=self._process_batch,
                        args=(gpu_id, batches[i])
                    )
                    processes.append(p)
                    p.start()
                    
            # Wait for all processes to complete
            for p in processes:
                p.join()
                
        except Exception as e:
            print(f"Error in multiprocessing: {e}")
            # Terminate any running processes
            for p in processes:
                if p.is_alive():
                    p.terminate()


if __name__ == "__main__":
    # Create and run the image generator with both GPUs
    generator = GenIRImageGenerator(
        save_path='genIR_images',
        # model_type="sd35",
        model_type="flux",
        max_length=None,  # Process all queries (no limit)
        max_dialog_length=1,
        gpu_ids=[0,1],  # Use both GPUs
        mscoco_dir="/data/mscoco"  # Path to original MSCOCO images
    )
    generator.run()