import torch
import json
import os
import shutil
from pathlib import Path
import torch.nn.functional as F
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from tqdm import tqdm
from PIL import Image
import logging
from typing import Dict, List, Callable, Union, Optional, Tuple
import time
import argparse
from ChatIR.baselines import ImageEmbedder, CLIP_ZERO_SHOT_BASELINE, BLIP_BASELINE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Corpus(torch.utils.data.Dataset):
    """Dataset class for the corpus images (the potential candidates)"""
    def __init__(self, corpus_path, preprocessor, dataset_root='/data/mscoco'):
        with open(corpus_path) as f:
            self.corpus = json.load(f)
        
        self.corpus = [os.path.join(dataset_root, path) for path in self.corpus]
            
        self.preprocessor = preprocessor
        self.path2id = {self.corpus[i]: i for i in range(len(self.corpus))}

    def __len__(self):
        return len(self.corpus)

    def path_to_index(self, path):
        """For finding a target image fast"""
        return self.path2id[path]

    def __getitem__(self, i):
        image = self.preprocessor(self.corpus[i])  # Load and prepare image
        return {'id': i, 'image': image}


class VisualPredictionFeedbackPipeline:
    """
    Pipeline that:
    1. Reads original images from a directory
    2. Generates initial captions using a vision-language model
    3. Uses captions to retrieve images from a corpus
    4. Compares retrieved images with original to refine captions
    5. Iteratively improves retrieval through visual prediction feedback
    """
    
    def __init__(
        self, 
        input_dir: str = "/data/mscoco",
        queries_path: str = "ChatIR/dialogues/VisDial_v1.0_queries_val.json",
        corpus_path: str = "ChatIR/ChatIR_Protocol/Search_Space_val_50k.json",
        output_dir: str = "visual_prediction_feedback_output",
        image_subdirs: List[str] = ["train2017", "val2017", "unlabeled2017"],
        max_images: Optional[int] = None,
        caption_model_id: str = "google/gemma-3-12b-it",
        retrieval_baseline: str = "blip-zero-shot",  # or "clip-zero-shot"
        caption_gpu_id: int = 0,
        retrieval_gpu_id: int = 1,
        refinement_rounds: int = 10,
        cache_corpus: str = "",  # Path to cache the indexed corpus
        sep_token: str = ", ",
        corpus_bs: int = 500,
        queries_bs: int = 500,
        num_workers: int = 32,
        retrieval_top_k: int = 5  # Number of top retrieved images to consider
    ):
        """
        Initialize the pipeline.
        
        Args:
            input_dir: Directory containing original images
            queries_path: Path to JSON file with queries/dialogues
            corpus_path: Path to JSON file with the corpus images
            output_dir: Directory to save generated content
            image_subdirs: Subdirectories in input_dir containing images to process
            max_images: Maximum number of images to process
            caption_model_id: Gemma model ID for captioning and comparison
            retrieval_baseline: Baseline model to use for retrieval ('blip-zero-shot' or 'clip-zero-shot')
            caption_gpu_id: GPU ID for captioning model
            retrieval_gpu_id: GPU ID for retrieval model
            refinement_rounds: Number of refinement iterations
            cache_corpus: Path to cache the indexed corpus
            sep_token: Separation token for dialogue rounds
            corpus_bs: Batch size for corpus processing
            queries_bs: Batch size for queries processing
            num_workers: Number of workers for data loading
            retrieval_top_k: Number of top retrieved images to consider for feedback
        """
        self.input_dir = Path(input_dir)
        self.queries_path = queries_path
        self.corpus_path = corpus_path
        self.output_dir = Path(output_dir)
        self.image_subdirs = image_subdirs
        self.max_images = max_images
        self.caption_model_id = caption_model_id
        self.retrieval_baseline = retrieval_baseline
        self.caption_gpu_id = caption_gpu_id
        self.retrieval_gpu_id = retrieval_gpu_id
        self.refinement_rounds = refinement_rounds
        self.cache_corpus = cache_corpus
        self.sep_token = sep_token
        self.corpus_bs = corpus_bs
        self.queries_bs = queries_bs
        self.num_workers = num_workers
        self.retrieval_top_k = retrieval_top_k
        
        # Create config for retrieval model
        self.cfg = {
            'corpus_bs': self.corpus_bs,
            'queries_bs': self.queries_bs,
            'num_workers': self.num_workers,
            'sep_token': self.sep_token,
            'cache_corpus': self.cache_corpus,
            'queries_path': self.queries_path,
            'corpus_path': self.corpus_path,
            'img_root': str(self.input_dir),
            'device': f'cuda:{self.retrieval_gpu_id}',
        }
        
        # Load queries
        self.queries = self._load_json(queries_path)
        
        # Create output directories
        self.original_images_dir = self.output_dir / "original_images"
        self.captions_dir = self.output_dir / "captions"
        self.retrieved_images_dir = self.output_dir / "retrieved_images"
        self.comparisons_dir = self.output_dir / "comparisons"
        self.dialogue_dir = self.output_dir / "dialogues"
        
        os.makedirs(self.original_images_dir, exist_ok=True)
        os.makedirs(self.captions_dir, exist_ok=True)
        os.makedirs(self.retrieved_images_dir, exist_ok=True)
        os.makedirs(self.comparisons_dir, exist_ok=True)
        os.makedirs(self.dialogue_dir, exist_ok=True)
        
        # Initialize models
        self.caption_model = None
        self.caption_processor = None
        self.dialog_encoder = None
        self.image_embedder = None
        self.corpus_dataset = None
        self.corpus_index = None
        
    def _load_json(self, path: str) -> Dict:
        """Load and parse a JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Could not find file at {path}. Will use image directory scanning instead.")
            return {}
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {path}")
    
    def _setup_caption_model(self):
        """Set up the Gemma3 model for captioning and comparison on GPU."""
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
    
    def _setup_retrieval_model(self):
        """Set up the retrieval model (BLIP or CLIP)."""
        logger.info(f"Setting up retrieval model on GPU {self.retrieval_gpu_id}")
        
        # Set CUDA device for the retrieval model
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.retrieval_gpu_id)
        device = f"cuda:{self.retrieval_gpu_id}"
        
        if self.retrieval_baseline == 'clip-zero-shot':
            # Default cache path for CLIP
            if not self.cache_corpus:
                self.cfg['cache_corpus'] = "temp/corpus_clip_16.pth"
            self.dialog_encoder, self.image_embedder = CLIP_ZERO_SHOT_BASELINE(device)
        else:  # Default to BLIP
            # Default cache path for BLIP
            if not self.cache_corpus:
                self.cfg['cache_corpus'] = "temp/corpus_blip_small.pth"
            self.dialog_encoder, self.image_embedder = BLIP_BASELINE(device)
        
        # Initialize corpus dataset
        self.corpus_dataset = Corpus(
            self.corpus_path, 
            self.image_embedder.processor,
            dataset_root=self.cfg['img_root']
        )
        
        logger.info(f"Retrieval model ({self.retrieval_baseline}) loaded on GPU {self.retrieval_gpu_id}")
    
    @torch.no_grad()
    def _index_corpus(self):
        """Index the corpus for efficient retrieval."""
        logger.info("Indexing corpus for retrieval...")
        retrieval_device = f"cuda:{self.retrieval_gpu_id}"
        
        # Check if cached corpus exists
        if self.cfg['cache_corpus'] and os.path.exists(self.cfg['cache_corpus']):
            logger.info(f"Loading cached corpus from {self.cfg['cache_corpus']}")
            loaded_corpus = torch.load(self.cfg['cache_corpus'])
            corpus_ids = loaded_corpus[0].to(retrieval_device)
            corpus_vectors = loaded_corpus[1].to(retrieval_device)
            self.corpus_index = (corpus_ids, corpus_vectors)
            return
        
        # Create dataloader for corpus
        dataloader = torch.utils.data.DataLoader(
            self.corpus_dataset,
            batch_size=self.cfg['corpus_bs'],
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=True,
            drop_last=False
        )
        
        logger.info("Computing image embeddings for corpus...")
        corpus_vectors = []
        corpus_ids = []
        
        
        for batch in tqdm(dataloader):
            batch_vectors = F.normalize(
                self.image_embedder.model(batch['image'].to(retrieval_device)), 
                dim=-1
            )
            corpus_vectors.append(batch_vectors)
            corpus_ids.append(batch['id'].to(retrieval_device))
            # corpus_vectors.append(batch_vectors.cpu())
            # corpus_ids.append(batch['id'].cpu())

        corpus_vectors = torch.cat(corpus_vectors)
        corpus_ids = torch.cat(corpus_ids)

        # Sort by id: important!
        arg_ids = torch.argsort(corpus_ids)
        corpus_vectors = corpus_vectors[arg_ids]
        corpus_ids = corpus_ids[arg_ids]

        self.corpus_index = corpus_ids, corpus_vectors
        
        # Cache corpus if path is provided
        if self.cfg['cache_corpus']:
            logger.info(f"Saving corpus index to {self.cfg['cache_corpus']}")
            os.makedirs(os.path.dirname(self.cfg['cache_corpus']), exist_ok=True)
            torch.save(self.corpus_index, self.cfg['cache_corpus'])
            
        logger.info("Corpus indexing complete")
    
    def _generate_caption(self, image_path: str, max_tokens: int = 200,
                         refinement_round: int = 0, current_caption: str = None,
                         retrieved_image_path: str = None, retrieval_rank: int = None) -> str:
        """
        Generate or refine a caption for an image using Gemma3.
        
        Args:
            image_path: Path to the target image
            max_tokens: Maximum number of tokens for caption
            refinement_round: Current refinement iteration (0 = initial caption)
            current_caption: Current caption to refine (if refinement_round > 0)
            retrieved_image_path: Path to the retrieved image (if refinement_round > 0)
            retrieval_rank: Rank of the retrieved image (if refinement_round > 0)
            
        Returns:
            str: Generated or refined caption
        """
        # Different prompts based on whether this is initial caption or refinement
        if refinement_round == 0:
            prompt = (
                "Generate a detailed, descriptive caption of this image for use in image retrieval. "
                "Focus on distinctive visual elements, style, composition, colors, and details that would "
                "help identify this specific image from a large collection. Generate ONLY a detailed caption "
                "for this image. DO NOT include any introductory text or explanations."
            )
            
            # System message for initial captioning
            system_message = f"You are an image captioning assistant that creates concise but detailed descriptions for distinguishing images in retrieval tasks. Keep your description under {max_tokens} tokens."
            
            # Just the image in the user message for initial captioning
            user_content = [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        else:
            # For refinement based on retrieval results
            prompt = (
                f"I'm trying to create a textual query that will accurately retrieve this target image from "
                f"a large collection. My current query is: \"{current_caption}\"\n\n"
                f"When I used this query for image retrieval, I got a result that ranked at position {retrieval_rank}. "
                f"Please analyze both images - the target image and the retrieved result. "
                f"Refine my query to better match the target image and distinguish it from similar images in the collection. "
                f"Focus on unique aspects of the target image that weren't captured in my initial query. "
                f"Generate ONLY the refined query text, without any explanations or surrounding text."
            )
            
            # System message for refinement
            system_message = (
                f"You are an image retrieval expert. Your task is to refine textual queries to better match "
                f"target images in retrieval systems. Keep your refined query under {max_tokens} tokens."
            )
            
            # Include both images in the user message for refinement
            user_content = [
                {"type": "image", "image": image_path},
                {"type": "image", "image": retrieved_image_path},
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
    
    def _retrieve_images(self, caption: str, target_path: str) -> Tuple[List[Dict], int]:
        """
        Retrieve images using the caption and return top results.
        
        Args:
            caption: Text caption to use for retrieval
            target_path: Path to the target image
            
        Returns:
            Tuple containing:
                - List of dictionaries with retrieved image info (path, score, rank)
                - Position of the target image in the retrieved results (-1 if not found)
        """
        # Get the target image ID
        try:
            target_id = self.corpus_dataset.path_to_index(target_path)
        except KeyError:
            logger.warning(f"Target image {target_path} not found in corpus")
            return [], -1
        
        # Encode the caption text
        retrieval_device = f"cuda:{self.retrieval_gpu_id}"
        caption_embedding = F.normalize(self.dialog_encoder([caption]), dim=-1)
        
        # Get similarity scores for all corpus images
        scores = caption_embedding @ self.corpus_index[1].T
        
        # Get ranked indices of images
        arg_ranks = torch.argsort(scores[0], descending=True).cpu().numpy()
        
        # Find where the target appears in the ranking
        target_rank = -1
        for i, idx in enumerate(arg_ranks):
            if idx == target_id:
                target_rank = i
                break
        
        # Get top K retrieved images
        top_results = []
        for i in range(min(self.retrieval_top_k, len(arg_ranks))):
            idx = arg_ranks[i].item()
            result = {
                'corpus_id': idx,
                'path': self.corpus_dataset.corpus[idx],
                'score': scores[0][idx].item(),
                'rank': i
            }
            top_results.append(result)
        
        return top_results, target_rank
    
    def _compare_images(self, target_image_path: str, retrieved_image_path: str, 
                    current_caption: str, max_new_tokens: int = 400) -> str:
        """
        Generate comparison analysis between target and retrieved images.
        
        Args:
            target_image_path: Path to the target image
            retrieved_image_path: Path to the retrieved image
            current_caption: Current caption used for retrieval
            retrieval_rank: Rank of the retrieved image
            target_rank: Rank of the target image in the retrieval results
            
        Returns:
            str: The generated comparison analysis
        """
        prompt = (
            f"Here is my text query for retrieving the first image: \"{current_caption}\". "
            f"But my retrieval system returned the second image instead. "
            f"Compare the content differences between these two images. "
            f"Only tell me what's different and what specific details I should add or modify in my query to better retrieve the target image. "
            f"Be concise and clear."
        )
        
        # System message - simplified
        system_message = f"You are an image analysis assistant that compares images to help improve text queries for image retrieval systems. Keep your analysis under {max_new_tokens} tokens."
        
        # Prepare messages with both images
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": target_image_path},
                    {"type": "image", "image": retrieved_image_path},
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
        
        # Move inputs to GPU
        caption_device = f"cuda:{self.caption_gpu_id}"
        inputs = {k: v.to(caption_device) for k, v in inputs.items()}
        
        # Track input length for extraction
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate text
        with torch.inference_mode():
            generation = self.caption_model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,  # Longer for analysis
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                early_stopping=True,
                repetition_penalty=1.1  
            )
            generation = generation[0][input_len:]
        
        # Decode the output
        comparison = self.caption_processor.decode(generation, skip_special_tokens=True)
        
        return comparison.strip()
    
    def _collect_image_paths(self) -> List[str]:
        """
        Collect image paths from self.queries.
        
        Returns:
            List[str]: List of image paths
        """
        image_paths = []
        
        # Use queries if available
        if self.queries:
            logger.info("Collecting image paths from queries JSON file")
            for query in self.queries:
                img_path = query['img']
                image_path = os.path.join(self.input_dir, img_path)
                
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                else:
                    logger.warning(f"Image path not found: {image_path}")
        
        # Limit the number of images if specified
        if self.max_images is not None and len(image_paths) > self.max_images:
            logger.info(f"Limiting to {self.max_images} images (from {len(image_paths)} found)")
            image_paths = image_paths[:self.max_images]
        
        return image_paths
    
    def process(self):
        """
        Process all images through the pipeline with visual prediction feedback.
        """
        # Initialize models and corpus
        self._setup_caption_model()
        self._setup_retrieval_model()
        self._index_corpus()
        
        # Collect image paths
        image_paths = self._collect_image_paths()
        logger.info(f"Processing {len(image_paths)} images with {self.refinement_rounds} refinement rounds")
        
        # Process each image
        for idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            # Get relative path for organization
            rel_path = os.path.relpath(str(image_path), str(self.input_dir))
            image_id = Path(rel_path).stem
            
            
            # Create a directory to store all rounds for this image
            image_output_dir = self.output_dir / f"image_{image_id}"
            os.makedirs(image_output_dir, exist_ok=True)
            
            # If the image is already processed, skip it
            if os.path.exists(image_output_dir / f"dialogue.json"):
                logger.info(f"Image {image_id} already processed. Skipping.")
                continue
            
            # Create subdirectories within the image directory
            image_captions_dir = image_output_dir / "captions"
            image_retrieved_dir = image_output_dir / "retrieved_images"
            image_comparisons_dir = image_output_dir / "comparisons"
            os.makedirs(image_captions_dir, exist_ok=True)
            os.makedirs(image_retrieved_dir, exist_ok=True)
            os.makedirs(image_comparisons_dir, exist_ok=True)
            
            # Copy original image
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
            initial_caption = self._generate_caption(str(image_path), refinement_round=0)
            
            # Save initial caption
            caption_0_path = self.captions_dir / f"{image_id}_caption_0.txt"
            image_caption_0_path = image_captions_dir / f"caption_0.txt"
            with open(caption_0_path, 'w') as f:
                f.write(initial_caption)
            with open(image_caption_0_path, 'w') as f:
                f.write(initial_caption)
            
            # Retrieve images using initial caption
            top_results, target_rank = self._retrieve_images(initial_caption, image_path)
            
            # Save top retrieved image (for initial round)
            if top_results:
                top_image_path = top_results[0]['path']
                retrieved_0_path = self.retrieved_images_dir / f"{image_id}_retrieved_0.jpg"
                image_retrieved_0_path = image_retrieved_dir / f"retrieved_0.jpg"
                shutil.copy2(top_image_path, retrieved_0_path)
                shutil.copy2(top_image_path, image_retrieved_0_path)
                
                # Generate comparison analysis
                comparison = self._compare_images(
                    str(image_path),
                    str(top_image_path),
                    initial_caption,
                )
                
                # Save comparison
                comparison_0_path = self.comparisons_dir / f"{image_id}_comparison_0.txt"
                image_comparison_0_path = image_comparisons_dir / f"comparison_0.txt"
                with open(comparison_0_path, 'w') as f:
                    f.write(comparison)
                with open(image_comparison_0_path, 'w') as f:
                    f.write(comparison)
            else:
                logger.warning(f"No results retrieved for image {image_id} with initial caption")
                comparison = "No retrieved results to compare."
                retrieved_0_path = None
            
            # Add initial round to dialogue
            dialogue["rounds"].append({
                "round": 0,
                "caption": initial_caption,
                "caption_path": str(caption_0_path),
                "retrieved_image_path": str(retrieved_0_path) if top_results else None,
                "retrieved_rank": 0 if top_results else None,
                "target_rank": target_rank,
                "comparison": comparison,
                "comparison_path": str(comparison_0_path) if top_results else None
            })
            
            # Current caption for refinement
            current_caption = initial_caption
            
            # Perform refinement rounds
            for round_idx in range(1, self.refinement_rounds + 1):
                # Get previous retrieved image path
                previous_retrieved_path = self.retrieved_images_dir / f"{image_id}_retrieved_{round_idx-1}.jpg"
                
                if not os.path.exists(previous_retrieved_path) and top_results:
                    # Fallback to first result if not found
                    previous_retrieved_path = self.retrieved_images_dir / f"{image_id}_retrieved_0.jpg"
                
                if os.path.exists(previous_retrieved_path):
                    # Refine caption based on retrieval feedback
                    refined_caption = self._generate_caption(
                        str(image_path),
                        refinement_round=round_idx,
                        current_caption=current_caption,
                        retrieved_image_path=str(previous_retrieved_path),
                        retrieval_rank=0 if top_results else None
                    )
                    
                    # Save refined caption
                    caption_path = self.captions_dir / f"{image_id}_caption_{round_idx}.txt"
                    image_caption_path = image_captions_dir / f"caption_{round_idx}.txt"
                    with open(caption_path, 'w') as f:
                        f.write(refined_caption)
                    with open(image_caption_path, 'w') as f:
                        f.write(refined_caption)
                    
                    # Retrieve images using refined caption
                    top_results, target_rank = self._retrieve_images(refined_caption, image_path)
                    
                    # Save top retrieved image
                    if top_results:
                        top_image_path = top_results[0]['path']
                        retrieved_path = self.retrieved_images_dir / f"{image_id}_retrieved_{round_idx}.jpg"
                        image_retrieved_path = image_retrieved_dir / f"retrieved_{round_idx}.jpg"
                        shutil.copy2(top_image_path, retrieved_path)
                        shutil.copy2(top_image_path, image_retrieved_path)
                        
                        # Generate comparison analysis
                        comparison = self._compare_images(
                            str(image_path),
                            str(top_image_path),
                            refined_caption,
                        )
                        
                        # Save comparison
                        comparison_path = self.comparisons_dir / f"{image_id}_comparison_{round_idx}.txt"
                        image_comparison_path = image_comparisons_dir / f"comparison_{round_idx}.txt"
                        with open(comparison_path, 'w') as f:
                            f.write(comparison)
                        with open(image_comparison_path, 'w') as f:
                            f.write(comparison)
                    else:
                        logger.warning(f"No results retrieved for image {image_id} with refined caption at round {round_idx}")
                        comparison = "No retrieved results to compare."
                        retrieved_path = None
                        comparison_path = None
                    
                    # Add round to dialogue
                    dialogue["rounds"].append({
                        "round": round_idx,
                        "caption": refined_caption,
                        "caption_path": str(caption_path),
                        "retrieved_image_path": str(retrieved_path) if top_results else None,
                        "retrieved_rank": 0 if top_results else None,
                        "target_rank": target_rank,
                        "comparison": comparison,
                        "comparison_path": str(comparison_path) if top_results else None
                    })
                    
                    # Update current caption for next round
                    current_caption = refined_caption
                else:
                    logger.warning(f"Previous retrieved image not found for image {image_id} at round {round_idx}")
                    break
            
            # Save the dialogue JSON
            dialogue_path = self.dialogue_dir / f"{image_id}_dialogue.json"
            image_dialogue_path = image_output_dir / f"dialogue.json"
            with open(dialogue_path, 'w') as f:
                json.dump(dialogue, f, indent=2)
            with open(image_dialogue_path, 'w') as f:
                json.dump(dialogue, f, indent=2)
            
            logger.info(f"Processed image {idx+1}/{len(image_paths)}: {rel_path} with {self.refinement_rounds} refinement rounds")
                
    
    def run(self):
        """Run the complete pipeline with visual prediction feedback."""
        logger.info(f"Starting Visual Prediction Feedback Pipeline")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Caption model: {self.caption_model_id} on GPU {self.caption_gpu_id}")
        logger.info(f"Retrieval model: {self.retrieval_baseline} on GPU {self.retrieval_gpu_id}")
        logger.info(f"Refinement rounds: {self.refinement_rounds}")
        
        start_time = time.time()
        self.process()
        end_time = time.time()
        
        logger.info(f"Pipeline complete! Processed in {end_time - start_time:.2f} seconds")


def parse_args():
    parser = argparse.ArgumentParser(description="Visual Prediction Feedback Pipeline")
    
    # Required arguments
    parser.add_argument("--input_dir", type=str, default="/data/mscoco", 
                        help="Input directory containing the image dataset")
    parser.add_argument("--queries_path", type=str, 
                        default="ChatIR/dialogues/VisDial_v1.0_queries_val.json",
                        help="Path to queries JSON file")
    parser.add_argument("--corpus_path", type=str,
                        default="ChatIR/ChatIR_Protocol/Search_Space_val_50k.json",
                        help="Path to corpus JSON file")
    parser.add_argument("--output_dir", type=str, 
                        default="results_genir/visual_prediction_feedback_output",
                        help="Output directory for results")
    
    # Model configuration
    parser.add_argument("--caption_model_id", type=str, default="google/gemma-3-4b-it",
                        help="Model ID for captioning")
    parser.add_argument("--retrieval_baseline", type=str, default="blip-zero-shot",
                        help="Baseline model for retrieval (blip-zero-shot or clip-zero-shot)")
    parser.add_argument("--caption_gpu_id", type=int, default=1,
                        help="GPU ID for caption model")
    parser.add_argument("--retrieval_gpu_id", type=int, default=1,
                        help="GPU ID for retrieval model")
    
    # Pipeline configuration
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to process")
    parser.add_argument("--refinement_rounds", type=int, default=10,
                        help="Number of refinement rounds")
    parser.add_argument("--cache_corpus", type=str, default="",
                        help="Path to cache the indexed corpus")
    parser.add_argument("--corpus_bs", type=int, default=500,
                        help="Batch size for corpus processing")
    parser.add_argument("--queries_bs", type=int, default=500,
                        help="Batch size for queries processing")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="Number of workers for data loading")
    parser.add_argument("--retrieval_top_k", type=int, default=5,
                        help="Number of top retrieved images to consider")
    
    return parser.parse_args()


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    args = parse_args()
    
    # Create and run the pipeline
    pipeline = VisualPredictionFeedbackPipeline(
        input_dir=args.input_dir,
        queries_path=args.queries_path,
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        max_images=args.max_images,
        caption_model_id=args.caption_model_id,
        retrieval_baseline=args.retrieval_baseline,
        caption_gpu_id=args.caption_gpu_id,
        retrieval_gpu_id=args.retrieval_gpu_id,
        refinement_rounds=args.refinement_rounds,
        cache_corpus=args.cache_corpus,
        corpus_bs=args.corpus_bs,
        queries_bs=args.queries_bs,
        num_workers=args.num_workers,
        retrieval_top_k=args.retrieval_top_k
    )
    pipeline.run()