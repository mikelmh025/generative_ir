import torch
import tqdm
import os.path
import json
import torch.nn.functional as F
from baselines import ImageEmbedder, CLIP_ZERO_SHOT_BASELINE, BLIP_BASELINE
import argparse

class GenIRCorpus(torch.utils.data.Dataset):
    """Dataset class for the corpus with support for fake generated images"""
    def __init__(self, corpus_path, preprocessor, fake_images_dir=None, dataset_root='/data/mscoco'):
        with open(corpus_path) as f:
            self.corpus = json.load(f)
        self.corpus = [os.path.join(dataset_root, path) for path in self.corpus]
            
        self.preprocessor = preprocessor
        self.path2id = {self.corpus[i]: i for i in range(len(self.corpus))}
        
        self.fake_images_dir = fake_images_dir
        
    def __len__(self):
        return len(self.corpus)

    def path_to_index(self, path):
        """For finding a target image fast"""
        return self.path2id[path]
    
    def get_original_image(self, index):
        """Get the original image"""
        image = self.preprocessor(self.corpus[index])
        return image
    
    def get_fake_image(self, index, dialog_length):
        """Get the fake image generated for the specified dialog length"""
        if self.fake_images_dir is None:
            raise ValueError("Fake images directory not specified")
        
        # Extract the original image filename without the path
        orig_filename = os.path.basename(self.corpus[index])
        if orig_filename.endswith('.jpg'):
            # Remove the extension
            orig_filename = orig_filename[:-4]
        elif orig_filename.endswith('.png'):
            orig_filename = orig_filename[:-4]
        
        # Construct the fake image path
        fake_img_path = os.path.join(self.fake_images_dir, f"{orig_filename}_{dialog_length}.jpg")
        
        # Check if the fake image exists
        if not os.path.exists(fake_img_path):
            return None  # Don't use fallback, return None to indicate missing image
        
        # Process and return the fake image
        image = self.preprocessor(fake_img_path)
        return image

    def __getitem__(self, i):
        # Default behavior is to return the original image
        image = self.preprocessor(self.corpus[i])
        return {'id': i, 'image': image}


class GenIRQueries(torch.utils.data.Dataset):
    """Dataset class for the queries with support for matching dialog length to fake images"""
    def __init__(self, cfg, queries_path,dataset_root='/data/mscoco'):
        with open(queries_path) as f:
            self.queries = json.load(f)
        
        for q in self.queries:
            q['img'] = os.path.join(dataset_root, q['img'])
        
        self.dialog_length = None  # Set the dialog length to evaluate on
        self.cfg = cfg

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        assert self.dialog_length is not None, "Please set self.dialog_length=<DIALOG_LENGTH> to any number [0,..,10]"
        target_path = self.queries[i]['img']
        # Concatenate the partial dialog information with a predefined separator
        text = self.cfg['sep_token'].join(self.queries[i]['dialog'][:self.dialog_length + 1])
        return {'text': text, 'target_path': target_path, 'dialog_length': self.dialog_length}


class GenIREval:
    """Evaluation class with support for fake image-based retrieval"""
    def __init__(self, cfg, dialog_encoder, image_embedder: ImageEmbedder):
        self.dialog_encoder = dialog_encoder
        self.image_embedder = image_embedder

        self.cfg = cfg
        self.corpus = None
        self.corpus_dataset = GenIRCorpus(
            self.cfg['corpus_path'], 
            self.image_embedder.processor,
            fake_images_dir=self.cfg.get('fake_images_dir', None),
            dataset_root=self.cfg['img_root']
        )

    def _get_recalls_text(self, dataloader, dialog_length):
        """Original text-to-image retrieval approach"""
        # Set dialog length
        dataloader.dataset.dialog_length = dialog_length
        recalls = []
        
        for batch in tqdm.tqdm(dataloader):
            target_ids = torch.tensor([self.corpus_dataset.path_to_index(p) for p in batch['target_path']]).unsqueeze(1).to(self.cfg['device'])
            pred_vec = F.normalize(self.dialog_encoder(batch['text']), dim=-1)
            
            # batch recalls
            scores = pred_vec @ self.corpus[1].T
            arg_ranks = torch.argsort(scores, descending=True, dim=1).long()
            target_recall = ((arg_ranks - target_ids) == 0).nonzero()[:, 1]
            recalls.append(target_recall.cpu())

        # No need to move to CPU here as we'll do it later before calling cumulative_hits_per_round
        return torch.cat(recalls)
    
    def _get_recalls_fake_image(self, dataloader, dialog_length):
        """Use fake images as query instead of text embeddings"""
        # Set dialog length
        dataloader.dataset.dialog_length = dialog_length
        recalls = []
        valid_targets_count = 0
        skipped_count = 0
        
        for batch in tqdm.tqdm(dataloader):
            batch_target_paths = batch['target_path']
            batch_target_ids = []
            fake_images = []
            valid_indices = []  # Track which targets have valid fake images
            
            # Collect valid fake images and their corresponding target IDs
            for i, path in enumerate(batch_target_paths):
                index = self.corpus_dataset.path_to_index(path)
                
                # Extract the original image filename without the path
                orig_filename = os.path.basename(path)
                if orig_filename.endswith('.jpg'):
                    orig_filename = orig_filename[:-4]
                elif orig_filename.endswith('.png'):
                    orig_filename = orig_filename[:-4]
                
                # Check if the fake image exists before trying to load it
                fake_img_path = os.path.join(self.cfg['fake_images_dir'], f"{orig_filename}_{dialog_length}.jpg")
                
                if os.path.exists(fake_img_path):
                    # Only include this target if the fake image exists
                    fake_img = self.corpus_dataset.preprocessor(fake_img_path)
                    fake_images.append(fake_img)
                    batch_target_ids.append(index)
                    valid_indices.append(i)
                    valid_targets_count += 1
                else:
                    skipped_count += 1
            
            # Skip this batch if no valid fake images were found
            if not fake_images:
                continue
                
            # Convert target IDs to tensor
            target_ids = torch.tensor(batch_target_ids).unsqueeze(1).to(self.cfg['device'])
            
            # Stack images and get embeddings
            fake_images = torch.stack(fake_images).to(self.cfg['device'])
            pred_vec = F.normalize(self.image_embedder.model(fake_images), dim=-1)
            
            # Compute scores
            scores = pred_vec @ self.corpus[1].T
            arg_ranks = torch.argsort(scores, descending=True, dim=1).long()
            target_recall = ((arg_ranks - target_ids) == 0).nonzero()[:, 1]
            recalls.append(target_recall)
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} targets due to missing fake images. Used {valid_targets_count} targets.")
            
        if not recalls:
            print(f"Warning: No valid recalls for dialog length {dialog_length}. All fake images were missing.")
            return torch.tensor([])
            
        return torch.cat(recalls)

    def run_text_retrieval(self, hits_at=10):
        """Run evaluation using the original text-to-image approach"""
        assert self.corpus, "Prepare corpus first (self.index_corpus())"
        dataset = GenIRQueries(self.cfg, self.cfg['queries_path'], dataset_root=self.cfg['img_root'])
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg['queries_bs'],
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=True,
            drop_last=False
        )
        
        all_recalls = []
        for dl in range(11):
            print(f"Calculate recalls for text retrieval with dialog length {dl}...")
            dialog_recalls = self._get_recalls_text(dataloader, dialog_length=dl)
            all_recalls.append(dialog_recalls)

        # Combine all recalls and use the cumulative_hits_per_round function
        all_recalls_tensor = torch.cat(all_recalls).view(11, -1).t()
        
        # Move tensor to CPU before passing to cumulative_hits_per_round
        all_recalls_tensor = all_recalls_tensor.cpu()
        hits_results = cumulative_hits_per_round(all_recalls_tensor, hitting_recall=hits_at).tolist()
        
        print(f"====== Results for Text-to-Image Hits@{hits_at} ====== ")
        for dl in range(11):
            print(f"\t Dialog Length: {dl}: {round(hits_results[dl], 2)}%")
        
        return hits_results
    
    def run_fake_image_retrieval(self, hits_at=10):
        """Run evaluation using the fake image-to-real image approach"""
        assert self.corpus, "Prepare corpus first (self.index_corpus())"
        assert self.cfg.get('fake_images_dir'), "Fake images directory must be specified"
        
        dataset = GenIRQueries(self.cfg, self.cfg['queries_path'], dataset_root=self.cfg['img_root'])
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg['queries_bs'],
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=True,
            drop_last=False
        )
        
        # First, collect information about the total number of samples in the dataset
        total_samples = len(dataset)
        
        # Create a consistent subset of samples to evaluate on (those with all fake images)
        consistent_sample_indices = self._find_consistent_samples(dataloader)
        print(f"Found {len(consistent_sample_indices)} samples with fake images for all dialog lengths")
        
        if len(consistent_sample_indices) == 0:
            print("No samples have fake images for all dialog lengths. Will report results for each dialog length separately.")
            # Fall back to the original approach if no consistent samples
            recalls_by_dl = {}
            sample_counts = [0] * 11
            
            for dl in range(11):
                print(f"Calculate recalls for fake image retrieval with dialog length {dl}...")
                # Set dialog length in the dataset
                dataloader.dataset.dialog_length = dl
                dialog_recalls = self._get_recalls_fake_image(dataloader, dialog_length=dl)
                
                if len(dialog_recalls) == 0:
                    print(f"No valid samples for dialog length {dl}, skipping in final results")
                    sample_counts[dl] = 0
                else:
                    # Move to CPU to avoid device issues
                    recalls_by_dl[dl] = dialog_recalls.cpu()
                    sample_counts[dl] = len(dialog_recalls)
            
            # Calculate hit rates for each dialog length individually
            hits_results = [0] * 11
            for dl in recalls_by_dl:
                recalls = recalls_by_dl[dl]
                hit_count = (recalls < hits_at).sum().item()
                hit_rate = (hit_count / len(recalls)) * 100
                hits_results[dl] = hit_rate
            
            print(f"====== Results for Fake-Image-to-Image Hits@{hits_at} (Individual) ====== ")
            for dl in range(11):
                if sample_counts[dl] > 0:
                    print(f"\t Dialog Length: {dl}: {round(hits_results[dl], 2)}% (samples: {sample_counts[dl]})")
                else:
                    print(f"\t Dialog Length: {dl}: N/A (no valid samples)")
            
            return hits_results, sample_counts
        
        else:
            # If we have consistent samples, create a filtered dataset
            # Instead of using Subset, recreate the dataset with only the consistent samples
            all_recalls = []
            
            for dl in range(11):
                print(f"Calculate recalls for fake image retrieval with dialog length {dl} (consistent subset)...")
                # Need to set dialog length for the main dataset
                dataset.dialog_length = dl
                
                # Process only the consistent samples directly
                batch_target_paths = []
                batch_target_ids = []
                fake_images = []
                
                # Process samples in batches to avoid memory issues
                batch_size = self.cfg['queries_bs']
                for start_idx in range(0, len(consistent_sample_indices), batch_size):
                    end_idx = min(start_idx + batch_size, len(consistent_sample_indices))
                    batch_indices = consistent_sample_indices[start_idx:end_idx]
                    
                    # Get paths for these samples
                    for idx in batch_indices:
                        sample = dataset[idx]
                        path = sample['target_path']
                        batch_target_paths.append(path)
                        
                        # Get target ID and fake image
                        index = self.corpus_dataset.path_to_index(path)
                        batch_target_ids.append(index)
                        
                        # Get the fake image
                        orig_filename = os.path.basename(path)
                        if orig_filename.endswith('.jpg'):
                            orig_filename = orig_filename[:-4]
                        elif orig_filename.endswith('.png'):
                            orig_filename = orig_filename[:-4]
                        
                        fake_img_path = os.path.join(self.cfg['fake_images_dir'], f"{orig_filename}_{dl}.jpg")
                        fake_img = self.corpus_dataset.preprocessor(fake_img_path)
                        fake_images.append(fake_img)
                    
                    # Process this mini-batch if we have samples
                    if fake_images:
                        # Convert target IDs to tensor
                        target_ids = torch.tensor(batch_target_ids).unsqueeze(1).to(self.cfg['device'])
                        
                        # Stack images and get embeddings
                        fake_images_tensor = torch.stack(fake_images).to(self.cfg['device'])
                        pred_vec = F.normalize(self.image_embedder.model(fake_images_tensor), dim=-1)
                        
                        # Compute scores
                        scores = pred_vec @ self.corpus[1].T
                        arg_ranks = torch.argsort(scores, descending=True, dim=1).long()
                        target_recall = ((arg_ranks - target_ids) == 0).nonzero()[:, 1]
                        
                        # Move to CPU and store
                        all_recalls.append(target_recall.cpu())
                        
                        # Clear lists for next batch
                        batch_target_paths = []
                        batch_target_ids = []
                        fake_images = []
            
            # Combine all recalls for all dialog lengths
            all_recalls_tensor = torch.cat(all_recalls).view(11, -1).t()
            
            # Calculate cumulative hits
            hits_results = cumulative_hits_per_round(all_recalls_tensor, hitting_recall=hits_at).tolist()
            
            print(f"====== Results for Fake-Image-to-Image Hits@{hits_at} (Cumulative) ====== ")
            for dl in range(11):
                print(f"\t Dialog Length: {dl}: {round(hits_results[dl], 2)}% (consistent samples: {len(consistent_sample_indices)})")
            
            return hits_results, [len(consistent_sample_indices)] * 11

    def _find_consistent_samples(self, dataloader):
        """Find samples that have fake images for all dialog lengths"""
        # This is a helper function to identify samples that have fake images 
        # for all dialog lengths (0-10), so we can make a fair comparison
        valid_samples = set(range(len(dataloader.dataset)))
        
        # To avoid loading the entire dataset multiple times, we'll check file existence directly
        filenames_by_idx = {}
        
        # First gather all image filenames
        original_dl = dataloader.dataset.dialog_length
        dataloader.dataset.dialog_length = 0  # Set to any valid value to avoid assertion errors
        
        print("Gathering image filenames...")
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
            for i, path in enumerate(batch['target_path']):
                # Calculate the global index of this sample
                global_idx = batch_idx * dataloader.batch_size + i
                if global_idx >= len(dataloader.dataset):
                    continue  # Skip if we're out of bounds (can happen in the last batch)
                
                # Extract the original image filename
                orig_filename = os.path.basename(path)
                if orig_filename.endswith('.jpg'):
                    orig_filename = orig_filename[:-4]
                elif orig_filename.endswith('.png'):
                    orig_filename = orig_filename[:-4]
                
                filenames_by_idx[global_idx] = orig_filename
        
        # Now check which samples have fake images for all dialog lengths
        print("Checking for consistent samples across all dialog lengths...")
        for dl in range(11):
            current_valid = set()
            
            for idx, filename in tqdm.tqdm(filenames_by_idx.items(), desc=f"Dialog length {dl}"):
                fake_img_path = os.path.join(self.cfg['fake_images_dir'], f"{filename}_{dl}.jpg")
                if os.path.exists(fake_img_path):
                    current_valid.add(idx)
            
            # Keep only samples that have been valid for all dialog lengths so far
            valid_samples = valid_samples.intersection(current_valid)
            print(f"Dialog length {dl}: {len(current_valid)} valid samples, {len(valid_samples)} consistent so far")
            
            if not valid_samples:
                break  # No need to continue if we already have no valid samples
        
        # Restore original dialog length
        dataloader.dataset.dialog_length = original_dl
        
        return list(valid_samples)

    def index_corpus(self):
        """Prepare corpus (image search space)"""
        if self.cfg['cache_corpus'] and os.path.exists(self.cfg['cache_corpus']):
            print(f"<<<<Cached corpus has been loaded: {self.cfg['cache_corpus']} >>>>>")
            print(f"Warning: Make sure this corpus has been indexed with the right image embedder!")
            self.corpus = torch.load(self.cfg['cache_corpus'])
            return
        
        dataloader = torch.utils.data.DataLoader(
            self.corpus_dataset,
            batch_size=self.cfg['corpus_bs'],
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=True,
            drop_last=False
        )
        
        print("Preparing corpus (search space)...")
        corpus_vectors = []
        corpus_ids = []
        for batch in tqdm.tqdm(dataloader):
            batch_vectors = F.normalize(self.image_embedder.model(batch['image'].to(self.cfg['device'])), dim=-1)
            corpus_vectors.append(batch_vectors)
            corpus_ids.append(batch['id'].to(self.cfg['device']))

        corpus_vectors = torch.cat(corpus_vectors)
        corpus_ids = torch.cat(corpus_ids)

        # sort by id: important!
        arg_ids = torch.argsort(corpus_ids)
        corpus_vectors = corpus_vectors[arg_ids]
        corpus_ids = corpus_ids[arg_ids]

        self.corpus = corpus_ids, corpus_vectors
        if self.cfg['cache_corpus']:
            torch.save(self.corpus, self.cfg['cache_corpus'])


def get_first_hitting_time(target_recall, hitting_recall=10):
    """ returns (11, n) tensor with hitting time in each round (0, 11). inf indicate a miss (no hit after 11 rounds) """
    # This function expects target_recall to be reshapable into (11, n) where n is the number of samples
    target_recalls = target_recall
    
    # Move everything to CPU to avoid device mismatch errors
    if isinstance(target_recalls, torch.Tensor):
        target_recalls = target_recalls.cpu()
    else:
        target_recalls = torch.tensor(target_recalls, device='cpu')
    
    # Ensure we have the right shape
    if target_recalls.dim() != 2:
        raise ValueError(f"Expected target_recalls to be a 2D tensor, got shape {target_recalls.shape}")
    
    if target_recalls.shape[0] != 11:
        # If the first dimension isn't 11, transpose if possible
        if target_recalls.shape[1] == 11:
            target_recalls = target_recalls.t()
        else:
            raise ValueError(f"Expected target_recalls to have a dimension of 11, got shape {target_recalls.shape}")
    
    hits = (target_recalls < hitting_recall)
    final_hits = torch.inf * torch.ones(target_recalls.shape[1], device='cpu')

    hitting_times = []
    for ro_i in range(11):
        rh = hits[ro_i, :]
        final_hits[rh] = torch.min(final_hits[rh], torch.ones(final_hits[rh].shape, device='cpu') * ro_i)
        hitting_times.append(final_hits.clone())

    return torch.stack(hitting_times)


def cumulative_hits_per_round(target_recall, hitting_recall=10):
    """ return calculation of avg number of hits until round x"""
    if type(hitting_recall) is tuple:
        assert len(hitting_recall) == 1
        hitting_recall = hitting_recall[0]
    
    # Ensure target_recall is on CPU
    if isinstance(target_recall, torch.Tensor):
        target_recall = target_recall.cpu()
        
    ht_times = get_first_hitting_time(target_recall, hitting_recall)
    return ((ht_times < torch.inf).sum(dim=-1) * 100 / ht_times[0].shape[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatIR Configuration')
    
    # Add arguments with default values
    parser.add_argument('--cuda_device', type=str, default='0',
                        help='CUDA device index to use')
    parser.add_argument('--corpus_bs', type=int, default=500,
                        help='Corpus batch size')
    parser.add_argument('--queries_bs', type=int, default=500,
                        help='Queries batch size')
    parser.add_argument('--num_workers', type=int, default=128,
                        help='Number of workers')
    parser.add_argument('--sep_token', type=str, default=', ',
                        help='Separation between dialog rounds')
    parser.add_argument('--cache_corpus', type=str, default='',
                        help='Cache path for saving indexed corpus')
    parser.add_argument('--queries_path', type=str, 
                        default='ChatIR/dialogues/VisDial_v1.0_queries_val.json',
                        help='Path to queries file')
    parser.add_argument('--corpus_path', type=str,
                        default='ChatIR/ChatIR_Protocol/Search_Space_val_50k.json',
                        help='Path to corpus file')
    parser.add_argument('--img_root', type=str, default='/data/mscoco',
                    help='Root directory for images')
    parser.add_argument('--fake_images_dir', type=str,
                        default='results_genir/caption_refinement_output_infinity/generated_images/',
                        help='Directory containing fake/generated images')
    parser.add_argument('--baseline', type=str, default='blip-zero-shot',
                        help='Baseline model to use (blip-zero-shot or clip-zero-shot)')
    parser.add_argument('--hits_at', type=int, default=10,
                        help='Hits@K value for evaluation')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    # Create config dictionary from arguments
    cfg = {
        'corpus_bs': args.corpus_bs,
        'queries_bs': args.queries_bs,
        'num_workers': args.num_workers,
        'sep_token': args.sep_token,
        'cache_corpus': args.cache_corpus,
        'queries_path': args.queries_path,
        'corpus_path': args.corpus_path,
        'img_root': args.img_root,
        'device': 'cuda',
        'fake_images_dir': args.fake_images_dir,
    }


    with torch.no_grad():
        baseline = args.baseline # 'blip-zero-shot' or 'clip-zero-shot'
        hits_at = args.hits_at
        print(f"Using baseline: {baseline}")
        print(f"Using hits_at: {hits_at}")
        

        if baseline == 'clip-zero-shot':
            if not cfg['cache_corpus']:
                cfg['cache_corpus'] = "temp/corpus_clip_16.pth"
            dialog_encoder, image_embedder = CLIP_ZERO_SHOT_BASELINE()
        else:
            if not cfg['cache_corpus']:
                cfg['cache_corpus'] = "temp/corpus_blip_small.pth"
            dialog_encoder, image_embedder = BLIP_BASELINE()
        
        # Create evaluator
        evaluator = GenIREval(cfg, dialog_encoder, image_embedder)
        evaluator.index_corpus()
        
        # Run traditional text-to-image retrieval. (This one has VRAM leak issues from the original ChatIR code)
        print("\nRunning text-to-image retrieval...")
        text_results = evaluator.run_text_retrieval(hits_at=hits_at)
        
        # Run fake-image-to-image retrieval
        print("\nRunning fake-image-to-image retrieval...")
        fake_image_results, sample_counts = evaluator.run_fake_image_retrieval(hits_at=hits_at)
        
        # Compare results
        print(f"\n====== Comparison of Text vs Fake Image Retrieval (Hits@{hits_at}) ======")
        print("Dialog Length | Text Retrieval | Fake Image Retrieval | Difference")
        print("----------------------------------------------------------------")
        for dl in range(11):
            if sample_counts[dl] > 0:
                diff = fake_image_results[dl] - text_results[dl]
                diff_sign = "+" if diff > 0 else ""
                print(f"{dl:12d} | {text_results[dl]:13.2f}% | {fake_image_results[dl]:19.2f}% | {diff_sign}{diff:.2f}%")
            else:
                print(f"{dl:12d} | {text_results[dl]:13.2f}% | {'N/A':>19} | {'N/A':>9}")