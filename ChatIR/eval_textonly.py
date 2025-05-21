
import torch
import tqdm
import os
import os.path
import json
import torch.nn.functional as F
from baselines import ImageEmbedder, CLIP_ZERO_SHOT_BASELINE, BLIP_BASELINE
import argparse

class Corpus(torch.utils.data.Dataset):
    """ Dataset class for the corpus images (the 50k potential candidates)"""
    def __init__(self, corpus_path, preprocessor, dataset_root = '/data/mscoco'):
        with open(corpus_path) as f:
            self.corpus = json.load(f)
        
        self.corpus = [os.path.join(dataset_root, path) for path in self.corpus]
            
        self.preprocessor = preprocessor
        self.path2id = {self.corpus[i]: i for i in range(len(self.corpus))}

    def __len__(self):
        return len(self.corpus)

    def path_to_index(self, path):
        """ For finding a target image fast"""
        return self.path2id[path]

    def __getitem__(self, i):
        image = self.preprocessor(self.corpus[i])  # Load and prepare image
        return {'id': i, 'image': image}


class Queries(torch.utils.data.Dataset):
    """ Dataset class for the queries and their targets (dialog and image)"""
    def __init__(self, cfg, queries_path, dataset_root = '/data/mscoco'):
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
        return {'text': text, 'target_path': target_path}


def get_first_hitting_time(target_recall, hitting_recall=10):
    """ 
    Returns (11, n) tensor with hitting time in each round (0, 11). 
    inf indicates a miss (no hit after 11 rounds) 
    """
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
    """ Return calculation of avg number of hits until round x"""
    if type(hitting_recall) is tuple:
        assert len(hitting_recall) == 1
        hitting_recall = hitting_recall[0]
    
    # Ensure target_recall is on CPU
    if isinstance(target_recall, torch.Tensor):
        target_recall = target_recall.cpu()
        
    ht_times = get_first_hitting_time(target_recall, hitting_recall)
    return ((ht_times < torch.inf).sum(dim=-1) * 100 / ht_times[0].shape[0])


class ImprovedChatIREval:
    """
    Improved evaluation class that supports both per-dialog-length and accumulative recall metrics.
    """
    def __init__(self, cfg, dialog_encoder, image_embedder: ImageEmbedder):
        self.dialog_encoder = dialog_encoder  # In paper was referred as "Image Retriever"
        self.image_embedder = image_embedder  # Image encoder

        self.cfg = cfg
        self.corpus = None
        self.corpus_dataset = Corpus(self.cfg['corpus_path'], self.image_embedder.processor, dataset_root=self.cfg['img_root'])
        
        # Track valid samples to ensure consistent evaluation
        self.valid_sample_ids = None
        
    def get_external_text(self, batch, dialog_length, text_root):
        """
        Get external text (e.g., from Gemma3) for the given batch and dialog length.
        Returns text list and valid indices.
        """
        text_list = []
        valid_indices = []
        valid_ids = []
        
        for i in range(len(batch['text'])):
            target_name = batch['target_path'][i].split('/')[-1].split('.')[0]
            text_path = f'{text_root}/{target_name}_caption_{dialog_length}.txt'
            
            # Check if file exists before attempting to open it
            if os.path.exists(text_path):
                text_lines = []
                with open(text_path, 'r') as f:
                    for line in f:
                        text_lines.append(line.strip())
                external_text = ', '.join(text_lines)
                text_list.append(external_text)
                valid_indices.append(i)
                valid_ids.append(target_name)
            else:
                # Skip this case - don't add to text_list or valid_indices
                pass
        
        return text_list, valid_indices, valid_ids
    
    def _find_consistent_samples(self, dataloader, text_root):
        """
        Find samples that have external text for all dialog lengths.
        This ensures we can calculate accumulative recall on a consistent set of samples.
        """
        print("Finding samples with text for all dialog lengths...")
        
        # Get all sample paths from dataset
        all_paths = []
        original_dl = dataloader.dataset.dialog_length
        dataloader.dataset.dialog_length = 0  # Set to any valid value to avoid assertion errors
        
        for batch in dataloader:
            all_paths.extend(batch['target_path'])
        
        # Extract target names from paths
        all_target_names = [path.split('/')[-1].split('.')[0] for path in all_paths]
        
        # Check which samples have text for all dialog lengths
        valid_samples = set(all_target_names)
        
        for dl in range(11):
            current_valid = set()
            for target_name in all_target_names:
                text_path = f'{text_root}/{target_name}_caption_{dl}.txt'
                if os.path.exists(text_path):
                    current_valid.add(target_name)
            
            # Keep only samples that have been valid for all dialog lengths so far
            valid_samples = valid_samples.intersection(current_valid)
            print(f"Dialog length {dl}: {len(current_valid)} valid samples, {len(valid_samples)} consistent so far")
        
        # Restore original dialog length
        dataloader.dataset.dialog_length = original_dl
        
        return list(valid_samples)
    
    def _get_recalls_standard(self, dataloader, dialog_length):
        """
        Get recalls for standard text-to-image retrieval.
        """
        # Set dialog length
        dataloader.dataset.dialog_length = dialog_length
        recalls = []
        
        # Keep track of which samples we're using
        processed_target_names = []
        
        for batch in tqdm.tqdm(dataloader):
            # If we have valid_sample_ids, filter the batch to only include those samples
            if self.valid_sample_ids:
                valid_indices = []
                for i, path in enumerate(batch['target_path']):
                    target_name = path.split('/')[-1].split('.')[0]
                    if target_name in self.valid_sample_ids:
                        valid_indices.append(i)
                
                if not valid_indices:
                    continue
                
                # Filter batch to only include valid samples
                texts = [batch['text'][i] for i in valid_indices]
                target_paths = [batch['target_path'][i] for i in valid_indices]
                
                # Record processed target names for debugging
                processed_target_names.extend([path.split('/')[-1].split('.')[0] for path in target_paths])
            else:
                texts = batch['text']
                target_paths = batch['target_path']
            
            # Get target IDs for the valid paths
            target_ids = torch.tensor([self.corpus_dataset.path_to_index(p) for p in target_paths]).unsqueeze(1).to(self.cfg['device'])
            
            # Process text
            pred_vec = F.normalize(self.dialog_encoder(texts), dim=-1)
            
            # Compute scores and ranks
            scores = pred_vec @ self.corpus[1].T
            arg_ranks = torch.argsort(scores, descending=True, dim=1).long()
            target_recall = ((arg_ranks - target_ids) == 0).nonzero()[:, 1]
            recalls.append(target_recall.cpu())  # Move to CPU right away to avoid device issues

        # Handle case where no valid samples were found
        if not recalls:
            return torch.tensor([], dtype=torch.long)
            
        return torch.cat(recalls)
    
    def _get_recalls_external_text(self, dataloader, dialog_length, text_root):
        """
        Get recalls using external text (e.g., from Gemma3).
        """
        # Set dialog length
        dataloader.dataset.dialog_length = dialog_length
        recalls = []
        
        # Keep track of which samples we're using
        processed_target_names = []
        
        for batch in tqdm.tqdm(dataloader):
            # Get external text and valid indices
            if self.valid_sample_ids:
                # First get all external text entries
                text_list, valid_indices, valid_ids = self.get_external_text(batch, dialog_length, text_root)
                
                # Then filter to only include consistent samples
                filtered_text_list = []
                filtered_valid_indices = []
                
                for i, target_id in enumerate(valid_ids):
                    if target_id in self.valid_sample_ids:
                        filtered_text_list.append(text_list[i])
                        filtered_valid_indices.append(valid_indices[i])
                
                text_list = filtered_text_list
                valid_indices = filtered_valid_indices
            else:
                text_list, valid_indices, valid_ids = self.get_external_text(batch, dialog_length, text_root)
            
            # Skip if there are no valid captions
            if not valid_indices:
                continue
                
            # Record processed target names for debugging
            processed_target_names.extend(valid_ids)
                
            # Filter batch to only include entries with valid captions
            target_paths = [batch['target_path'][i] for i in valid_indices]
            target_ids = torch.tensor([self.corpus_dataset.path_to_index(p) for p in target_paths]).unsqueeze(1).to(self.cfg['device'])
            
            # Process valid text entries
            pred_vec = F.normalize(self.dialog_encoder(text_list), dim=-1)
            
            # Compute scores and ranks
            scores = pred_vec @ self.corpus[1].T
            arg_ranks = torch.argsort(scores, descending=True, dim=1).long()
            target_recall = ((arg_ranks - target_ids) == 0).nonzero()[:, 1]
            recalls.append(target_recall.cpu())  # Move to CPU right away to avoid device issues

        # Handle the case where all batches were skipped
        if not recalls:
            return torch.tensor([], dtype=torch.long)
            
        return torch.cat(recalls)
    
    def run_standard_evaluation(self, dataloader, hits_at=10):
        """
        Run standard evaluation using dialog text.
        """
        print("\n====== Standard Text-to-Image Evaluation ======")
        
        # Collect recalls for all dialog lengths
        all_recalls = []
        sample_counts = []
        
        for dl in range(11):
            print(f"Calculate recalls for dialog length {dl}...")
            dialog_recalls = self._get_recalls_standard(dataloader, dialog_length=dl)
            
            if dialog_recalls.numel() > 0:
                all_recalls.append(dialog_recalls)
                sample_counts.append(dialog_recalls.numel())
                # Calculate and report per-dialog hit rate
                # hit_rate = (dialog_recalls < hits_at).float().mean().item() * 100
                # print(f"\t Dialog Length {dl}: {round(hit_rate, 2)}% (samples: {dialog_recalls.numel()})")
                print(f"\t Dialog Length {dl}: samples: {dialog_recalls.numel()}")
            else:
                all_recalls.append(torch.tensor([]))
                sample_counts.append(0)
                print(f"\t Dialog Length {dl}: No valid samples")
        
        # If we have at least one valid dialog length with samples
        if any(count > 0 for count in sample_counts):
            # Reshape all_recalls for cumulative calculation
            # We need to ensure all dialog lengths have the same number of samples
            # and these samples are consistent across dialog lengths
            
            # First, check if we have valid_sample_ids set (meaning we've already filtered for consistent samples)
            if self.valid_sample_ids:
                # Combine all recalls for cumulative calculation
                all_recalls_tensor = torch.stack([r for r in all_recalls if r.numel() > 0]).t()
                all_recalls_tensor = all_recalls_tensor.float()
                
                # If we don't have all 11 dialog lengths, pad with inf for missing ones
                if all_recalls_tensor.shape[0] < 11:
                    padded_tensor = torch.full((all_recalls_tensor.shape[1], 11), float('inf'), dtype=all_recalls_tensor.dtype)
                    valid_dl_indices = [i for i, r in enumerate(all_recalls) if r.numel() > 0]
                    for i, dl_idx in enumerate(valid_dl_indices):
                        padded_tensor[:, dl_idx] = all_recalls_tensor[:, i]
                    all_recalls_tensor = padded_tensor.t()
                
                # Calculate cumulative hits
                cumulative_hits = cumulative_hits_per_round(all_recalls_tensor, hitting_recall=hits_at).tolist()
                
                print(f"\n====== Cumulative Hits@{hits_at} for Standard Text ======")
                for dl in range(11):
                    print(f"\t Dialog Length {dl}: {round(cumulative_hits[dl], 2)}%")
                
                return cumulative_hits, sample_counts
            else:
                print("\nWarning: Inconsistent samples across dialog lengths. Cumulative metrics not calculated.")
                print("Use find_consistent_samples=True for cumulative evaluation.")
                return None, sample_counts
        else:
            print("\nWarning: No valid samples found for any dialog length.")
            return None, sample_counts
    
    def run_external_text_evaluation(self, dataloader, text_root, hits_at=10):
        """
        Run evaluation using external text (e.g., from Gemma3).
        """
        print(f"\n====== External Text Evaluation ({text_root}) ======")
        
        # Collect recalls for all dialog lengths
        all_recalls = []
        sample_counts = []
        
        for dl in range(11):
            print(f"Calculate recalls for external text with dialog length {dl}...")
            dialog_recalls = self._get_recalls_external_text(dataloader, dialog_length=dl, text_root=text_root)
            
            if dialog_recalls.numel() > 0:
                all_recalls.append(dialog_recalls)
                sample_counts.append(dialog_recalls.numel())
                # Calculate and report per-dialog hit rate
                # hit_rate = (dialog_recalls < hits_at).float().mean().item() * 100
                # print(f"\t Dialog Length {dl}: {round(hit_rate, 2)}% (samples: {dialog_recalls.numel()})")
                print(f"\t Dialog Length {dl}: samples: {dialog_recalls.numel()}")
            else:
                all_recalls.append(torch.tensor([]))
                sample_counts.append(0)
                print(f"\t Dialog Length {dl}: No valid samples")
        
        # If we have at least one valid dialog length with samples
        if any(count > 0 for count in sample_counts):
            # If we have valid_sample_ids set (meaning we've already filtered for consistent samples)
            if self.valid_sample_ids:
                # Get indices of valid dialog lengths (those with samples)
                valid_dl_indices = [i for i, r in enumerate(all_recalls) if r.numel() > 0]
                
                # If we have all 11 dialog lengths
                if len(valid_dl_indices) == 11:
                    # Stack the recall tensors
                    all_recalls_tensor = torch.stack(all_recalls).t()
                else:
                    # We need to create a padded tensor with inf for missing dialog lengths
                    # First get the number of samples from any valid dialog length
                    num_samples = next(r.numel() for r in all_recalls if r.numel() > 0)
                    padded_tensor = torch.full((num_samples, 11), float('inf'), dtype=torch.long)
                    
                    # Fill in the valid dialog lengths
                    for dl_idx in valid_dl_indices:
                        padded_tensor[:, dl_idx] = all_recalls[dl_idx]
                    
                    all_recalls_tensor = padded_tensor.t()
                
                # Calculate cumulative hits
                cumulative_hits = cumulative_hits_per_round(all_recalls_tensor, hitting_recall=hits_at).tolist()
                
                print(f"\n====== Cumulative Hits@{hits_at} for External Text ======")
                for dl in range(11):
                    if dl in valid_dl_indices:
                        print(f"\t Dialog Length {dl}: {round(cumulative_hits[dl], 2)}%")
                    else:
                        print(f"\t Dialog Length {dl}: No valid samples")
                
                return cumulative_hits, sample_counts
            else:
                print("\nWarning: Inconsistent samples across dialog lengths. Cumulative metrics not calculated.")
                print("Use find_consistent_samples=True for cumulative evaluation.")
                return None, sample_counts
        else:
            print("\nWarning: No valid samples found for any dialog length.")
            return None, sample_counts
    
    def run(self, hits_at=10, find_consistent_samples=False, external_text_root=None):
        """
        Run evaluation for both standard and external text (if provided).
        
        Parameters:
        -----------
        hits_at: int
            The K value for Hits@K metric
        find_consistent_samples: bool
            If True, find samples that have data for all dialog lengths to ensure fair comparison
        external_text_root: str or None
            Path to external text directory. If None, only standard evaluation is run.
        """
        assert self.corpus, "Prepare corpus first (self.index_corpus())"
        
        # Set up dataset and dataloader
        dataset = Queries(self.cfg, self.cfg['queries_path'],dataset_root=self.cfg['img_root'])
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg['queries_bs'],
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=True,
            drop_last=False
        )
        
        # If requested, find samples that have data for all dialog lengths
        if find_consistent_samples:
            if external_text_root:
                # If we have external text, find samples with external text for all dialog lengths
                self.valid_sample_ids = self._find_consistent_samples(dataloader, external_text_root)
                if not self.valid_sample_ids:
                    print("Warning: No samples have external text for all dialog lengths.")
                    print("Will proceed with per-dialog evaluation only.")
            else:
                # If no external text, all samples should be valid for standard evaluation
                print("Using all samples for standard evaluation.")
                # We still set valid_sample_ids to all sample IDs to enable cumulative metrics
                all_paths = []
                for batch in dataloader:
                    all_paths.extend(batch['target_path'])
                self.valid_sample_ids = [path.split('/')[-1].split('.')[0] for path in all_paths]
        
        # Run standard evaluation
        standard_results, standard_counts = self.run_standard_evaluation(dataloader, hits_at)
        
        # Run external text evaluation if requested
        if external_text_root:
            external_results, external_counts = self.run_external_text_evaluation(
                dataloader, external_text_root, hits_at
            )
            
            # Compare results if both evaluations were successful
            if standard_results and external_results:
                print(f"\n====== Comparison of Standard vs External Text (Hits@{hits_at}) ======")
                print("Dialog Length | Standard | External | Difference")
                print("------------------------------------------------")
                for dl in range(11):
                    if dl < len(standard_results) and dl < len(external_results):
                        diff = external_results[dl] - standard_results[dl]
                        diff_sign = "+" if diff > 0 else ""
                        print(f"{dl:12d} | {standard_results[dl]:8.2f}% | {external_results[dl]:8.2f}% | {diff_sign}{diff:.2f}%")
                    else:
                        print(f"{dl:12d} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}")
    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatIR configuration')
    
    # Add arguments with default values from the original config
    parser.add_argument('--cuda_device', type=str, default='0',
                        help='CUDA device ID')
    parser.add_argument('--corpus_bs', type=int, default=500,
                        help='Batch size for corpus')
    parser.add_argument('--queries_bs', type=int, default=500,
                        help='Batch size for queries')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Number of workers')
    parser.add_argument('--sep_token', type=str, default=', ',
                        help='Separation between dialog rounds')
    parser.add_argument('--cache_corpus', type=str, default='',
                        help='Cache path for saving indexed corpus')
    parser.add_argument('--queries_path', type=str, 
                        default='ChatIR/dialogues/VisDial_v1.0_queries_val.json',
                        help='Path to queries JSON file')
    parser.add_argument('--corpus_path', type=str,
                        default='ChatIR/ChatIR_Protocol/Search_Space_val_50k.json',
                        help='Path to corpus JSON file')
    parser.add_argument('--img_root', type=str, default='/data/mscoco',
                        help='Root directory for images')
    parser.add_argument('--baseline', type=str, default='blip-zero-shot',
                        help='Baseline model to use (blip-zero-shot or clip-zero-shot)')
    parser.add_argument('--external_text_root', type=str, default=None,
                        help='Path to external text directory (if using)')
    parser.add_argument('--hits_at', type=int, default=10,
                        help='Hits@K value for evaluation')

    
    args = parser.parse_args()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    # Create config dictionary from parsed arguments
    cfg = {
        'corpus_bs': args.corpus_bs,
        'queries_bs': args.queries_bs,
        'num_workers': args.num_workers,
        'sep_token': args.sep_token,
        'cache_corpus': args.cache_corpus,
        'queries_path': args.queries_path,
        'corpus_path': args.corpus_path,
        'img_root': args.img_root,
        'device': 'cuda:0',
    }

    with torch.no_grad():
        baseline = args.baseline # 'blip-zero-shot' or 'clip-zero-shot'
        hits_at = args.hits_at
        print(f"Using baseline: {baseline}")
        print(f"Using hits_at: {hits_at}")
        
        # Set external text directory (if using)
        # external_text_root = './results_genir/self_dialogue_caption_output_gemma_3_12b_it/captions' # 77.12%-88.89% but only ~150 samples
        # external_text_root = './results_genir/self_dialogue_caption_output_gemma_3_4b_it/captions' # 67.43%-90.23% but ~500 samples
        # external_text_root = './results_genir/caption_refinement_output_12b_77t_concise/captions'  # 83.33% 42 samples, DL=0 only
        external_text_root = args.external_text_root
        
        if baseline == 'clip-zero-shot':
            if not cfg['cache_corpus']:
                cfg['cache_corpus'] = "temp/corpus_clip_16.pth"
            dialog_encoder, image_embedder = CLIP_ZERO_SHOT_BASELINE()
        else:
            if not cfg['cache_corpus']:
                cfg['cache_corpus'] = "temp/corpus_blip_small.pth"
            dialog_encoder, image_embedder = BLIP_BASELINE()
        
        # Create evaluator
        evaluator = ImprovedChatIREval(cfg, dialog_encoder, image_embedder)
        evaluator.index_corpus()
        
        # Run evaluation with accumulative recall calculation
        # Set find_consistent_samples=True to ensure fair comparison
        evaluator.run(
            hits_at=hits_at,
            find_consistent_samples=True,
            external_text_root=external_text_root
        )