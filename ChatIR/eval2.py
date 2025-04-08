import torch
import tqdm
import os.path
import json
import torch.nn.functional as F
from baselines import ImageEmbedder, CLIP_ZERO_SHOT_BASELINE, BLIP_BASELINE


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
        # Concatenate the partial dialog information with a predefined seperator.
        text = self.cfg['sep_token'].join(self.queries[i]['dialog'][:self.dialog_length + 1])
        return {'text': text, 'target_path': target_path}


class ChatIREval:
    """ This class run the main evaluation process.
    """
    def __init__(self, cfg, dialog_encoder, image_embedder: ImageEmbedder):
        self.dialog_encoder = dialog_encoder  # In paper was referred as "Image Retriever"
        self.image_embedder = image_embedder  # Image encoder

        self.cfg = cfg
        self.corpus = None
        self.corpus_dataset = Corpus(self.cfg['corpus_path'], self.image_embedder.processor)
        
    def _get_recalls_image(self, dataloader, dialog_length):
        # Set dialog length
        dataloader.dataset.dialog_length = dialog_length
        recalls = []
        for batch in tqdm.tqdm(dataloader):
            target_ids = torch.tensor([self.corpus_dataset.path_to_index(p) for p in batch['target_path']]).unsqueeze(1).to(self.cfg['device'])
            
            # Instead of using language embedding use the fake image embedding
            pred_vec = F.normalize(self.dialog_encoder(batch['text']), dim=-1)
            # batch recalls
            scores = pred_vec @ self.corpus[1].T
            arg_ranks = torch.argsort(scores, descending=True, dim=1).long()
            target_recall = ((arg_ranks - target_ids) == 0).nonzero()[:, 1]
            recalls.append(target_recall)

        return torch.cat(recalls)
    
    def _gemma3_text(self, batch,dialog_length):
        # gemma3_text_root = './caption_to_image_output/captions' # 84.25%
        # gemma3_text_root = './caption_refinement_output/captions' # 83.89%
        # gemma3_text_root = './caption_refinement_output_try2/captions' # 67.97%
        # gemma3_text_root = './caption_refinement_output_try_veryLongCaption/captions' # 60.1%
        # gemma3_text_root = './caption_refinement_output_12b_200t_concise/captions'  # 77.14%
        gemma3_text_root = './caption_refinement_output_12b_77t_concise/captions'  # 81.25%
        
        text_list = []
        valid_indices = []
        
        for i in range(len(batch['text'])):
            target_name = batch['target_path'][i].split('/')[-1].split('.')[0]
            gemma3_text_path = f'{gemma3_text_root}/{target_name}_caption_{dialog_length}.txt'
            
            # Check if file exists before attempting to open it
            if os.path.exists(gemma3_text_path):
                _gemma3_text = []
                with open(gemma3_text_path, 'r') as f:
                    for line in f:
                        _gemma3_text.append(line.strip())
                _gemma3_text = ', '.join(_gemma3_text)
                text_list.append(_gemma3_text)
                valid_indices.append(i)
            else:
                # Skip this case - don't add to text_list or valid_indices
                pass
        
        # Return text list and valid indices for skipping in _get_recalls
        return text_list, valid_indices
    
    def _get_recalls(self, dataloader, dialog_length):
        # Set dialog length
        dataloader.dataset.dialog_length = dialog_length
        recalls = []
        for batch in tqdm.tqdm(dataloader):
            # Get gemma3 text and valid indices
            text_list, valid_indices = self._gemma3_text(batch,dialog_length)
            
            # Skip if there are no valid captions
            if not valid_indices:
                continue
                
            # Filter batch to only include entries with valid captions
            target_paths = [batch['target_path'][i] for i in valid_indices]
            target_ids = torch.tensor([self.corpus_dataset.path_to_index(p) for p in target_paths]).unsqueeze(1).to(self.cfg['device'])
            
            # Process valid text entries
            pred_vec = F.normalize(self.dialog_encoder(text_list), dim=-1)
            
            # Compute scores and ranks
            scores = pred_vec @ self.corpus[1].T
            arg_ranks = torch.argsort(scores, descending=True, dim=1).long()
            target_recall = ((arg_ranks - target_ids) == 0).nonzero()[:, 1]
            recalls.append(target_recall)

        # Handle the case where all batches were skipped
        if not recalls:
            return torch.tensor([], dtype=torch.long)
            
        return torch.cat(recalls)

    def run(self, hits_at=10):
        assert self.corpus, f"Prepare corpus first (self.index_corpus())"
        dataset = Queries(cfg, self.cfg['queries_path'])
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.cfg['queries_bs'],
                                                 shuffle=False,
                                                 num_workers=self.cfg['num_workers'],
                                                 pin_memory=True,
                                                 drop_last=False
                                                 )
        
        # Store dialog recall tensors for each dialog length separately
        dialog_recalls_dict = {}
        
        for dl in range(11):
            print(f"Calculate recalls for each dialogues of length {dl}...")
            dialog_recalls = self._get_recalls(dataloader, dialog_length=dl)
            
            # Store the recalls for this dialog length
            if dialog_recalls.numel() > 0:
                dialog_recalls_dict[dl] = dialog_recalls
            else:
                print(f"Warning: No valid captions found for dialog length {dl}")
        
        if not dialog_recalls_dict:
            print("No valid dialog recalls were found across any dialog lengths.")
            return
            
        # Calculate hits@K for each dialog length separately
        print(f"====== Results for Hits@{hits_at} ====== ")
        for dl in range(11):
            if dl in dialog_recalls_dict:
                # Calculate hits for this specific dialog length
                recalls = dialog_recalls_dict[dl]
                hit_rate = (recalls < hits_at).float().mean().item() * 100
                print(f"\t Dialog Length: {dl}: {round(hit_rate, 2)}%")
            else:
                print(f"\t Dialog Length: {dl}: No valid data")

    def index_corpus(self):
        """ Prepare corpus (image search space)"""
        # self.corpus = torch.arange(50000).to(cfg['device']), torch.randn(50_000, 512).to(cfg['device']).half()
        if self.cfg['cache_corpus'] and os.path.exists(self.cfg['cache_corpus']):
            print(f"<<<<Cached corpus has been loaded: {self.cfg['cache_corpus']} >>>>>")
            print(f"Warning: Make sure this corpus has been indexed with the right image embedder!")
            self.corpus = torch.load(self.cfg['cache_corpus'])
            return
        # return
        dataloader = torch.utils.data.DataLoader(self.corpus_dataset,
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


# We'll modify the approach to calculate metrics directly in the run method
# Instead of using these utility functions which expect a specific tensor structure
def get_simple_hit_rate(recalls, hits_at=10):
    """Simple function to calculate the hit rate"""
    if recalls.numel() == 0:
        return 0.0
    return (recalls < hits_at).float().mean().item() * 100


if __name__ == '__main__':
    cfg = {'corpus_bs': 500,
           'queries_bs': 500,
           'num_workers': 64,
           'sep_token': ', ',  # Separation between dialog rounds
           'cache_corpus': '',  # Cache path for saving indexed corpus
           'queries_path': 'ChatIR/dialogues/VisDial_v1.0_queries_val.json',
           'corpus_path': 'ChatIR/ChatIR_Protocol/Search_Space_val_50k.json',
           'device': 'cuda:0',  # 'cpu'
           }

    with torch.no_grad():
        # ==== For CLIP zero-shot baseline: =====
        baseline = 'clip-zero-shot'
        # === for BLIP dialog-trained baseline ===
        baseline = 'blip-dialog-encoder'

        if baseline == 'clip-zero-shot':
            cfg['cache_corpus'] = "temp/corpus_clip_16.pth"
            dialog_encoder, image_embedder = CLIP_ZERO_SHOT_BASELINE()
        else:
            cfg['cache_corpus'] = "temp/corpus_blip_small.pth"
            dialog_encoder, image_embedder = BLIP_BASELINE()
        # ---------
        evaluator = ChatIREval(cfg, dialog_encoder, image_embedder)
        evaluator.index_corpus()
        evaluator.run(hits_at=10)  # Hit@10 as in the paper