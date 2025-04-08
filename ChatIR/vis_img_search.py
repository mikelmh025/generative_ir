import torch
from eval_img import GenIRQueries
def visualize_fake_image_retrieval(evaluator, dataloader, dialog_length, num_samples=5, k=10):
    """
    Visualize the fake image retrieval process to understand why it's not working well.
    
    Args:
        evaluator: The GenIREval instance
        dataloader: DataLoader for queries
        dialog_length: Which dialog length to evaluate
        num_samples: Number of samples to visualize
        k: Number of top results to display
    
    Returns:
        Visualization data for each sample
    """
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    import numpy as np
    import os
    from PIL import Image
    import torchvision.transforms as T
    from torchvision.utils import make_grid
    
    # Set dialog length
    dataloader.dataset.dialog_length = dialog_length
    
    # To convert tensor images back to PIL for visualization
    to_pil = T.ToPILImage()
    
    # For denormalizing images
    denormalize = T.Compose([
        T.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ),
        T.Lambda(lambda x: torch.clamp(x, 0, 1))
    ])
    
    # List to store visualization data
    visualization_data = []
    
    # Counter for samples
    sample_count = 0
    
    # Process batches until we get enough samples
    for batch in dataloader:
        if sample_count >= num_samples:
            break
            
        batch_target_paths = batch['target_path']
        batch_prompts = batch['text']
        
        for i, (path, prompt) in enumerate(zip(batch_target_paths, batch_prompts)):
            if sample_count >= num_samples:
                break
                
            index = evaluator.corpus_dataset.path_to_index(path)
            
            # Extract the original image filename without the path
            orig_filename = os.path.basename(path)
            if orig_filename.endswith('.jpg'):
                orig_filename = orig_filename[:-4]
            
            # Check if the fake image exists before trying to load it
            fake_img_path = os.path.join(evaluator.cfg['fake_images_dir'], f"{orig_filename}_{dialog_length}.jpg")
            
            if not os.path.exists(fake_img_path):
                continue
                
            # Load original target image and fake image
            target_img = evaluator.corpus_dataset.preprocessor(path).unsqueeze(0).to(evaluator.cfg['device'])
            fake_img = evaluator.corpus_dataset.preprocessor(fake_img_path).unsqueeze(0).to(evaluator.cfg['device'])
            
            # Store preprocessed versions for visualization
            target_img_viz = target_img.clone()
            fake_img_viz = fake_img.clone()
            
            # Get embeddings
            target_embedding = F.normalize(evaluator.image_embedder.model(target_img), dim=-1)
            fake_embedding = F.normalize(evaluator.image_embedder.model(fake_img), dim=-1)
            
            # Calculate similarity between target and fake
            target_fake_similarity = torch.sum(target_embedding * fake_embedding).item()
            
            # Compute scores against the entire corpus
            scores = fake_embedding @ evaluator.corpus[1].T
            
            # Get top k indices and scores
            top_scores, top_indices = torch.topk(scores.squeeze(), k)
            
            # Check if target is in top k
            target_id = torch.tensor([index]).to(evaluator.cfg['device'])
            target_in_top_k = (top_indices == target_id).any().item()
            
            # Find rank of target
            arg_ranks = torch.argsort(scores, descending=True, dim=1).long()
            target_rank = ((arg_ranks - target_id) == 0).nonzero()
            if len(target_rank) > 0:
                target_rank = target_rank[0, 1].item() + 1  # +1 for 1-indexed rank
            else:
                target_rank = "Not found"
            
            # Get top k images
            top_k_images = []
            for idx in top_indices:
                corpus_idx = evaluator.corpus[0][idx].item()
                corpus_path = evaluator.corpus_dataset.corpus[corpus_idx]
                corpus_img = evaluator.corpus_dataset.preprocessor(corpus_path).unsqueeze(0).to(evaluator.cfg['device'])
                # Store a copy for visualization
                corpus_img_viz = corpus_img.clone()
                top_k_images.append(corpus_img_viz)
            
            # Check if target is the actual top result
            is_top_result = top_indices[0].item() == index
            
            # Store visualization data
            sample_data = {
                'prompt': prompt,
                'target_img': target_img_viz.squeeze().cpu(),
                'fake_img': fake_img_viz.squeeze().cpu(),
                'target_fake_similarity': target_fake_similarity,
                'target_in_top_k': target_in_top_k,
                'target_rank': target_rank,
                'is_top_result': is_top_result,
                'top_indices': top_indices.cpu(),
                'top_scores': top_scores.cpu(),
                'top_k_images': [img.squeeze().cpu() for img in top_k_images]
            }
            
            visualization_data.append(sample_data)
            sample_count += 1
            print(f"Processed sample {sample_count}/{num_samples}")
    
    # Return the visualization data
    return visualization_data


def save_visualizations(visualization_data, output_dir="visualization_results", k=10):
    """
    Save visualizations to disk with detailed information
    
    Args:
        visualization_data: Data from visualize_fake_image_retrieval
        output_dir: Directory to save visualizations
        k: Number of top results displayed
    """
    import matplotlib.pyplot as plt
    import os
    import torch
    import torchvision.transforms as T
    from torchvision.utils import make_grid
    import numpy as np
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # To convert tensor images to numpy for matplotlib with proper normalization
    def to_numpy(x):
        # First permute from (C,H,W) to (H,W,C)
        x = x.permute(1, 2, 0)
        # Denormalize if needed - assuming images are normalized with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
        x = x * std + mean
        # Clamp to ensure values are in [0, 1] range
        x = torch.clamp(x, 0, 1)
        return x.numpy()
    
    for i, data in enumerate(visualization_data):
        # Create a figure with subplots for visualization
        fig = plt.figure(figsize=(20, 15))
        
        # Calculate needed grid dimensions
        grid_size = 3 + k  # Target + Fake + Grid of top k + text area
        rows = max(3, (grid_size + 2) // 3)  # At least 3 rows, or more if needed
        
                    # Add target image
        ax1 = plt.subplot(rows, 4, 1)
        ax1.imshow(to_numpy(data['target_img']))
        ax1.set_title(f"Target Image (Ground Truth)\nRank: {data['target_rank']}")
        ax1.axis('off')
        
        # Add fake image
        ax2 = plt.subplot(rows, 4, 2)
        ax2.imshow(to_numpy(data['fake_img']))
        ax2.set_title(f"Fake Image (Generated)\nSimilarity to Target: {data['target_fake_similarity']:.4f}")
        ax2.axis('off')
        
        # Add searched image (same as fake but labeled as search query)
        # ax3 = plt.subplot(rows, 4, 3)
        # ax3.imshow(to_numpy(data['fake_img']))
        # ax3.set_title("Searched Image (Query)\nThis is what we're using to search")
        # ax3.axis('off')
        
        # Add text information in the fourth cell
        ax4 = plt.subplot(rows, 4, 4)
        ax4.text(0.1, 0.9, f"Dialog Length: {len(data['prompt'].split(', ')) - 1}", fontsize=10)
        ax4.text(0.1, 0.8, f"Target in Top {k}: {data['target_in_top_k']}", fontsize=10)
        ax4.text(0.1, 0.7, f"Target Rank: {data['target_rank']}", fontsize=10)
        ax4.text(0.1, 0.6, f"Is Top Result: {data['is_top_result']}", fontsize=10)
        ax4.text(0.1, 0.5, f"Prompt:", fontsize=10)
        
        # Wrap text for prompts
        wrapped_prompt = '\n'.join([data['prompt'][i:i+50] for i in range(0, len(data['prompt']), 50)])
        ax4.text(0.1, 0.4, wrapped_prompt, fontsize=8, wrap=True)
        ax4.axis('off')
        
        # Add top k images
        for j in range(k):
            ax = plt.subplot(rows, 4, 5 + j)  # Start from position 5 since we now have 4 items in the first row
            ax.imshow(to_numpy(data['top_k_images'][j]))
            # Check if this is the target image
            is_target = data['top_indices'][j].item() == data['top_indices'][data['top_indices'] == data['top_indices'][j]].item()
            title = f"Rank {j+1}: Score {data['top_scores'][j]:.4f}"
            if is_target:
                title += " (TARGET)"
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        
        # Set the title for the whole figure
        plt.suptitle(f"Sample {i+1}: {'Target found in top ' + str(k) if data['target_in_top_k'] else 'Target not in top ' + str(k)}", 
                     fontsize=16)
        
        # Save the figure
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        plt.savefig(os.path.join(output_dir, f"sample_{i+1}.png"), dpi=150)
        plt.close()
        
        print(f"Saved visualization for sample {i+1}")


# def analyze_fake_image_retrieval(evaluator, num_samples=50, dialog_lengths=[0, 2, 5, 10], k=10):
#     """
#     Comprehensive analysis of fake image retrieval across multiple dialog lengths
    
#     Args:
#         evaluator: The GenIREval instance
#         num_samples: Number of samples to analyze per dialog length
#         dialog_lengths: List of dialog lengths to analyze
#         k: Number of top results to analyze
        
#     Returns:
#         Dictionary with analysis results
#     """
#     import torch
#     import torch.nn.functional as F
#     import numpy as np
#     import os
#     from collections import defaultdict
    
#     # Create dataset and dataloader
#     dataset = GenIRQueries(evaluator.cfg, evaluator.cfg['queries_path'])
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=evaluator.cfg['queries_bs'],
#         shuffle=True,  # Shuffle to get diverse samples
#         num_workers=evaluator.cfg['num_workers'],
#         pin_memory=True,
#         drop_last=False
#     )
    
#     analysis_results = {}
    
#     for dl in dialog_lengths:
#         print(f"\n===== Analyzing Dialog Length {dl} =====")
        
#         # Set dialog length
#         dataloader.dataset.dialog_length = dl
        
#         # Track statistics
#         stats = {
#             'target_fake_similarities': [],
#             'target_ranks': [],
#             'hit_at_k': 0,
#             'median_rank': 0,
#             'mean_rank': 0,
#             'similarity_to_top1': [],
#             'similarity_to_target': [],
#             'success_prompt_examples': [],
#             'failure_prompt_examples': []
#         }
        
#         # Process samples
#         processed_count = 0
#         batch_idx = 0
        
#         while processed_count < num_samples and batch_idx < len(dataloader):
#             for batch in dataloader:
#                 if processed_count >= num_samples:
#                     break
                    
#                 batch_target_paths = batch['target_path']
#                 batch_prompts = batch['text']
                
#                 for i, (path, prompt) in enumerate(zip(batch_target_paths, batch_prompts)):
#                     if processed_count >= num_samples:
#                         break
                        
#                     index = evaluator.corpus_dataset.path_to_index(path)
                    
#                     # Extract the original image filename without the path
#                     orig_filename = os.path.basename(path)
#                     if orig_filename.endswith('.jpg'):
#                         orig_filename = orig_filename[:-4]
                    
#                     # Check if the fake image exists before trying to load it
#                     fake_img_path = os.path.join(evaluator.cfg['fake_images_dir'], f"{orig_filename}_{dl}.jpg")
                    
#                     if not os.path.exists(fake_img_path):
#                         continue
                        
#                     # Load original target image and fake image
#                     target_img = evaluator.corpus_dataset.preprocessor(path).unsqueeze(0).to(evaluator.cfg['device'])
#                     fake_img = evaluator.corpus_dataset.preprocessor(fake_img_path).unsqueeze(0).to(evaluator.cfg['device'])
                    
#                     # Get embeddings
#                     target_embedding = F.normalize(evaluator.image_embedder.model(target_img), dim=-1)
#                     fake_embedding = F.normalize(evaluator.image_embedder.model(fake_img), dim=-1)
                    
#                     # Calculate similarity between target and fake
#                     target_fake_similarity = torch.sum(target_embedding * fake_embedding).item()
#                     stats['target_fake_similarities'].append(target_fake_similarity)
                    
#                     # Compute scores against the entire corpus
#                     scores = fake_embedding @ evaluator.corpus[1].T
                    
#                     # Get top k indices and scores
#                     top_scores, top_indices = torch.topk(scores.squeeze(), k)
                    
#                     # Check if target is in top k
#                     target_id = torch.tensor([index]).to(evaluator.cfg['device'])
#                     target_in_top_k = (top_indices == target_id).any().item()
                    
#                     if target_in_top_k:
#                         stats['hit_at_k'] += 1
#                         stats['success_prompt_examples'].append(prompt)
#                     else:
#                         stats['failure_prompt_examples'].append(prompt)
                    
#                     # Find rank of target
#                     arg_ranks = torch.argsort(scores, descending=True, dim=1).long()
#                     target_rank_tensor = ((arg_ranks - target_id) == 0).nonzero()
#                     if len(target_rank_tensor) > 0:
#                         target_rank = target_rank_tensor[0, 1].item() + 1  # +1 for 1-indexed rank
#                         stats['target_ranks'].append(target_rank)
                    
#                     # Record similarity to top-1 result
#                     top1_idx = top_indices[0].item()
#                     corpus_idx = evaluator.corpus[0][top1_idx].item()
#                     top1_path = evaluator.corpus_dataset.corpus[corpus_idx]
#                     top1_img = evaluator.corpus_dataset.preprocessor(top1_path).unsqueeze(0).to(evaluator.cfg['device'])
#                     top1_embedding = F.normalize(evaluator.image_embedder.model(top1_img), dim=-1)
                    
#                     # Calculate similarities
#                     fake_to_top1_similarity = torch.sum(fake_embedding * top1_embedding).item()
#                     stats['similarity_to_top1'].append(fake_to_top1_similarity)
                    
#                     # If different from target, calculate target to top1 similarity
#                     if top1_idx != index:
#                         target_to_top1_similarity = torch.sum(target_embedding * top1_embedding).item()
#                         stats['similarity_to_target'].append(target_to_top1_similarity)
                    
#                     processed_count += 1
#                     print(f"Processed {processed_count}/{num_samples} for dialog length {dl}")
                
#                 batch_idx += 1
        
#         # Compute summary statistics
#         if stats['target_ranks']:
#             stats['median_rank'] = np.median(stats['target_ranks'])
#             stats['mean_rank'] = np.mean(stats['target_ranks'])
        
#         if processed_count > 0:
#             stats['hit_at_k_percentage'] = (stats['hit_at_k'] / processed_count) * 100
#         else:
#             stats['hit_at_k_percentage'] = 0
            
#         # Compute average similarities
#         stats['avg_target_fake_similarity'] = np.mean(stats['target_fake_similarities'])
#         stats['avg_similarity_to_top1'] = np.mean(stats['similarity_to_top1'])
#         if stats['similarity_to_target']:
#             stats['avg_similarity_to_target'] = np.mean(stats['similarity_to_target'])
        
#         # Store only a few examples
#         stats['success_prompt_examples'] = stats['success_prompt_examples'][:5]
#         stats['failure_prompt_examples'] = stats['failure_prompt_examples'][:5]
        
#         # Print summary
#         print(f"\nSummary for Dialog Length {dl}:")
#         print(f"Processed samples: {processed_count}")
#         print(f"Hit@{k}: {stats['hit_at_k']}/{processed_count} ({stats['hit_at_k_percentage']:.2f}%)")
#         if stats['target_ranks']:
#             print(f"Median Rank: {stats['median_rank']}")
#             print(f"Mean Rank: {stats['mean_rank']:.2f}")
#         print(f"Avg Target-Fake Similarity: {stats['avg_target_fake_similarity']:.4f}")
#         print(f"Avg Fake-Top1 Similarity: {stats['avg_similarity_to_top1']:.4f}")
#         if 'avg_similarity_to_target' in stats:
#             print(f"Avg Target-Top1 Similarity: {stats['avg_similarity_to_target']:.4f}")
        
#         analysis_results[dl] = stats
    
#     return analysis_results


# def execute_visualization_pipeline(evaluator, output_dir="visualization_results"):
#     """
#     Execute the full visualization pipeline
    
#     Args:
#         evaluator: The GenIREval instance
#         output_dir: Directory to save results
#     """
#     import os
#     import json
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Create dataset and dataloader for visualization
#     dataset = GenIRQueries(evaluator.cfg, evaluator.cfg['queries_path'])
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=evaluator.cfg['queries_bs'],
#         shuffle=True,  # Use shuffle to get diverse examples
#         num_workers=evaluator.cfg['num_workers'],
#         pin_memory=True,
#         drop_last=False
#     )
    
#     # 1. Generate visualizations for a few examples at different dialog lengths
#     dialog_lengths = [0, 2, 5, 10]
#     for dl in dialog_lengths:
#         dl_dir = os.path.join(output_dir, f"dialog_length_{dl}")
#         os.makedirs(dl_dir, exist_ok=True)
        
#         print(f"\nGenerating visualizations for dialog length {dl}...")
#         vis_data = visualize_fake_image_retrieval(evaluator, dataloader, dl, num_samples=5)
        
#         if vis_data:
#             save_visualizations(vis_data, output_dir=dl_dir)
#         else:
#             print(f"No valid samples found for dialog length {dl}")
    
#     # 2. Run comprehensive analysis
#     print("\nRunning comprehensive analysis...")
#     analysis_results = analyze_fake_image_retrieval(evaluator, num_samples=50, dialog_lengths=dialog_lengths)
    
#     # 3. Generate summary visualizations
#     print("\nGenerating summary visualizations...")
    
#     # Plot target-fake similarity across dialog lengths
#     plt.figure(figsize=(10, 6))
#     x = []
#     y = []
#     for dl in dialog_lengths:
#         if dl in analysis_results and analysis_results[dl]['target_fake_similarities']:
#             x.extend([dl] * len(analysis_results[dl]['target_fake_similarities']))
#             y.extend(analysis_results[dl]['target_fake_similarities'])
    
#     if x and y:
#         plt.scatter(x, y, alpha=0.5)
#         for dl in dialog_lengths:
#             if dl in analysis_results and analysis_results[dl]['target_fake_similarities']:
#                 avg = analysis_results[dl]['avg_target_fake_similarity']
#                 plt.scatter(dl, avg, color='red', s=100, zorder=5)
#                 plt.text(dl, avg+0.02, f"{avg:.3f}", ha='center')
        
#         plt.xlabel("Dialog Length")
#         plt.ylabel("Target-Fake Similarity")
#         plt.title("Target-Fake Similarity Across Dialog Lengths")
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.savefig(os.path.join(output_dir, "target_fake_similarity.png"), dpi=150)
    
#     # Plot Hit@k across dialog lengths
#     plt.figure(figsize=(10, 6))
#     hit_rates = [analysis_results[dl]['hit_at_k_percentage'] if dl in analysis_results else 0 for dl in dialog_lengths]
#     plt.bar(dialog_lengths, hit_rates)
#     for i, v in enumerate(hit_rates):
#         plt.text(dialog_lengths[i], v+1, f"{v:.1f}%", ha='center')
#     plt.xlabel("Dialog Length")
#     plt.ylabel("Hit@10 (%)")
#     plt.title("Hit@10 Rate Across Dialog Lengths")
#     plt.grid(True, linestyle='--', alpha=0.7, axis='y')
#     plt.savefig(os.path.join(output_dir, "hit_rate.png"), dpi=150)
    
#     # Save analysis results as JSON
#     with open(os.path.join(output_dir, "analysis_results.json"), 'w') as f:
#         # Convert numpy values to Python native types for JSON serialization
#         json_safe_results = {}
#         for dl, stats in analysis_results.items():
#             json_safe_results[str(dl)] = {}
#             for k, v in stats.items():
#                 if isinstance(v, (np.ndarray, list)):
#                     if k in ['target_fake_similarities', 'target_ranks', 'similarity_to_top1', 'similarity_to_target']:
#                         json_safe_results[str(dl)][k] = [float(x) for x in v]
#                     else:
#                         json_safe_results[str(dl)][k] = v
#                 elif isinstance(v, (np.integer, np.floating)):
#                     json_safe_results[str(dl)][k] = float(v)
#                 else:
#                     json_safe_results[str(dl)][k] = v
        
#         json.dump(json_safe_results, f, indent=2)
    
#     print(f"\nVisualization pipeline complete. Results saved to {output_dir}")


def execute_visualization_pipeline(evaluator, output_dir="visualization_results"):
    """
    Execute the full visualization pipeline
    
    Args:
        evaluator: The GenIREval instance
        output_dir: Directory to save results
    """
    import os
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset and dataloader for visualization
    dataset = GenIRQueries(evaluator.cfg, evaluator.cfg['queries_path'])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=evaluator.cfg['queries_bs'],
        shuffle=True,  # Use shuffle to get diverse examples
        num_workers=evaluator.cfg['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    # 1. Generate visualizations for a few examples at different dialog lengths
    dialog_lengths = [0, 2, 5, 10]
    for dl in dialog_lengths:
        dl_dir = os.path.join(output_dir, f"dialog_length_{dl}")
        os.makedirs(dl_dir, exist_ok=True)
        
        print(f"\nGenerating visualizations for dialog length {dl}...")
        vis_data = visualize_fake_image_retrieval(evaluator, dataloader, dl, num_samples=5)
        
        if vis_data:
            save_visualizations(vis_data, output_dir=dl_dir)
        else:
            print(f"No valid samples found for dialog length {dl}")
    
    # 2. Run comprehensive analysis
    print("\nRunning comprehensive analysis...")
    analysis_results = analyze_fake_image_retrieval(evaluator, num_samples=50, dialog_lengths=dialog_lengths)
    
    # 3. Generate summary visualizations
    print("\nGenerating summary visualizations...")
    
    # Plot target-fake similarity across dialog lengths
    plt.figure(figsize=(10, 6))
    x = []
    y = []
    for dl in dialog_lengths:
        if dl in analysis_results and analysis_results[dl]['target_fake_similarities']:
            x.extend([dl] * len(analysis_results[dl]['target_fake_similarities']))
            y.extend(analysis_results[dl]['target_fake_similarities'])
    
    if x and y:
        plt.scatter(x, y, alpha=0.5)
        for dl in dialog_lengths:
            if dl in analysis_results and analysis_results[dl]['target_fake_similarities']:
                avg = analysis_results[dl]['avg_target_fake_similarity']
                plt.scatter(dl, avg, color='red', s=100, zorder=5)
                plt.text(dl, avg+0.02, f"{avg:.3f}", ha='center')
        
        plt.xlabel("Dialog Length")
        plt.ylabel("Target-Fake Similarity")
        plt.title("Target-Fake Similarity Across Dialog Lengths")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, "target_fake_similarity.png"), dpi=150)
    
    # Plot Hit@k across dialog lengths
    plt.figure(figsize=(10, 6))
    hit_rates = [analysis_results[dl]['hit_at_k_percentage'] if dl in analysis_results else 0 for dl in dialog_lengths]
    plt.bar(dialog_lengths, hit_rates)
    for i, v in enumerate(hit_rates):
        plt.text(dialog_lengths[i], v+1, f"{v:.1f}%", ha='center')
    plt.xlabel("Dialog Length")
    plt.ylabel("Hit@10 (%)")
    plt.title("Hit@10 Rate Across Dialog Lengths")
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.savefig(os.path.join(output_dir, "hit_rate.png"), dpi=150)
    
    # NEW: Plot rank statistics across dialog lengths
    plt.figure(figsize=(12, 8))
    
    # Create data for plotting
    means = []
    medians = []
    std_devs = []
    valid_dl = []
    
    for dl in dialog_lengths:
        if dl in analysis_results and analysis_results[dl]['target_ranks']:
            ranks = analysis_results[dl]['target_ranks']
            means.append(analysis_results[dl]['mean_rank'])
            medians.append(analysis_results[dl]['median_rank'])
            std_devs.append(np.std(ranks))
            valid_dl.append(dl)
    
    if valid_dl:
        # Plot mean and median ranks
        x = np.arange(len(valid_dl))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Bar chart for mean and median
        bar1 = ax1.bar(x - width/2, means, width, label='Mean Rank', color='skyblue')
        bar2 = ax1.bar(x + width/2, medians, width, label='Median Rank', color='lightgreen')
        
        # Add error bars for standard deviation
        ax1.errorbar(x - width/2, means, yerr=std_devs, fmt='none', ecolor='blue', capsize=5)
        
        # Add value labels on the bars
        for i, v in enumerate(means):
            ax1.text(i - width/2, v + 0.1, f"{v:.1f}", ha='center', fontsize=9)
        for i, v in enumerate(medians):
            ax1.text(i + width/2, v + 0.1, f"{v:.1f}", ha='center', fontsize=9)
        
        # Add standard deviation values as text
        for i, std in enumerate(std_devs):
            ax1.text(i - width/2, means[i] + std + 1, f"σ={std:.1f}", ha='center', fontsize=8, color='blue')
        
        ax1.set_xlabel('Dialog Length')
        ax1.set_ylabel('Rank')
        ax1.set_title('Mean and Median Rank Across Dialog Lengths')
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_dl)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Adjust y-axis to start from 0
        ax1.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rank_statistics.png"), dpi=150)
        
        # NEW: Create an additional visualization with boxplots for rank distribution
        plt.figure(figsize=(12, 6))
        rank_data = [analysis_results[dl]['target_ranks'] for dl in valid_dl]
        
        bp = plt.boxplot(rank_data, patch_artist=True)
        
        # Set colors
        for box in bp['boxes']:
            box.set(facecolor='lightblue')
            
        # Add scatter points to see actual distribution
        for i, ranks in enumerate(rank_data):
            # Add jitter to points for better visualization
            x = np.random.normal(i+1, 0.1, size=len(ranks))
            plt.scatter(x, ranks, alpha=0.4, color='blue', s=15)
        
        plt.xlabel('Dialog Length')
        plt.ylabel('Rank')
        plt.title('Rank Distribution Across Dialog Lengths')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.xticks(range(1, len(valid_dl) + 1), valid_dl)
        
        # Add mean values as text
        for i, mean_val in enumerate(means):
            plt.text(i + 1, max(rank_data[i]) + 5, f"Mean: {mean_val:.1f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rank_distribution.png"), dpi=150)
    
    # Save analysis results as JSON
    with open(os.path.join(output_dir, "analysis_results.json"), 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        json_safe_results = {}
        for dl, stats in analysis_results.items():
            json_safe_results[str(dl)] = {}
            for k, v in stats.items():
                if isinstance(v, (np.ndarray, list)):
                    if k in ['target_fake_similarities', 'target_ranks', 'similarity_to_top1', 'similarity_to_target']:
                        json_safe_results[str(dl)][k] = [float(x) for x in v]
                    else:
                        json_safe_results[str(dl)][k] = v
                elif isinstance(v, (np.integer, np.floating)):
                    json_safe_results[str(dl)][k] = float(v)
                else:
                    json_safe_results[str(dl)][k] = v
        
        json.dump(json_safe_results, f, indent=2)
    
    # NEW: Create a summary table of statistics
    plt.figure(figsize=(12, 6))
    plt.axis('tight')
    plt.axis('off')
    
    valid_dl_data = [(dl, analysis_results[dl]) for dl in dialog_lengths if dl in analysis_results]
    
    if valid_dl_data:
        table_data = []
        
        # Headers
        headers = ['Dialog Length', 'Hit@10 (%)', 'Mean Rank', 'Median Rank', 'Std Dev', 'Samples']
        
        # Prepare data
        for dl, stats in valid_dl_data:
            std_dev = np.std(stats['target_ranks']) if stats['target_ranks'] else 'N/A'
            row = [
                dl,
                f"{stats['hit_at_k_percentage']:.1f}%",
                f"{stats['mean_rank']:.1f}" if 'mean_rank' in stats else 'N/A',
                f"{stats['median_rank']}" if 'median_rank' in stats else 'N/A',
                f"{std_dev:.1f}" if isinstance(std_dev, float) else std_dev,
                len(stats['target_ranks']) if stats['target_ranks'] else 0
            ]
            table_data.append(row)
        
        # Create table
        table = plt.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        
        # Adjust table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Add title
        plt.title('Summary of Rank Statistics', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rank_summary_table.png"), dpi=150)
    
    print(f"\nVisualization pipeline complete. Results saved to {output_dir}")


def analyze_fake_image_retrieval(evaluator, num_samples=50, dialog_lengths=[0, 2, 5, 10], k=10):
    """
    Comprehensive analysis of fake image retrieval across multiple dialog lengths
    
    Args:
        evaluator: The GenIREval instance
        num_samples: Number of samples to analyze per dialog length
        dialog_lengths: List of dialog lengths to analyze
        k: Number of top results to analyze
        
    Returns:
        Dictionary with analysis results
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    import os
    from collections import defaultdict
    
    # Create dataset and dataloader
    dataset = GenIRQueries(evaluator.cfg, evaluator.cfg['queries_path'])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=evaluator.cfg['queries_bs'],
        shuffle=True,  # Shuffle to get diverse samples
        num_workers=evaluator.cfg['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    analysis_results = {}
    
    for dl in dialog_lengths:
        print(f"\n===== Analyzing Dialog Length {dl} =====")
        
        # Set dialog length
        dataloader.dataset.dialog_length = dl
        
        # Track statistics
        stats = {
            'target_fake_similarities': [],
            'target_ranks': [],
            'hit_at_k': 0,
            'median_rank': 0,
            'mean_rank': 0,
            'similarity_to_top1': [],
            'similarity_to_target': [],
            'success_prompt_examples': [],
            'failure_prompt_examples': [],
            'rank_distribution': defaultdict(int)  # NEW: Track distribution of ranks
        }
        
        # Process samples
        processed_count = 0
        batch_idx = 0
        
        while processed_count < num_samples and batch_idx < len(dataloader):
            for batch in dataloader:
                if processed_count >= num_samples:
                    break
                    
                batch_target_paths = batch['target_path']
                batch_prompts = batch['text']
                
                for i, (path, prompt) in enumerate(zip(batch_target_paths, batch_prompts)):
                    if processed_count >= num_samples:
                        break
                        
                    index = evaluator.corpus_dataset.path_to_index(path)
                    
                    # Extract the original image filename without the path
                    orig_filename = os.path.basename(path)
                    if orig_filename.endswith('.jpg'):
                        orig_filename = orig_filename[:-4]
                    
                    # Check if the fake image exists before trying to load it
                    fake_img_path = os.path.join(evaluator.cfg['fake_images_dir'], f"{orig_filename}_{dl}.jpg")
                    
                    if not os.path.exists(fake_img_path):
                        continue
                        
                    # Load original target image and fake image
                    target_img = evaluator.corpus_dataset.preprocessor(path).unsqueeze(0).to(evaluator.cfg['device'])
                    fake_img = evaluator.corpus_dataset.preprocessor(fake_img_path).unsqueeze(0).to(evaluator.cfg['device'])
                    
                    # Get embeddings
                    target_embedding = F.normalize(evaluator.image_embedder.model(target_img), dim=-1)
                    fake_embedding = F.normalize(evaluator.image_embedder.model(fake_img), dim=-1)
                    
                    # Calculate similarity between target and fake
                    target_fake_similarity = torch.sum(target_embedding * fake_embedding).item()
                    stats['target_fake_similarities'].append(target_fake_similarity)
                    
                    # Compute scores against the entire corpus
                    scores = fake_embedding @ evaluator.corpus[1].T
                    
                    # Get top k indices and scores
                    top_scores, top_indices = torch.topk(scores.squeeze(), k)
                    
                    # Check if target is in top k
                    target_id = torch.tensor([index]).to(evaluator.cfg['device'])
                    target_in_top_k = (top_indices == target_id).any().item()
                    
                    if target_in_top_k:
                        stats['hit_at_k'] += 1
                        stats['success_prompt_examples'].append(prompt)
                    else:
                        stats['failure_prompt_examples'].append(prompt)
                    
                    # Find rank of target
                    arg_ranks = torch.argsort(scores, descending=True, dim=1).long()
                    target_rank_tensor = ((arg_ranks - target_id) == 0).nonzero()
                    if len(target_rank_tensor) > 0:
                        target_rank = target_rank_tensor[0, 1].item() + 1  # +1 for 1-indexed rank
                        stats['target_ranks'].append(target_rank)
                        
                        # Update rank distribution
                        stats['rank_distribution'][target_rank] += 1
                    
                    # Record similarity to top-1 result
                    top1_idx = top_indices[0].item()
                    corpus_idx = evaluator.corpus[0][top1_idx].item()
                    top1_path = evaluator.corpus_dataset.corpus[corpus_idx]
                    top1_img = evaluator.corpus_dataset.preprocessor(top1_path).unsqueeze(0).to(evaluator.cfg['device'])
                    top1_embedding = F.normalize(evaluator.image_embedder.model(top1_img), dim=-1)
                    
                    # Calculate similarities
                    fake_to_top1_similarity = torch.sum(fake_embedding * top1_embedding).item()
                    stats['similarity_to_top1'].append(fake_to_top1_similarity)
                    
                    # If different from target, calculate target to top1 similarity
                    if top1_idx != index:
                        target_to_top1_similarity = torch.sum(target_embedding * top1_embedding).item()
                        stats['similarity_to_target'].append(target_to_top1_similarity)
                    
                    processed_count += 1
                    print(f"Processed {processed_count}/{num_samples} for dialog length {dl}")
                
                batch_idx += 1
        
        # Compute summary statistics
        if stats['target_ranks']:
            stats['median_rank'] = np.median(stats['target_ranks'])
            stats['mean_rank'] = np.mean(stats['target_ranks'])
            stats['rank_std_dev'] = np.std(stats['target_ranks'])  # NEW: Standard deviation
            stats['rank_min'] = np.min(stats['target_ranks'])      # NEW: Minimum rank
            stats['rank_max'] = np.max(stats['target_ranks'])      # NEW: Maximum rank
            stats['rank_quartiles'] = np.percentile(stats['target_ranks'], [25, 75])  # NEW: Quartiles
        
        if processed_count > 0:
            stats['hit_at_k_percentage'] = (stats['hit_at_k'] / processed_count) * 100
        else:
            stats['hit_at_k_percentage'] = 0
            
        # Compute average similarities
        stats['avg_target_fake_similarity'] = np.mean(stats['target_fake_similarities'])
        stats['avg_similarity_to_top1'] = np.mean(stats['similarity_to_top1'])
        if stats['similarity_to_target']:
            stats['avg_similarity_to_target'] = np.mean(stats['similarity_to_target'])
        
        # Store only a few examples
        stats['success_prompt_examples'] = stats['success_prompt_examples'][:5]
        stats['failure_prompt_examples'] = stats['failure_prompt_examples'][:5]
        
        # Print summary
        print(f"\nSummary for Dialog Length {dl}:")
        print(f"Processed samples: {processed_count}")
        print(f"Hit@{k}: {stats['hit_at_k']}/{processed_count} ({stats['hit_at_k_percentage']:.2f}%)")
        if stats['target_ranks']:
            print(f"Median Rank: {stats['median_rank']}")
            print(f"Mean Rank: {stats['mean_rank']:.2f}")
            print(f"Std Dev Rank: {stats['rank_std_dev']:.2f}")  # NEW
            print(f"Min Rank: {stats['rank_min']}")              # NEW
            print(f"Max Rank: {stats['rank_max']}")              # NEW
            print(f"25th/75th Percentiles: {stats['rank_quartiles'][0]:.1f}/{stats['rank_quartiles'][1]:.1f}")  # NEW
        print(f"Avg Target-Fake Similarity: {stats['avg_target_fake_similarity']:.4f}")
        print(f"Avg Fake-Top1 Similarity: {stats['avg_similarity_to_top1']:.4f}")
        if 'avg_similarity_to_target' in stats:
            print(f"Avg Target-Top1 Similarity: {stats['avg_similarity_to_target']:.4f}")
        
        analysis_results[dl] = stats
    
    return analysis_results