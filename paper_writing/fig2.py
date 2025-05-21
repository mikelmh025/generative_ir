import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

def create_genIR_multiplot(csv_file, output_file=None):
    """
    Creates a visualization of GenIR results with three datasets in one row.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing GenIR results for multiple datasets
    output_file : str, optional
        Path to save the output figure
    """
    # Read the CSV file using pandas for more robust handling
    df = pd.read_csv(csv_file, header=None)
    
    # Extract dataset names (first row)
    dataset_names = [df.iloc[0, 0], df.iloc[0, 5], df.iloc[0, 11]]
    
    # Extract data for each dataset
    datasets = []
    
    # Dataset 1 (FFHQ): columns 0-3
    dataset1 = {
        'name': dataset_names[0],
        'x_values': df.iloc[2:, 0].astype(float).tolist(),
        'series': [
            {'name': df.iloc[1, 1], 'condition': 'Text only', 'values': df.iloc[2:, 1].astype(float).tolist()},
            {'name': df.iloc[1, 2], 'condition': 'Text with prediction Feed back', 'values': df.iloc[2:, 2].astype(float).tolist()},
            {'name': df.iloc[1, 3], 'condition': 'Text - Fake Image', 'values': df.iloc[2:, 3].astype(float).tolist()}
        ]
    }
    datasets.append(dataset1)
    
    # Dataset 2 (Flicker 30k): columns 5-9
    dataset2 = {
        'name': dataset_names[1],
        'x_values': df.iloc[2:, 5].astype(float).tolist(),
        'series': [
            {'name': df.iloc[1, 6], 'condition': 'Text only', 'values': df.iloc[2:, 6].astype(float).tolist()},
            {'name': df.iloc[1, 7], 'condition': 'Text only', 'values': df.iloc[2:, 7].astype(float).tolist()},
            {'name': df.iloc[1, 8], 'condition': 'Text with prediction Feed back', 'values': df.iloc[2:, 8].astype(float).tolist()},
            {'name': df.iloc[1, 9], 'condition': 'Text - Fake Image', 'values': df.iloc[2:, 9].astype(float).tolist()}
        ]
    }
    datasets.append(dataset2)
    
    # Dataset 3 (ClothingADC): columns 11-14
    dataset3 = {
        'name': dataset_names[2],
        'x_values': df.iloc[2:, 11].astype(float).tolist(),
        'series': [
            {'name': df.iloc[1, 12], 'condition': 'Text only', 'values': df.iloc[2:, 12].astype(float).tolist()},
            {'name': df.iloc[1, 13], 'condition': 'Text with prediction Feed back', 'values': df.iloc[2:, 13].astype(float).tolist()},
            {'name': df.iloc[1, 14], 'condition': 'Text - Fake Image', 'values': df.iloc[2:, 14].astype(float).tolist()}
        ]
    }
    datasets.append(dataset3)
    
    # Define groups with their performance hierarchy, marker styles, and line styles
    # Performance: "Text - Fake Image" > "Text with prediction Feed back" > "Text only"
    groups = [
        {
            "name": "Text - Fake Image",
            "models": {
                "default": {"color": "orange"},
                "Gemma3-4b-with-Fake-Image-Feedback": {"color": "orange"}
            },
            "zorder": 30,  # Highest performance group - draw on top
            "marker": "o",  # Circle markers for this group
            "markersize": 8,  # Marker size
            "linestyle": "-",  # Solid line
            "linewidth": 2.0  # Line width
        },
        {
            "name": "Text with prediction Feed back",
            "models": {
                "default": {"color": "blue"},
                "Gemma3-4b-with-Prediction-Feedback": {"color": "blue"}
            },
            "zorder": 20,  # Middle performance group
            "marker": "s",  # Square markers for this group
            "markersize": 8,  # Marker size
            "linestyle": "--",  # Dashed line
            "linewidth": 2.0  # Line width
        },
        {
            "name": "Text only",
            "models": {
                "default": {"color": "green"},
                "ChatIR-Text-Only": {"color": "green"},
                "Gemma3-4b-Text-Only": {"color": "red"}
            },
            "zorder": 10,  # Lowest performance group - draw at bottom
            "marker": "^",  # Triangle markers for this group
            "markersize": 8,  # Marker size
            "linestyle": ":",  # Dotted line
            "linewidth": 2.0  # Line width
        }
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)  # Removed sharey=True to allow different y-axis ranges
    
    # Set a clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # For each dataset/subplot
    for i, (dataset, ax) in enumerate(zip(datasets, axes)):
        all_values = []
        
        # Plot each group in the hierarchy order
        for group in groups:
            for series in dataset['series']:
                if series['condition'] == group['name']:
                    # Determine color: use model-specific color if available, otherwise use condition default
                    color = group['models']['default']['color']
                    if series['name'] in group['models']:
                        color = group['models'][series['name']]['color']
                    
                    # Special cases for specific models
                    if series['name'] == "Gemma3-4b-Text-Only":
                        color = "red"
                    elif series['name'] == "Gemma3-4b-with-Prediction-Feedback":
                        color = "blue"
                    elif series['name'] == "ChatIR-Text-Only":
                        color = "green"
                    
                    ax.plot(
                        dataset['x_values'], 
                        series['values'],
                        label=f"{series['name']}",
                        color=color,
                        linewidth=group['linewidth'],
                        marker=group['marker'],
                        markevery=1,  # Add marker at every data point
                        markersize=group['markersize'],
                        linestyle=group['linestyle'],
                        zorder=group['zorder']
                    )
                    
                    # Collect all values for y-axis range calculation
                    all_values.extend(series['values'])
        
        # Set title and labels
        ax.set_title(dataset['name'], fontsize=14, weight='bold')
        ax.set_xlabel('Dialog Length', fontsize=12, weight='bold')
        if i == 0:  # Only add y-axis label to the leftmost subplot
            ax.set_ylabel('Hits@10 (in %)', fontsize=12, weight='bold')
        
        # Calculate appropriate y-axis range with 5% padding
        if all_values:
            y_min = max(0, min(all_values) * 0.95)  # 5% below minimum, but not below 0
            y_max = min(100, max(all_values) * 1.05)  # 5% above maximum, but not above 100
            
            # Round to nicer values
            y_min = np.floor(y_min / 5) * 5  # Round down to nearest 5
            y_max = np.ceil(y_max / 5) * 5   # Round up to nearest 5
            
            # Set axis limits
            ax.set_ylim(y_min, y_max)
        
        # Set x-axis limit
        ax.set_xlim(0, 10)
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add a grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create a better organized legend with the updated model names
    legend_elements = []
    
    # Text - Fake Image category
    legend_elements.append(Line2D([0], [0], color='orange', marker='o', linestyle='-', 
                              markersize=8, label='Gemma3-4b-with-Synthetic-Image-Feedback'))
    
    # Text with prediction Feed back category
    legend_elements.append(Line2D([0], [0], color='blue', marker='s', linestyle='--', 
                              markersize=8, label='Gemma3-4b-with-Prediction-Feedback'))
    
    # Text only category
    legend_elements.append(Line2D([0], [0], color='red', marker='^', linestyle=':', 
                              markersize=8, label='Gemma3-4b-Text-Only'))
    legend_elements.append(Line2D([0], [0], color='green', marker='^', linestyle=':', 
                              markersize=8, label='ChatGPT-Text-Only'))
    
    
    # Create the legend - MODIFIED POSITION HERE
    legend = fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),  # Changed from -0.15 to -0.05 to move legend closer to figure
        ncol=2,
        frameon=True,
        facecolor='white',
        framealpha=0.9,
        fontsize=12
    )
    
    # Style the category headers (white space lines) in bold
    for text in legend.get_texts():
        if text.get_text().endswith(':'):
            text.set_weight('bold')
            text.set_style('italic')
    
    # Adjust layout to make room for the legend - MODIFIED SPACING HERE
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)  # Changed from 0.28 to 0.18 to reduce bottom margin
    
    # Save the figure if needed
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {output_file}")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    create_genIR_multiplot('paper_writing/GenIR results - Figure 2.csv', 'paper_writing/plot2.png')