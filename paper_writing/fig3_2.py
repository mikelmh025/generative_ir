import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from matplotlib.text import Text

def create_dual_dataset_ablation_plot(csv_file, output_file=None):
    """
    Creates a visualization with two subplots for MSCOCO and FFHQ datasets from the CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing both MSCOCO and FFHQ results
    output_file : str, optional
        Path to save the output figure
    """
    # Read the CSV file - skip the first two rows (which has dataset and group headers)
    raw_data = pd.read_csv(csv_file, header=None)
    
    # Extract MSCOCO data (columns 0-6)
    mscoco_columns = list(range(7))
    mscoco_data = raw_data.iloc[2:, mscoco_columns].copy()
    mscoco_data.columns = raw_data.iloc[2, mscoco_columns]
    mscoco_data = mscoco_data.iloc[1:].reset_index(drop=True)
    
    # Extract FFHQ data (columns 8-14)
    ffhq_columns = list(range(8, 15))
    ffhq_data = raw_data.iloc[2:, ffhq_columns].copy()
    ffhq_data.columns = raw_data.iloc[2, ffhq_columns]
    ffhq_data = ffhq_data.iloc[1:].reset_index(drop=True)
    
    # Make sure column names are strings (not nan or other types)
    mscoco_data.columns = [str(col) for col in mscoco_data.columns]
    ffhq_data.columns = [str(col) for col in ffhq_data.columns]
    
    # Convert data to numeric
    for col in mscoco_data.columns[1:]:
        mscoco_data[col] = pd.to_numeric(mscoco_data[col], errors='coerce')
    
    for col in ffhq_data.columns[1:]:
        ffhq_data[col] = pd.to_numeric(ffhq_data[col], errors='coerce')
    
    # Convert dialog length to numeric
    mscoco_data['Dialog Length'] = pd.to_numeric(mscoco_data['Dialog Length'], errors='coerce')
    ffhq_data['Dialog Length'] = pd.to_numeric(ffhq_data['Dialog Length'], errors='coerce')
    
    # Define groups with their performance hierarchy, marker styles, and line styles
    groups = [
        {
            "name": "Fake Image Feedback",
            "models": [
                {"name": "Gemma3-12b", "column": "Fake Image Feedback Gemma3-12b", "color": "gold"},
                {"name": "Gemma3-4b", "column": "Fake Image Feedback Gemma3-4b", "color": "orange"}
            ],
            "zorder": 10,  # Highest performance group - draw on top
            "marker": "o",  # Circle markers for this group
            "markersize": 9,  # Increased marker size
            "linestyle": "-"  # Solid line
        },
        {
            "name": "Prediction Feedback",
            "models": [
                {"name": "Gemma3-12b", "column": "Prediction Feedback Gemma3-12b", "color": "blue"},
                {"name": "Gemma3-4b", "column": "Prediction Feedback Gemma3-4b", "color": "skyblue"}
            ],
            "zorder": 20,  # Middle performance group
            "marker": "s",  # Square markers for this group
            "markersize": 9,  # Increased marker size
            "linestyle": "--"  # Dashed line
        },
        {
            "name": "No Feedback",
            "models": [
                {"name": "Gemma3-12b", "column": "No Feedback Gemma3-12b", "color": "red"},
                {"name": "Gemma3-4b", "column": "No Feedback Gemma3-4b", "color": "salmon"}
            ],
            "zorder": 30,  # Lowest performance group - draw at bottom
            "marker": "^",  # Triangle markers for this group
            "markersize": 9,  # Increased marker size
            "linestyle": ":"  # Dotted line
        }
    ]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    
    # Set a clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Add a title for the entire figure
    # fig.suptitle('Gemma3 Ablation Study: Impact of Feedback Type Across Datasets', fontsize=18, weight='bold', y=0.98)
    
    # Function to plot data on a specific axis
    def plot_dataset(ax, data, title, ylim):
        x_values = data['Dialog Length'].values
        
        # Plot each group in the hierarchy order (low to high)
        # This ensures the highest performing group is on top visually
        for group in reversed(groups):
            for model in group["models"]:
                col_name = model["column"]
                if col_name in data.columns:
                    y_values = data[col_name].values
                    ax.plot(
                        x_values, y_values, 
                        label=f"{model['name']} ({group['name']})",
                        color=model["color"], 
                        linewidth=2.5,
                        marker=group["marker"],
                        markevery=1,  # Add marker at each data point
                        markersize=group["markersize"],
                        linestyle=group["linestyle"],
                        zorder=group["zorder"]
                    )
        
        # Customize the grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis limits
        ax.set_ylim(ylim)
        ax.set_xlim(min(x_values), max(x_values))
        
        # Add labels with larger fonts
        ax.set_xlabel('Dialog Length', fontsize=14, weight='bold')
        ax.set_ylabel('Hits@10 (in %)', fontsize=14, weight='bold')
        ax.set_title(title, fontsize=16, weight='bold')
        
        # Increase font size of tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot MSCOCO dataset on the left subplot
    plot_dataset(ax1, mscoco_data, 'MSCOCO Dataset', (65, 100))
    
    # Plot FFHQ dataset on the right subplot
    plot_dataset(ax2, ffhq_data, 'FFHQ Dataset', (10, 85))
    
    # Create a single legend for both plots
    # Get handles and labels from the first axes (they're the same for both)
    handles, labels = ax1.get_legend_handles_labels()
    
    # Group the legend by category
    by_group = {}
    for handle, label in zip(handles, labels):
        group_name = label.split('(')[1].split(')')[0]
        if group_name not in by_group:
            by_group[group_name] = []
        by_group[group_name].append((handle, label.split(' (')[0]))
    
    # Create a legend with group headers
    legend_handles = []
    legend_labels = []
    
    # Define font properties for the group headers
    group_header_props = {
        'family': 'sans-serif',
        'weight': 'bold',
        'style': 'italic',
        'size': 11  # Larger size for headers
    }
    
    # Add items in our hierarchy order
    for group in groups:
        # Add a header for each group
        legend_handles.append(plt.Line2D([0], [0], color='white'))
        legend_labels.append(f"{group['name']}:")
        
        # Add models in this group
        if group["name"] in by_group:
            for handle, label in by_group[group["name"]]:
                legend_handles.append(handle)
                legend_labels.append(label)
    
    # Add a single legend for the entire figure
    legend = fig.legend(
        legend_handles, legend_labels,
        loc='upper center',  # Position at the upper center between plots
        bbox_to_anchor=(0.5, 0.07),  # Adjust as needed
        ncol=3,
        frameon=True, 
        facecolor='white', 
        framealpha=0.9,
        fontsize=12,
        columnspacing=-1.5,
        handletextpad=0.5,
        borderpad=0.3
    )
    
    # Customize font for group headers in the legend
    header_indices = []
    for i, text in enumerate(legend.get_texts()):
        if legend_labels[i].endswith(':'):  # This is a group header
            text.set_fontproperties(fm.FontProperties(**group_header_props))
            # Store indices of headers to adjust their position
            header_indices.append(i)
    
    fig = plt.gcf()
    fig.canvas.draw()
    for i in header_indices:
        # Get the text object and shift it left
        text = legend.get_texts()[i]
        # Adjust horizontal position of the group headers
        # Negative values move it left
        text.set_position((-120, 0))  # Adjust the value (-120) as needed
    
    # Adjust layout with space for the legend at the bottom
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])  # Adjust the rect for title and legend
    
    # Save the figure if needed
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {output_file}")
    
    # Show the plot
    plt.show()


# Call the function with the filename
if __name__ == "__main__":
    print("Generating Gemma3 ablation study visualization...")
    create_dual_dataset_ablation_plot('paper_writing/GenIR results - Figure 3.csv', 'paper_writing/plot3_2.png')
    print("Visualization complete and saved as 'gemma3_ablation_study.png'")
    print("To view the visualization, open the saved PNG file or look at the displayed figure window.")
