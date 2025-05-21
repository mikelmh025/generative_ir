import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from matplotlib.text import Text

def create_gemma3_ablation_plot(csv_file, output_file=None):
    """
    Creates a visualization of Gemma3 ablation study results from the CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing Gemma3 ablation results
    output_file : str, optional
        Path to save the output figure
    """
    # Read the CSV file - skip the first row (which has group headers)
    df = pd.read_csv(csv_file, skiprows=1)
    
    # Get x-axis values (Dialog Length)
    x_values = df.iloc[:, 0].values  # First column is Dialog Length
    
    # Define groups with their performance hierarchy, marker styles, and line styles
    # Three settings: No Feedback, Prediction Feedback, Fake Image Feedback
    # For each, we have 4b and 12b model sizes
    groups = [
        {
            "name": "Fake Image Feedback",
            "models": [
                {"name": "Gemma3-12b", "column": 5, "color": "gold"},
                {"name": "Gemma3-4b", "column": 6, "color": "orange"}
            ],
            "zorder": 10,  # Highest performance group - draw on top
            "marker": "o",  # Circle markers for this group
            "markersize": 9,  # Increased marker size
            "linestyle": "-"  # Solid line
        },
        {
            "name": "Prediction Feedback",
            "models": [
                {"name": "Gemma3-12b", "column": 3, "color": "blue"},
                {"name": "Gemma3-4b", "column": 4, "color": "skyblue"}
            ],
            "zorder": 20,  # Middle performance group
            "marker": "s",  # Square markers for this group
            "markersize": 9,  # Increased marker size
            "linestyle": "--"  # Dashed line
        },
        {
            "name": "No Feedback",
            "models": [
                {"name": "Gemma3-12b", "column": 1, "color": "red"},
                {"name": "Gemma3-4b", "column": 2, "color": "salmon"}
            ],
            "zorder": 30,  # Lowest performance group - draw at bottom
            "marker": "^",  # Triangle markers for this group
            "markersize": 9,  # Increased marker size
            "linestyle": ":"  # Dotted line
        }
    ]
    
    # Create figure with a clean, modern style
    plt.figure(figsize=(8, 8))
    
    # Set a clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot each group in the hierarchy order (low to high)
    # This ensures the highest performing group is on top visually
    for group in reversed(groups):
        for model in group["models"]:
            col_idx = model["column"]
            if col_idx < len(df.columns):
                y_values = df.iloc[:, col_idx].values
                plt.plot(
                    x_values, y_values, 
                    label=f"{model['name']} ({group['name']})",
                    color=model["color"], 
                    linewidth=2.5,
                    marker=group["marker"],
                    markevery=2,  # Add marker every 2 data points to avoid crowding
                    markersize=group["markersize"],
                    linestyle=group["linestyle"],
                    zorder=group["zorder"]
                )
    
    # Customize the grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits
    plt.ylim(65, 100)
    plt.xlim(min(x_values), max(x_values))
    
    # Add labels and title with larger fonts
    plt.xlabel('Dialog Length', fontsize=14, weight='bold')
    plt.ylabel('Hits@10 (in %)', fontsize=14, weight='bold')
    # plt.title('Gemma3 Ablation Study: Impact of Model Size Across Feedback Types', fontsize=16, weight='bold')
    
    # Increase font size of tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Create a more organized, categorized legend
    handles, labels = plt.gca().get_legend_handles_labels()
    
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
    
    # Add the legend inside the plot (lower right corner)
    legend = plt.legend(
        legend_handles, legend_labels,
        loc='lower right',  # Position in lower right corner of the plot
        frameon=True, facecolor='white', framealpha=0.9,
        fontsize=12,  # Increased font size
        ncol=3,
        columnspacing=-1.5,  # Reduce space between columns
        handletextpad=0.5,   # Reduce space between handle and text
        borderpad=0.3        # Adjust internal padding
    )
    
    # Customize font for group headers in the legend and adjust position
    header_indices = []
    for i, text in enumerate(legend.get_texts()):
        if legend_labels[i].endswith(':'):  # This is a group header
            text.set_fontproperties(fm.FontProperties(**group_header_props))
            # Store indices of headers to adjust their position
            header_indices.append(i)
    
    # Move group headers to the left
    # This needs to be done after legend is drawn
    fig = plt.gcf()
    fig.canvas.draw()
    for i in header_indices:
        # Get the text object and shift it left
        text = legend.get_texts()[i]
        # Adjust horizontal position of the group headers
        # Negative values move it left
        text.set_position((-120, 0))  # Adjust the value (-120) as needed

    # Add model size comparison annotations
    # plt.annotate('Model Size Comparison', 
    #             xy=(0.5, 0.95), xycoords='figure fraction',
    #             horizontalalignment='center', verticalalignment='top',
    #             fontsize=12, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if needed
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {output_file}")
    
    # Show the plot
    plt.show()


# Call the function with the filename
if __name__ == "__main__":
    create_gemma3_ablation_plot('paper_writing/GenIR results - Figure 3.csv', 'paper_writing/plot3.png')
