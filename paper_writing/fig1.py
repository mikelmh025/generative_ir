import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from matplotlib.text import Text

def create_genIR_plot(csv_file, output_file=None):
    """
    Creates a visualization of GenIR results from the CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing GenIR results
    output_file : str, optional
        Path to save the output figure
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract model names from the first row
    models = df.iloc[0]
    
    # Extract the actual data (starting from row 1)
    data = df.iloc[1:].copy()
    data = data.apply(pd.to_numeric, errors='ignore')
    
    # Get x-axis values (Dialog Length)
    x_values = data.iloc[:, 0].values
    
    # Define groups with their performance hierarchy, marker styles, and line styles
    # Performance: "Text - Fake Image" > "Text with Prediction feedback" > "Text only - No Feed Back"
    groups = [
        {
            "name": "Fake Image Feedback",
            "models": [
                {"name": "Infinity", "column": 4, "color": "gold"},
                {"name": "Lumina", "column": 7, "color": "skyblue"},
                {"name": "SD3.5", "column": 5, "color": "orange"},
                {"name": "Flux", "column": 6, "color": "teal"},
                {"name": "HiDream", "column": 8, "color": "salmon"}
            ],
            "zorder": 10,  # Highest performance group - draw on top
            "marker": "o",  # Circle markers for this group
            "markersize": 9,  # Increased marker size
            "linestyle": "-"  # Solid line
        },
        {
            "name": "Prediction Feedback",
            "models": [
                {"name": "Gemma3-12b", "column": 3, "color": "blue"}
            ],
            "zorder": 20,  # Middle performance group
            "marker": "s",  # Square markers for this group
            "markersize": 9,  # Increased marker size
            "linestyle": "--"  # Dashed line
        },
        {
            "name": "No Feedback",
            "models": [
                {"name": "Gemma3-12b", "column": 2, "color": "red"},
                {"name": "ChatIR", "column": 1, "color": "green"}
            ],
            "zorder": 30,  # Lowest performance group - draw at bottom
            "marker": "^",  # Triangle markers for this group
            "markersize": 9,  # Increased marker size
            "linestyle": ":"  # Dotted line
        }
    ]
    
    # Create figure with a clean, modern style
    plt.figure(figsize=(8, 6))
    
    # Set a clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot each group in the hierarchy order (low to high)
    # This ensures the highest performing group is on top visually
    for group in reversed(groups):
        for model in group["models"]:
            col_idx = model["column"]
            if col_idx < len(df.columns):
                y_values = data.iloc[:, col_idx].values
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
    plt.ylim(60, 100)
    plt.xlim(0, 10)
    
    # Add labels and title with larger fonts
    plt.xlabel('Dialog Length', fontsize=14, weight='bold')
    plt.ylabel('Hits@10 (in %)', fontsize=14, weight='bold')
    # plt.title('GenIR Results by Model and Condition', fontsize=18, weight='bold')
    
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
    
    # Add the legend inside the plot (bottom right corner)
    legend = plt.legend(
        legend_handles, legend_labels,
        loc='lower right',  # Position in lower right corner of the plot
        frameon=True, facecolor='white', framealpha=0.9,
        fontsize=11,  # Increased font size
        # title="Models by Condition"
        ncol=2,
        columnspacing=-1.5,  # Reduce space between columns
        handletextpad=0.5,   # Reduce space between handle and text
        borderpad=0.3        # Adjust internal padding
    )
    
    # Set title font size
    legend.get_title().set_fontsize(12)
    
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
    create_genIR_plot('paper_writing/GenIR results - Figure 1.csv', 'paper_writing/plot1.png')