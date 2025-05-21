import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from matplotlib.text import Text

def create_dual_dataset_ablation_plot(csv_file, output_file=None):
    """
    Creates a visualization with six subplots (2x3 grid):
    - Rows: MSCOCO and FFHQ datasets
    - Columns: Fake Image Feedback, Prediction Feedback, and No Feedback
    
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
    
    # Define feedback types with their associated models, marker styles, and line styles
    feedback_types = [
        {
            "name": "Synthetic Image Feedback",
            "models": [
                {"name": "Gemma3-12b", "column": "Fake Image Feedback Gemma3-12b", "color": "gold"},
                {"name": "Gemma3-4b", "column": "Fake Image Feedback Gemma3-4b", "color": "orange"}
            ],
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
            "marker": "s",  # Square markers for this group
            "markersize": 9,  # Increased marker size
            "linestyle": "--"  # Dashed line
        },
        {
            "name": "Verbal Feedback",
            "models": [
                {"name": "Gemma3-12b", "column": "No Feedback Gemma3-12b", "color": "red"},
                {"name": "Gemma3-4b", "column": "No Feedback Gemma3-4b", "color": "salmon"}
            ],
            "marker": "^",  # Triangle markers for this group
            "markersize": 9,  # Increased marker size
            "linestyle": ":"  # Dotted line
        }
    ]
    
    # Create figure with 2x3 subplot grid (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharey='row')
    
    # Set a clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Define y-limits for each dataset (same for all feedback types in a row)
    y_limits = {
        'MSCOCO': (65, 100),
        'FFHQ': (0, 100)
    }
    
    # Dataset titles and corresponding data
    datasets = [
        {'name': 'MSCOCO Hits@10 (in %)', 'data': mscoco_data, 'ylim': y_limits['MSCOCO']},
        {'name': 'FFHQ Hits@10 (in %)', 'data': ffhq_data, 'ylim': y_limits['FFHQ']}
    ]
    
    # Function to plot a specific feedback type on a specific axis
    def plot_feedback_type(ax, data, feedback_type, show_ylabel=False, show_xlabel=False, ylim=None):
        x_values = data['Dialog Length'].values
        
        # Plot each model for this feedback type
        for model in feedback_type["models"]:
            col_name = model["column"]
            if col_name in data.columns:
                y_values = data[col_name].values
                ax.plot(
                    x_values, y_values, 
                    label=f"{model['name']}",
                    color=model["color"], 
                    linewidth=2.5,
                    marker=feedback_type["marker"],
                    markevery=1,  # Add marker at each data point
                    markersize=feedback_type["markersize"],
                    linestyle=feedback_type["linestyle"]
                )
        
        # Customize the grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis limits
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlim(min(x_values), max(x_values))
        
        # Customize the y-ticks to show just a few values
        if ylim:
            min_val, max_val = ylim
            step = (max_val - min_val) / 4  # Use 5 ticks (including min and max)
            ticks = [min_val + i * step for i in range(5)]
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{int(t)}" for t in ticks])
        
        # Only add x-label for bottom plots
        if show_xlabel:
            ax.set_xlabel('Dialog Length', fontsize=12, weight='bold')
        
        # Increase font size of tick labels
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add a small legend for each subplot
        ax.legend(fontsize=10, loc='lower right')
    
    # Plot each dataset (row) and feedback type (column)
    for row_idx, dataset in enumerate(datasets):
        for col_idx, feedback_type in enumerate(feedback_types):
            # Determine if we should show labels
            show_ylabel = (col_idx == 0)  # Only show y-label for first column
            show_xlabel = (row_idx == 1)  # Only show x-label for bottom row
            
            # Plot this specific combination of dataset and feedback type
            plot_feedback_type(
                axes[row_idx, col_idx],
                dataset['data'],
                feedback_type,
                show_ylabel=show_ylabel,
                show_xlabel=show_xlabel,
                ylim=dataset['ylim']  # Use same y-limit for all plots in this row
            )
    
    # Add dataset names as row titles (vertical text on y-axis)
    for idx, dataset in enumerate(datasets):
        # Add the dataset name to the left of the row
        fig.text(
            0.02,  # x position - far left
            0.75 - idx * 0.5,  # y position - centered vertically for each row
            dataset['name'],
            fontsize=14,
            weight='bold',
            rotation=90,  # Vertical text
            ha='center',
            va='center'
        )
    
    # Add feedback type names as column titles
    fig_width = fig.get_figwidth()
    col_positions = [1/6, 3/6, 5/6]  # Centers of the three columns
    
    for idx, feedback_type in enumerate(feedback_types):
        fig.text(
            col_positions[idx],  # x position - centered for each column
            0.95,  # y position - top of figure
            feedback_type['name'],
            fontsize=14,
            weight='bold',
            ha='center',
            va='center'
        )
    
    # Adjust layout with more space on the left for row titles and reduced spacing between subplots
    plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.1, wspace=0.04, hspace=0.2)
    
    # Save the figure if needed
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {output_file}")
    
    # Show the plot
    plt.show()

# Call the function with the filename
if __name__ == "__main__":
    print("Generating Gemma3 ablation study visualization...")
    create_dual_dataset_ablation_plot('paper_writing/GenIR results - Figure 3.csv', 'paper_writing/plot3_4.png')
    print("Visualization complete and saved as 'plot3_4.png'")
    print("To view the visualization, open the saved PNG file or look at the displayed figure window.")