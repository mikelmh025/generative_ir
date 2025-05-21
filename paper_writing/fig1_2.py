import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

def create_genIR_split_plot(csv_file, output_file=None):
    """
    Creates a visualization of GenIR results from the CSV file split into two subfigures.
    
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
            "zorder": 10,
            "marker": "o",
            "markersize": 9,
            "linestyle": "-"
        },
        {
            "name": "Prediction Feedback",
            "models": [
                {"name": "Gemma3-12b", "column": 3, "color": "blue"}
            ],
            "zorder": 20,
            "marker": "s",
            "markersize": 9,
            "linestyle": "--"
        },
        {
            "name": "No Feedback",
            "models": [
                {"name": "Gemma3-12b", "column": 2, "color": "red"},
                {"name": "ChatIR", "column": 1, "color": "green"}
            ],
            "zorder": 30,
            "marker": "^",
            "markersize": 9,
            "linestyle": ":"
        }
    ]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Set a clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # SUBPLOT 1: Comparison across feedback types
    # Only include Infinity from Fake Image Feedback, and all models from other groups
    for group in groups:
        if group["name"] == "Fake Image Feedback":
            # Only plot Infinity for this group
            model = next(m for m in group["models"] if m["name"] == "Infinity")
            col_idx = model["column"]
            if col_idx < len(df.columns):
                y_values = data.iloc[:, col_idx].values
                ax1.plot(
                    x_values, y_values, 
                    # Renamed to feedback category instead of model name
                    label="Fake Image Feedback (Ours)",
                    color=model["color"], 
                    linewidth=2.5,
                    marker=group["marker"],
                    markevery=2,
                    markersize=group["markersize"],
                    linestyle=group["linestyle"],
                    zorder=group["zorder"]
                )
        elif group["name"] == "Prediction Feedback":
            # Plot all models from Prediction Feedback without group name
            for model in group["models"]:
                col_idx = model["column"]
                if col_idx < len(df.columns):
                    y_values = data.iloc[:, col_idx].values
                    ax1.plot(
                        x_values, y_values, 
                        # Renamed to feedback category instead of model name
                        label="Prediction Feedback",
                        color=model["color"], 
                        linewidth=2.5,
                        marker=group["marker"],
                        markevery=2,
                        markersize=group["markersize"],
                        linestyle=group["linestyle"],
                        zorder=group["zorder"]
                    )
        else:
            # Keep No Feedback group as-is
            for model in group["models"]:
                col_idx = model["column"]
                if col_idx < len(df.columns):
                    y_values = data.iloc[:, col_idx].values
                    ax1.plot(
                        x_values, y_values, 
                        label=f"{model['name']} ({group['name']})",
                        color=model["color"], 
                        linewidth=2.5,
                        marker=group["marker"],
                        markevery=2,
                        markersize=group["markersize"],
                        linestyle=group["linestyle"],
                        zorder=group["zorder"]
                    )
    
    # SUBPLOT 2: Comparison within Fake Image Feedback models
    fake_image_group = next(g for g in groups if g["name"] == "Fake Image Feedback")
    for model in fake_image_group["models"]:
        col_idx = model["column"]
        if col_idx < len(df.columns):
            y_values = data.iloc[:, col_idx].values
            ax2.plot(
                x_values, y_values, 
                label=model["name"],
                color=model["color"], 
                linewidth=2.5,
                marker=fake_image_group["marker"],
                markevery=2,
                markersize=fake_image_group["markersize"],
                linestyle=fake_image_group["linestyle"]
            )
    
    # Customize subplot 1
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(58, 100)  # Same overall range as original
    ax1.set_xlim(0, 10)
    ax1.set_xlabel('Dialog Length', fontsize=14, weight='bold')
    ax1.set_ylabel('Hits@10 (in %)', fontsize=14, weight='bold')
    ax1.set_title('Comparison Across Feedback Types', fontsize=16, weight='bold')
    ax1.tick_params(axis='both', labelsize=12)
    
    # Customize subplot 2
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(82, 100)  # Adjusted range to better show differences
    ax2.set_xlim(0, 10)
    ax2.set_xlabel('Dialog Length', fontsize=14, weight='bold')
    ax2.set_ylabel('Hits@10 (in %)', fontsize=14, weight='bold')
    ax2.set_title('Comparison of Fake Image Feedback Models', fontsize=16, weight='bold')
    ax2.tick_params(axis='both', labelsize=12)
    
    # Modified legend creation for subplot 1
    handles1, labels1 = ax1.get_legend_handles_labels()
    
    # Organize handles and labels by feedback type
    no_feedback_handles = []
    no_feedback_labels = []
    direct_handles = []
    direct_labels = []
    
    for handle, label in zip(handles1, labels1):
        if "(No Feedback)" in label:
            no_feedback_handles.append(handle)
            no_feedback_labels.append(label.split(' (')[0])
        else:
            direct_handles.append(handle)
            direct_labels.append(label)
    
    # Create the final legend handles and labels in the desired order
    legend_handles1 = []
    legend_labels1 = []
    
    # Add direct models first (Infinity and Gemma3-12b without group headers)
    for handle, label in zip(direct_handles, direct_labels):
        legend_handles1.append(handle)
        legend_labels1.append(label)
    
    # Then add No Feedback group with its header
    if no_feedback_handles:
        # Add a header for No Feedback group
        legend_handles1.append(plt.Line2D([0], [0], color='white'))
        legend_labels1.append("No Feedback:")
        
        # Add models in this group
        for handle, label in zip(no_feedback_handles, no_feedback_labels):
            legend_handles1.append(handle)
            legend_labels1.append(label)
    
    # For subplot 2 - simpler legend since all are from the same group
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    # Font properties for group headers
    group_header_props = {
        'family': 'sans-serif',
        'weight': 'bold',
        'style': 'italic',
        'size': 13
    }
    
    # Add legends to plots
    legend1 = ax1.legend(
        legend_handles1, legend_labels1,
        loc='lower right',
        frameon=True, facecolor='white', framealpha=0.9,
        fontsize=13,
        ncol=1,
        # columnspacing=-1.5,
    )
    
    legend2 = ax2.legend(
        handles2, labels2,
        loc='lower right',
        frameon=True, facecolor='white', framealpha=0.9,
        fontsize=13,
    )
    
    # Customize group headers in legend 1
    header_indices = []
    for i, text in enumerate(legend1.get_texts()):
        if legend_labels1[i].endswith(':'):  # This is a group header
            text.set_fontproperties(fm.FontProperties(**group_header_props))
            header_indices.append(i)
    
    # Draw figure to apply text properties
    fig.canvas.draw()
    for i in header_indices:
        text = legend1.get_texts()[i]
        text.set_position((-140, 0))
    
    # Set title for legend 2
    legend2.get_title().set_fontsize(12)
    
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
    create_genIR_split_plot('paper_writing/GenIR results - Figure 1.csv', 'paper_writing/plot1_2.png')