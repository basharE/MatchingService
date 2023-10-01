from matplotlib import pyplot as plt
import random


def plot_line_graph(x1, y1, x2, y2, labels, output_file, image):
    # Create a dictionary to map unique labels to colors
    label_colors = {}
    for label in set(labels):
        if label not in label_colors:
            label_colors[label] = generate_random_color()

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plotting the lines with colored points on the first subplot
    for label, color in label_colors.items():
        x_points = [x1[i] for i in range(len(x1)) if labels[i] == label]
        y_points = [y1[i] for i in range(len(y1)) if labels[i] == label]
        ax1.plot(x_points, y_points, marker='o', linestyle='-', label=label, color=color)

    # Plotting the lines with colored points on the second subplot
    for label, color in label_colors.items():
        x_points = [x2[i] for i in range(len(x2)) if labels[i] == label]
        y_points = [y2[i] for i in range(len(y2)) if labels[i] == label]
        ax2.plot(x_points, y_points, marker='o', linestyle='-', label=label, color=color)

    # Naming the y-axes for each subplot
    ax1.set_xlabel('Index')
    ax2.set_xlabel('Index')

    # Naming the y-axes for each subplot
    ax1.set_ylabel('Similarity')
    ax2.set_ylabel('Similarity')

    # Giving titles to the subplots
    ax1.set_title(image + ' - Similarity Line Graph For Clip')
    ax2.set_title(image + ' - Similarity Line Graph For Resnet')

    # Show a legend with unique labels and their corresponding colors
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color)
               for label, color in label_colors.items()]
    ax1.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title='Clip Objects Id')
    ax2.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title='Resnet Objects Id')

    # Save the plot to a file
    plt.savefig(output_file, bbox_inches='tight')

    # Function to show the plot (optional)
    plt.show()


def generate_random_color():
    # Generate a random color in hexadecimal format
    return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
