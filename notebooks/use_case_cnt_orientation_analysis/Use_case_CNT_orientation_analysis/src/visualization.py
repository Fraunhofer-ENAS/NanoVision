import matplotlib.pyplot as plt
import numpy as np



def show_angle_orientation(ax, image, angles, arrow_length=50, colors=None):
    """
    Show the image with arrows indicating dominant FFT orientations.
    
    Parameters:
        ax: matplotlib axis to draw on
        image: 2D image
        angles: list of orientations in degrees (0°=vertical)
        arrow_length: length of arrows
        colors: list of colors for arrows
    """
    h, w = image.shape
    center = (w/2, h/2)
    ax.imshow(image, cmap='gray')
    
    if colors is None:
        colors = ['red'] * len(angles)
        
    for angle, color in zip(angles, colors):
        # Convert angle to radians
        theta = np.deg2rad(angle)
        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)  # negative because image y-axis goes down
        ax.arrow(center[0], center[1], dx, dy, color=color, linewidth=2, head_width=5)
    
    ax.set_title("Input Image with Orientations")
    ax.axis('off')




def draw_arrow(center, angle_deg, length, color, label):
    """Draws an arrow for a given orientation angle in degrees."""
    angle_rad = np.deg2rad(angle_deg)
    dx = length * np.cos(angle_rad)
    dy = -length * np.sin(angle_rad)
    plt.arrow(center[0], center[1], dx, dy, 
              color=color, width=2, head_width=10, head_length=15, 
              label=label, alpha=0.7)
    

def show_fft_orientation(image, angle):
    h, w = image.shape
    center = (w/2, h/2)
    length = max(h, w) * 0.4

    plt.imshow(image, cmap='gray')
    draw_arrow(center, angle, length, color='red', label='FFT Orientation')
    plt.title(f'FFT main orientation: {angle:.2f}°')
    plt.axis('off')
    plt.show()


def show_mask_orientation(mask, angle):
    plt.imshow(mask, cmap='gray')
    plt.title(f'Mask orientation: {angle:.2f}°')
    plt.axis('off')
    plt.show()

def plot_circular_distribution(angles):
    plt.hist(angles, bins=36)
    plt.xlabel('Angle (°)')
    plt.ylabel('Count')
    plt.title('Orientation Distribution')
    plt.show()

def plot_error_histogram(errors):
    plt.hist(errors, bins=36)
    plt.xlabel('Error (°)')
    plt.ylabel('Count')
    plt.title('Circular Error Histogram')
    plt.show()


