import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

def rotcoorddeg(x, y, angle):
    # Rotate coordinates by a given angle (in degrees)
    angle_rad = np.deg2rad(angle)
    xo = np.cos(angle_rad) * x + np.sin(angle_rad) * y
    yo = np.cos(angle_rad) * y - np.sin(angle_rad) * x
    return xo, yo

def renderprojection(x, y, z, alpha, beta, rangex, rangey, pixelsize, sigma, ploton, cmax, save_dir=None, filename="output_image.png"):
    # Rotate the x and y coordinates
    xo, yo = rotcoorddeg(x, y, alpha)
    # Rotate the y and z coordinates
    y2, zo = rotcoorddeg(yo, z, beta)
    
    # Create bins for the histogram
    rx = np.arange(rangex[0], rangex[-1] + pixelsize, pixelsize)
    ry = np.arange(rangey[0], rangey[-1] + pixelsize, pixelsize)
    
    # Create a 2D histogram of the rotated coordinates
    img, xedges, yedges = np.histogram2d(y2, xo, bins=[ry, rx])
    
    # Apply Gaussian filter if sigma > 0
    if sigma > 0:
        img = gaussian_filter(img, sigma=sigma)
    
    # Plot the image if ploton is True
    if ploton:
        plt.figure()
        plt.imshow(img, cmap='hot', interpolation='nearest', origin='lower', vmax=cmax)
        plt.gca().set_aspect('equal', adjustable='box')

        # Remove grid, axis ticks, and labels
        plt.gca().axis('off')

        # Save the figure if save_dir is provided
        if save_dir:
            # Ensure the directory exists
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            print(f"Figure saved to {save_path}")

        plt.show()

    return img
