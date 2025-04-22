import pandas as pd
import numpy as np
import sys
sys.path.append('/home/shaghparast/Point_VAE/smlm-pc-vae')  # Adjust the path to your actual directory
from render_projection import renderprojection
# Load the CSV file (replace 'data.csv' with the actual file path)
data = pd.read_csv('data.csv', header=None)
x = data[0].values  # First column (x values)
y = data[1].values  # Second column (y values)

# Create the third column as zeros (z values)
z = np.zeros_like(x)

# Set rendering parameters
alpha = 0  # No rotation for top view
beta = 0
rangex = [-100, 100]
rangey = [-100, 100]
pixelsize = 1
sigma = 2
ploton = True
cmax = 0.3

# Set the directory where the image will be saved
save_directory = "/home/shaghparast/Point_VAE/smlm-pc-vae/testrender"  # Replace with the actual path
filename = "projection_image.png"

# Generate and display the top view plot, and save the figure
img = renderprojection(x, y, z, alpha, beta, rangex, rangey, pixelsize, sigma, ploton, cmax, save_dir=save_directory, filename=filename)
