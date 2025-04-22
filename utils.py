import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
from matplotlib import pyplot as plt

def visualize_point_cloud(points, mask, dim=2):
    """
    Visualizes a point cloud with a black background and red points.
    
    Args:
        points (torch.Tensor): Point cloud of shape (N, 2).
        mask (torch.Tensor): Mask indicating valid points.
    """
    if dim == 3:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-128, 128)
        ax.set_ylim(-128, 128)
        ax.set_zlim(-128, 128)
        # background color black
        ax.set_facecolor('black')
        ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], s=50, c='red', marker='o')
        return
    else:
        # Filter out padding points
        filtered_points = points[mask]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, facecolor='black')
        ax.set_xlim(-128, 128)
        ax.set_ylim(-128, 128)
        # background color black
        ax.set_facecolor('black')
        ax.scatter(filtered_points[:, 0], filtered_points[:, 1], s=50, c='red', marker='o')
        
        # TODO: instead of scatter, please according to the density of the points, plot the points

        return


