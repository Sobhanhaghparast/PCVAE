import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random

from utils import *

class PointCloudDataset(Dataset):
    """
    Point Cloud Dataset.

    Args:
        csv_dir (str): Directory containing CSV files.
        num_files (int): Number of CSV files.
        max_points (int): Maximum number of points in the point cloud.
        augment (bool): Whether to apply data augmentation.
    """

    def __init__(self, csv_dir, num_files, max_points, augment=False, modality = 'simulated', shuffle=True):
        self.csv_dir = csv_dir
        self.num_files = num_files
        self.max_points = max_points
        self.augment = augment
        self.modality = modality
        self.shuffle = shuffle # re ordering the data for sub sampling
    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        if self.modality == 'simulated':
            file_path = os.path.join(self.csv_dir, f"{idx + 1}.csv")
            data = pd.read_csv(file_path)
        # shuffle the rows in training data
        #data = data.sample(frac=1).reset_index(drop=True)
        elif self.modality == 'emory':
            file_path = os.path.join(self.csv_dir, f"NUP_{idx + 1}.csv")
            # skip the fitst row
            data = pd.read_csv(file_path, skiprows=1)
        elif self.modality == 'Sim_registered_3D_tetra':
            file_path = os.path.join(self.csv_dir, f"file{idx + 1}.csv")
            data = pd.read_csv(file_path)
        elif self.modality == 'Exp_registered_3D_tetra':
            file_path = os.path.join(self.csv_dir, f"file{idx + 1}.csv")
            data = pd.read_csv(file_path)
        elif self.modality == 'Sim_registered_2D':
            file_path = os.path.join(self.csv_dir, f"file{idx + 1}.csv")
            data = pd.read_csv(file_path)
        elif self.modality == 'Exp_registered_2D':
            file_path = os.path.join(self.csv_dir, f"file{idx + 1}.csv")
            data = pd.read_csv(file_path)
        else:
            assert False, "modality not supported"
        
        if self.shuffle:
            data = data.sample(frac=1).reset_index(drop=True)

        if '3D' in self.modality:
            x = torch.tensor(data.iloc[:, 0].values, dtype=torch.float32)/128
            y = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32)/128
            z = torch.tensor(data.iloc[:, 2].values, dtype=torch.float32)/128
            uncertainty = torch.tensor(data.iloc[:, 3].values, dtype=torch.float32)
            points = torch.stack((x, y, z), dim=1)
        else:
            # Extract x, y coordinates and uncertainty, with a normalization factor of 128
            x = torch.tensor(data.iloc[:, 0].values, dtype=torch.float32)/128
            y = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32)/128
            uncertainty = torch.tensor(data.iloc[:, 2].values, dtype=torch.float32)
            points = torch.stack((x, y), dim=1)
            # points transpose [C,N]
        
        if self.augment:
            points = rotate_point_cloud(points)
            #points = translate_point_cloud(points)
        # Pad point cloud and uncertainty
        padded_points, padded_uncertainty, num_points = pad_point_cloud(points, uncertainty, self.max_points)

        # Extract label (fourth column)
        if self.modality == 'simulated':
            label = torch.tensor(data.iloc[0, 3], dtype=torch.float32)

        if self.modality == 'emory':
            # placeholder for emory data
            label = torch.tensor(data.iloc[0, 2], dtype=torch.float32)

        if self.modality == 'Sim_registered_2D':
            # placeholder
            label = torch.tensor(data.iloc[0, 2], dtype=torch.float32)

        if self.modality == 'Sim_registered_3D_tetra':
            # placeholder
            label = torch.tensor(data.iloc[0, 4], dtype=torch.float32)
        
        if self.modality == 'Exp_registered_3D_tetra':
            # placeholder
            label = torch.tensor(data.iloc[0, 2], dtype=torch.float32)
        
        if self.modality == 'Exp_registered_2D':
            # placeholder
            label = torch.tensor(data.iloc[0, 2], dtype=torch.float32)

        mask = torch.tensor([1] * num_points + [0] * (self.max_points - num_points), dtype=torch.float32)
        # crop if the number of points is larger than max_points
        mask = mask[:self.max_points]

        # the transpose of all of them to be [C,N]
        return padded_points.T, padded_uncertainty.T, label.T, mask.T
        
#        return padded_points, padded_uncertainty, label, mask



def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Args:
        batch (list): List of samples.

    Returns:
        tuple: Batch of points, uncertainties, labels, and masks.
    """
    points = torch.stack([item[0] for item in batch])
    uncertainties = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    masks = torch.stack([item[3] for item in batch])
    return points, uncertainties, labels, masks


def rotate_point_cloud(points):
    """
    Randomly rotates the point cloud around the Z-axis for 2D, or around a random axis for 3D.

    Args:
        points (torch.Tensor): Input point cloud of shape (N, 2) or (N, 3).

    Returns:
        torch.Tensor: Rotated point cloud of shape (N, 2) or (N, 3).
    """
    if points.shape[1] == 2:
        angle = torch.rand(1).item() * 2 * torch.pi
        angle = torch.tensor(angle)
        rotation_matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)]
        ])
        return points @ rotation_matrix.T
    if points.shape[1] == 3:  # Ensure points are in 3D
        # Generate a random rotation angle (in radians) as a PyTorch tensor
        angle = torch.rand(1).item() * 2 * torch.pi
        angle = torch.tensor(angle, dtype=torch.float32)  # Ensure angle is a PyTorch tensor
        
        # Generate a random axis by sampling from a uniform distribution and normalizing
        axis = torch.rand(3) - 0.5
        axis = axis / axis.norm()  # Normalize the axis vector to have unit length
        
        # Identity matrix (3x3) for constructing the rotation matrix
        I = torch.eye(3)
        
        # Construct the skew-symmetric matrix (cross-product matrix) from the axis
        axis_skew = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # Calculate the rotation matrix using Rodrigues' rotation formula
        # Convert angle to a tensor for operations
        rotation_matrix = I + torch.sin(angle) * axis_skew + (1 - torch.cos(angle)) * (axis.unsqueeze(1) @ axis.unsqueeze(0))
        
        # Rotate the points by multiplying them with the rotation matrix
        # Points are of shape [N, 3] and rotation_matrix is [3, 3], so we use matrix multiplication
        rotated_points = points @ rotation_matrix.T
        
        return rotated_points
    else:
        raise ValueError("Points should have shape (N, 2) or (N, 3)")

def translate_point_cloud(points):
    """
    Randomly translates the point cloud.

    Args:
        points (torch.Tensor): Input point cloud of shape (N, 2) or (N, 3).

    Returns:
        torch.Tensor: Translated point cloud of shape (N, 2) or (N, 3).
    """
    translation = torch.rand(1, points.shape[1]) - 0.5
    return points + translation



def pad_point_cloud(point_cloud, uncertainty, max_points):
    """
    Pads the point cloud to a fixed number of points. If the number of points is larger than
    the maximum number of points, the point cloud is cropped.

    Args:
        point_cloud (torch.Tensor): Input point cloud of shape (N, 2).
        uncertainty (torch.Tensor): Uncertainty values of shape (N,).
        max_points (int): Number of points to pad to.

    Returns:
        tuple: Padded point cloud of shape (max_points, 2), padded uncertainty, and the number of original points.
    """
    num_points = point_cloud.shape[0]
    if num_points < max_points:
        padded_points = torch.zeros((max_points, point_cloud.shape[1]), dtype=torch.float32)
        padded_uncertainty = torch.zeros(max_points, dtype=torch.float32)
        padded_points[:num_points, :] = point_cloud
        padded_uncertainty[:num_points] = uncertainty
    # croppoint
    else:
        # print howmany points are cropped
        #print(f"cropped {num_points - max_points} points")
        padded_points = point_cloud[:max_points, :]
        padded_uncertainty = uncertainty[:max_points]
        
    return padded_points, padded_uncertainty, num_points

def create_dataloader(csv_dir, num_files, max_points, batch_size, augment=False, shuffle=True, modality = 'simulated', order_shuffle=True):
    """
    Creates a DataLoader for the Point Cloud Dataset.

    Args:
        csv_dir (str): Directory containing CSV files.
        num_files (int): Number of CSV files.
        max_points (int): Maximum number of points in the point cloud.
        batch_size (int): Batch size.
        augment (bool): Whether to apply data augmentation.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = PointCloudDataset(csv_dir, num_files, max_points, augment=augment, modality = modality, shuffle=order_shuffle)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader
