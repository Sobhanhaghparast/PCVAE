import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import random
from utils import *
from dataset import *
from models import PointNetVAE, PointNetVAEV2
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from scipy.stats import linregress
import yaml


def loss_function(recon_x, x, mean, logvar, mask, loss_type='chamfer', kl_weight=1.0):
    """
    Loss function for the VAE, including reconstruction loss and KL divergence.
    
    Args:
        recon_x (torch.Tensor): Reconstructed point cloud.
        x (torch.Tensor): Original point cloud.
        mean (torch.Tensor): Mean of the latent space distribution.
        logvar (torch.Tensor): Log variance of the latent space distribution.
        mask (torch.Tensor): Mask indicating valid points.
        loss_type (str): Type of reconstruction loss ('mse', 'chamfer', 'emd').

    Returns:
        tuple: Reconstruction loss and KL divergence.
    """
    recon_x = recon_x.permute(0, 2, 1)
    x = x.permute(0, 2, 1)
    mask = mask.unsqueeze(2).float()
    # return the number of points of the mask by calculating the sum of the mask along batch
    mask_count = mask.sum(dim=1).sum(dim=1).long()


    if loss_type == 'mse':
        recon_loss = F.mse_loss(recon_x * mask, x * mask, reduction='sum')
    elif loss_type == 'chamfer':
        recon_loss = chamfer_distance(recon_x, x,x_lengths = mask_count, y_lengths=mask_count)[0]
    elif loss_type == 'emd':
        # Note: PyTorch3D does not have a direct EMD implementation, consider implementing your own or using an available library.
        raise NotImplementedError("EMD loss is not implemented.")
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    # check kld loss
    
    kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
    return recon_loss, kl_weight*kld_loss

class Trainer:
    """
    Trainer class for training and validating the PointNetVAE model.
    
    Args:
        model (nn.Module): The PointNetVAE model.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (function): Loss function for the model.
        device (str): Device to run the model on ('cpu' or 'cuda').
        log_dir (str): Directory to save TensorBoard logs.
    """
    def __init__(self, model, train_loader, val_loader, analysis_dataloader, optimizer, criterion, device='cuda', log_dir='./logs', kl_weight_end=1.0, modality = 'simulated',
                 kl_weight_begin = 1e-5, warmup_steps=500, plot_rsquared = True, plot_pca = False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.analysis_dataloader = analysis_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = SummaryWriter(log_dir)
        self.model.to(self.device)
        self.kl_weight_end = kl_weight_end
        self.modality = modality
        self.step = 0
        self.kl_weight_begin = kl_weight_begin
        self.warmup_steps = warmup_steps
        self.plot_rsquared = False
        self.plot_pca = False
    def train(self, epochs):
        """
        Trains the model for a given number of epochs.

        Args:
            epochs (int): Number of epochs to train the model for.
        """
        for epoch in range(epochs):
            self.model.train()
            train_recon_loss = 0
            train_kld_loss = 0
            
            for batch in tqdm(self.train_loader):
                self.step += 1
                
                points, uncertainties, labels, masks = batch
                points, masks = points.to(self.device), masks.to(self.device)

                self.optimizer.zero_grad()
                recon_points, mean, logvar = self.model(points,masks)
                # warm up if step is less than warmup_steps
                if self.step < self.warmup_steps:
                    self.kl_weight = self.kl_weight_warmup(self.step, self.kl_weight_begin, self.kl_weight_end, self.warmup_steps)
                    print(f"KL weight: {self.kl_weight}")
                recon_loss, kld_loss = self.criterion(recon_points, points, mean, logvar, masks, kl_weight=self.kl_weight)
                loss = recon_loss + kld_loss
                # step logging
                self.writer.add_scalar('Loss/recon_loss_step', recon_loss.item(), self.step)
                self.writer.add_scalar('Loss/kld_loss_step', kld_loss.item()/self.kl_weight, self.step)
                self.writer.add_scalar('Loss/total_loss_step', loss.item(), self.step)
                

                loss.backward()
                self.optimizer.step()

                train_recon_loss += recon_loss.item()
                train_kld_loss += kld_loss.item()

            train_recon_loss /= len(self.train_loader)
            train_kld_loss /= len(self.train_loader)
            self.writer.add_scalar('Loss/train_recon', train_recon_loss, epoch)
            self.writer.add_scalar('Loss/train_kld', train_kld_loss/self.kl_weight, epoch)
            self.writer.add_scalar('Loss/total_loss', train_recon_loss + train_kld_loss, epoch)

            
            
            if epoch % 20 == 0:
                #self._log_images(points, recon_points, masks, epoch, mode='train')
                val_recon_loss, val_kld_loss = self.validate(epoch)
                self.analyze_latent_space(epoch)
                print(f"Epoch [{epoch + 1}/{epochs}], Train Recon Loss: {train_recon_loss:.4f}, Train KLD Loss: {train_kld_loss/self.kl_weight:.4f}, Val Recon Loss: {val_recon_loss:.4f}, Val KLD Loss: {val_kld_loss:.4f}")

    def validate(self, epoch):
        """
        Validates the model on the validation set.

        Args:
            epoch (int): Current epoch number.

        Returns:
            tuple: Validation reconstruction loss and KL divergence.
        """
        self.model.eval()
        val_recon_loss = 0
        val_kld_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                points, uncertainties, labels, masks = batch
                points, masks = points.to(self.device), masks.to(self.device)

                recon_points, mean, logvar = self.model(points, masks)
                recon_loss, kld_loss = self.criterion(recon_points, points, mean, logvar, masks, kl_weight=self.kl_weight)
                val_recon_loss += recon_loss.item()
                val_kld_loss += kld_loss.item()

            val_recon_loss /= len(self.val_loader)
            val_kld_loss /= len(self.val_loader)
            self.writer.add_scalar('Loss/val_recon', val_recon_loss, epoch)
            self.writer.add_scalar('Loss/val_kld', val_kld_loss/self.kl_weight, epoch)
            self.writer.add_scalar('Loss/val_total', val_recon_loss + val_kld_loss, epoch)
            self._log_images(points, recon_points, masks, epoch, mode='val')

        return val_recon_loss, val_kld_loss

    def _log_images(self, original, reconstructed, mask, epoch, mode):
        """
        Logs the original and reconstructed point clouds to TensorBoard.

        Args:
            original (torch.Tensor): Original point clouds.
            reconstructed (torch.Tensor): Reconstructed point clouds.
            mask (torch.Tensor): Mask indicating valid points.
            epoch (int): Current epoch number.
            mode (str): Mode of logging ('train' or 'val').
        """
        original = original.cpu().detach()
        reconstructed = reconstructed.cpu().detach()
        mask = mask.cpu().detach()
        

        for i in range(min(8, original.size(0))):
            # find the last non-zero element in the mask
            last = mask[i].nonzero(as_tuple=True)[0].max()
            orig = original[i][:, :last]
            recon = reconstructed[i][:, :last]
            diff = (orig - recon).abs()

            self.writer.add_figure(f'{mode}_original/point_cloud_{i}', self._plot_point_cloud(orig), epoch)
            self.writer.add_figure(f'{mode}_reconstructed/point_cloud_{i}', self._plot_point_cloud(recon), epoch)
            self.writer.add_figure(f'{mode}_difference/point_cloud_{i}', self._plot_point_cloud(diff), epoch)

    def analyze_latent_space(self, epoch):
            
            mu_list = []
            label_list = []
            out_points_list=[]
            for data in tqdm(self.analysis_dataloader):
                # we use mu as the latent representation and label
                points, uncertainties, label, masks = data
                points, masks = points.to(self.device), masks.to(self.device)
                out_points, mean, _ = self.model(points,masks)
                label_list.append(label)
                mu_list.append(mean)
                out_points_list.append(out_points)

            
            log_dir = os.path.join(self.writer.log_dir, f"recons_epochs_{epoch}")
            # Ensure the directory exists
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Save each tensor in out_points_list as a CSV file
            for i, out_points in enumerate(out_points_list):
                # Convert tensor to numpy array
                out_points_np = out_points.cpu().detach().numpy()  # Move to CPU and convert to numpy array

                # Define the file name
                file_name = os.path.join(log_dir, f"out_{i + 1}.csv")

                # Save the numpy array to a CSV file
                np.savetxt(file_name, out_points_np.squeeze(0).transpose(1, 0), delimiter=",")

            mu_matrix = torch.stack(mu_list)
            # make sure the shape is N, latent_dim
            matrix = mu_matrix.squeeze(1)    
            # label should be [batch_size, 1] and we need to stack them
            label = torch.stack(label_list)
            # make sure the shape is N
            label = label.squeeze(1)
            # put all back to cpu
            matrix = matrix.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            print('label shape:', label.shape)
            
            print('mu_matrix shape:', matrix.shape)
            # Step 2: Perform PCA
            n_components = latent_dim  # Number of principal components to compute
            pca = PCA(n_components=n_components)
            pca.fit(matrix)
            np_mu_matrix = mu_matrix.cpu().detach().numpy().reshape(-1,latent_dim)
            pd.DataFrame(np_mu_matrix).to_csv(os.path.join(self.writer.log_dir, f'latent_space_epoch_{epoch}.csv'), index=False)

            if self.modality == 'emory':
            # load label.csv
                label = pd.read_csv('tabular_dataset/Labels.csv')
                Radius = label.iloc[:, 1]  # Index 3 corresponds to the fourth column
                Ellipticity = label.iloc[:, 2]  # Index 3 corresponds to the fourth column
                try:
                    Numberb = label.iloc[:, 0]  # Index 3 corresponds to the fourth column
                except:
                    Numberb = Ellipticity
            if self.modality == 'Sim_registered_2D':
                label = pd.read_csv('data/Sim_registered_2D/labels.csv', header= None)
                print('label length:', len(label))
                Radius = label.iloc[:, 0]
                Ellipticity = label.iloc[:, 1]
                Numberb = label.iloc[:, 2]
            if self.modality == 'Exp_registered_2D':
                label = pd.read_csv(f'data/{self.modality}/labels.csv', header= None)
                print('label length:', len(label))
                Radius = label.iloc[:, 2]  # Index 3 corresponds to the fourth column
                Ellipticity = label.iloc[:, 3]
                Numberb = label.iloc[:, 4]

            if self.modality == 'Sim_registered_3D_tetra':
                label = pd.read_csv(f'data/{self.modality}/labels.csv', header= None)
                print('label length:', len(label))
                Radius = label.iloc[:, 0]
            
            if self.modality == 'Exp_registered_3D_tetra':
                label = pd.read_csv(f'data/{self.modality}/labels.csv', header= None)
                print('label length:', len(label))
                Radius = label.iloc[:, 0]

            # Step 3: Get the principal component axes and eigenvalues
            principal_components = pca.components_  # Principal component axes
            eigenvalues = pca.explained_variance_   # Eigenvalues

            variance_explained = eigenvalues / np.sum(eigenvalues)
            data_projection = np.dot(matrix, principal_components.T)

            # Plot the histogram of variance explained
            fig = plt.figure(figsize=(8, 4))
            plt.plot((variance_explained), marker='o')
            plt.xlabel('Number of Dimensions (Principal Components)')
            plt.ylabel(' Variance Explained')
            plt.title(' Variance Explained vs Number of Dimensions')
            plt.grid(True)         
            # log this into tensorboard
            self.writer.add_figure("Variance Explained vs Number of Dimensions", fig, epoch)
            # save the figure
            fig.savefig(os.path.join(self.writer.log_dir, f"Variance Explained vs Number of Dimensions_epoch_{epoch}.png"))
            # close
            plt.close(fig)

            if True: # the r2 analysis, can be turned off if only reconstruction is wanted
            # raw fitting R^2
                if self.modality == 'simulation':
                    labels = np.array([label, label, label])
                elif self.modality == 'emory':
                    labels = np.array([Radius, Ellipticity, Numberb])
                elif self.modality == 'Sim_registered_2D':
                    labels = np.array([Radius, Ellipticity, Numberb])
                elif self.modality == 'Exp_registered_2D':
                    labels = np.array([Radius, Ellipticity, Numberb])
                elif self.modality == 'Sim_registered_3D_tetra':
                    labels = np.array([Radius,Radius,Radius])
                elif self.modality == 'Exp_registered_3D_tetra':
                    labels = np.array([Radius,Radius,Radius])

                

                matrix = matrix
                x_transform = np.transpose(matrix)
                regression_matrix = np.zeros((latent_dim, 3))  # Assuming the shape based on the loop range

                for i in range(3):
                    for j in range(latent_dim):
                        slope, intercept, r_value, p_value, std_err = linregress(x_transform[j, :], labels[i, :])
                        # Calculate R-squared value
                        regression_matrix[j, i] = r_value**2
                
                print("Raw: Latent unit", np.argmax(regression_matrix[:,0]), "is related to Radius and r_squared is:",np.max(regression_matrix[:,0]))
                # log the best raw r squared value
                self.writer.add_scalar("Raw_Radius", np.max(regression_matrix[:,0]), epoch)
                print("Raw: Latent unit", np.argmax(regression_matrix[:,1]), "is related to Ellipticity and r_squared is:",np.max(regression_matrix[:,1]))

                # log the best raw r squared value
                self.writer.add_scalar("Raw_Ellipticity", np.max(regression_matrix[:,1]), epoch)

                print("Raw: Latent unit", np.argmax(regression_matrix[:,2]), "is related to Numberb and r_squared is:",np.max(regression_matrix[:,2]))
                # log the best raw r squared value
                self.writer.add_scalar("Raw_Numberb", np.max(regression_matrix[:,2]), epoch)
                if self.plot_rsquared:
                    #print("Raw: Latent unit", np.argmax(regression_matrix[:,2]), "is related to Numberb and r_squared is:",np.max(regression_matrix[:,2]))
                    fig, axs = plt.subplots(latent_dim, 3, figsize=(9, 6*latent_dim))

                    c = 1
                    for i in range(3):
                        for j in range(latent_dim):
                            # Scatter plot
                            axs[j, i].scatter(x_transform[j, :], labels[i, :])
                            axs[j, i].set_title(f"dim{j} label{i}")

                            # Calculate and plot regression line
                            slope, intercept, r_value, p_value, std_err = linregress(x_transform[j, :], labels[i, :])
                            line = slope * x_transform[j, :] + intercept
                            axs[j, i].plot(x_transform[j, :], line, color='red')

                            # Calculate R-squared value
                            r_squared = r_value ** 2

                            # Add R-squared value as text annotation
                            axs[j, i].text(0.5, 0.9, f'R-squared: {r_squared:.4f}', horizontalalignment='center', verticalalignment='center',
                                        transform=axs[j, i].transAxes, bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.3'))

                    plt.tight_layout()
                    #plt.show()
                    # save the figure as isomap result
                    # log
                    self.writer.add_figure("Raw_result", fig, epoch)
                    fig.savefig(os.path.join(self.writer.log_dir, f"Raw_result_epoch_{epoch}.png"))
                    plt.close(fig)

            if self.plot_pca:
                # PCA-based analysis
                X_transformed = pca.fit_transform(matrix)
                x_transform = np.transpose(X_transformed)
                regression_matrix = np.zeros((latent_dim, 3))

                for i in range(3):
                    for j in range(latent_dim):
                        slope, intercept, r_value, p_value, std_err = linregress(x_transform[j, :], labels[i, :])
                        # Calculate R-squared value
                        regression_matrix[j, i] = r_value**2
                
                print("PCA: Latent unit", np.argmax(regression_matrix[:,0]), "is related to Radius and r_squared is:",np.max(regression_matrix[:,0]))
                # log the best raw r squared value
                self.writer.add_scalar("PCA_Radius", np.max(regression_matrix[:,0]), epoch)
                print("PCA: Latent unit", np.argmax(regression_matrix[:,1]), "is related to Ellipticity and r_squared is:",np.max(regression_matrix[:,1]))

                # log the best raw r squared value
                self.writer.add_scalar("PCA_Ellipticity", np.max(regression_matrix[:,1]), epoch)

                print("PCA: Latent unit", np.argmax(regression_matrix[:,2]), "is related to Numberb and r_squared is:",np.max(regression_matrix[:,2]))
                # log the best raw r squared value
                self.writer.add_scalar("PCA_Numberb", np.max(regression_matrix[:,2]), epoch)

            # set the figure size
                fig, axs = plt.subplots(latent_dim, 3, figsize=(9, 6*latent_dim))

                c = 1
                for i in range(3):
                    for j in range(latent_dim):
                        # Scatter plot
                        axs[j, i].scatter(x_transform[j, :], labels[i, :])
                        axs[j, i].set_title(f"dim{j} label{i}")

                        # Calculate and plot regression line
                        slope, intercept, r_value, p_value, std_err = linregress(x_transform[j, :], labels[i, :])
                        line = slope * x_transform[j, :] + intercept
                        axs[j, i].plot(x_transform[j, :], line, color='red')

                        # Calculate R-squared value
                        r_squared = r_value ** 2

                        # Add R-squared value as text annotation
                        axs[j, i].text(0.5, 0.9, f'R-squared: {r_squared:.4f}', horizontalalignment='center', verticalalignment='center',
                                    transform=axs[j, i].transAxes, bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.3'))

                plt.tight_layout()

                self.writer.add_figure("PCA_result", fig, epoch)
                fig.savefig(os.path.join(self.writer.log_dir, f"PCA_result_epoch_{epoch}.png"))
                plt.close(fig)
    def kl_weight_warmup(self, step, kl_weight_begin, kl_weight_end, warmup_steps):
        """
        Adjusts the KL divergence weight during training to prevent posterior collapse.

        Args:
            step (int): Current training step.
            kl_weight_begin (float): Initial KL divergence weight.
            kl_weight_end (float): Final KL divergence weight.
            warmup_steps (int): Number of steps for the KL divergence weight to reach its final value.

        Returns:
            float: Updated KL divergence weight.
        """
        kl_weight = min(kl_weight_end, kl_weight_begin + (kl_weight_end - kl_weight_begin) * (step / warmup_steps))

        return kl_weight
    @staticmethod
    def _plot_point_cloud(points):
        """
        Plots the point cloud.

        Args:
            points (torch.Tensor): Point cloud to plot.

        Returns:
            matplotlib.figure.Figure: Figure of the plotted point cloud.
        """
        
        # if 2d, by checking the shape
        if points.shape[0] == 2:
        
            fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
            ax.scatter(points[0, :], points[1, :], c='red', s=50, marker='o')
            ax.set_facecolor('black')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.axis('off')
            return fig

        if points.shape[0] == 3:
            fig, ax = plt.subplots(figsize=(8, 8), facecolor='black', subplot_kw={'projection': '3d'})
            ax.scatter(points[0, :], points[1, :], points[2, :], c='red', s=50, marker='o')
            ax.set_facecolor('black')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.axis('off')
            return fig
        
        else:
            raise ValueError("Points should have shape (N, 2) or (N, 3)")
# use a cli to load the config file
import argparse
import yaml

parser = argparse.ArgumentParser(description='Train a PointNetVAE model.')
parser.add_argument('--config', type=str, default='config/config_3d_exp.yaml', help='Path to the config file.')
args = parser.parse_args()

file = args.config

# load file 
with open(file) as file:
    config = yaml.safe_load(file)




# print the config
num_files = config['num_files']
max_points = config['max_points']
input_dim = config['input_dim']
output_dim = config['output_dim']
batch_size = config['batch_size']
latent_dim = config['latent_dim']
kl_weight_begin = config['kl_weight_begin']
kl_weight_end = config['kl_weight_end']
augment = config['augment']
use_tnet = config['use_tnet']
modality = config['modality']
csv_dir = config['csv_dir']


for key, value in config.items():
    print(key, ' : ', value)
# Create data loaders
train_dataloader = create_dataloader(csv_dir,num_files, max_points, batch_size, augment=augment, shuffle = True, modality = modality, order_shuffle=True)
val_dataloader = create_dataloader(csv_dir, 16, max_points, batch_size, augment=False, shuffle = False, modality = modality, order_shuffle=False)
analysis_dataloader = create_dataloader(csv_dir, num_files, max_points, batch_size = 1, augment=False, shuffle = False, modality = modality, order_shuffle=False)
#train_indices = list(range(int(len(train_dataloader) * 0.8)))
#val_indices = list(range(int(len(train_dataloader) * 0.8), len(train_dataloader)))
#train_loader = DataLoader(Subset(train_dataloader.dataset, train_indices), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#val_loader = DataLoader(Subset(val_dataloader.dataset, val_indices), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model = PointNetVAE(latent_dim, max_points, use_tnet=use_tnet, input_dim=input_dim, output_dim=output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

log_name = f'{modality}/max_points_{max_points}_bs_{batch_size}_latent_dim_{latent_dim}_kl_weight_{kl_weight_end}/{time.time()}'

trainer = Trainer(model, train_dataloader, val_dataloader, analysis_dataloader,optimizer, loss_function,
                  log_dir=f'./logs/{log_name}', kl_weight_end=kl_weight_end, modality = modality,
                  kl_weight_begin = kl_weight_begin, warmup_steps=1000)
trainer.train(epochs=200)


