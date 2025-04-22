import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

class TNet(nn.Module):
    """
    T-Net module for learning a transformation matrix to align input point clouds.
    
    Args:
        k (int): The dimension of the input point cloud.
    """
    def __init__(self, k):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.iden = torch.eye(k).flatten().unsqueeze(0)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        iden = self.iden.repeat(batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetEncoder(nn.Module):
    """
    PointNet encoder for encoding point clouds into a latent space representation.
    
    Args:
        latent_dim (int): The dimension of the latent space.
        input_dim (int): The dimension of the input point cloud.
        use_tnet (bool): Whether to use the T-Net module.
    """
    def __init__(self, latent_dim, input_dim=2, use_tnet=True):
        super(PointNetEncoder, self).__init__()
        self.use_tnet = use_tnet
        self.input_dim = input_dim
        if self.use_tnet:
            self.tnet = TNet(k=input_dim)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3_mean = nn.Linear(256, latent_dim)
        self.fc3_logvar = nn.Linear(256, latent_dim)
       # self.pooling = nn.MaxPool1d(1)
       # make a parallel conv for max pooling feature 
        self.conv1_max = nn.Conv1d(input_dim, 64, 1)
        self.bn1_max = nn.BatchNorm1d(64)
        self.conv2_max = nn.Conv1d(64, 128, 1)
        self.bn2_max = nn.BatchNorm1d(128)
        self.conv3_max = nn.Conv1d(128, 1024, 1)
        self.bn3_max = nn.BatchNorm1d(1024) 

    def forward(self, x, mask):
        if self.use_tnet:
            trans = self.tnet(x)
            x = torch.bmm(trans, x)

        if True:
            x_mean = F.relu(self.bn1(self.conv1(x)))
            x_mean = F.relu(self.bn2(self.conv2(x_mean)))
            x_mean = F.relu(self.bn3(self.conv3(x_mean)))
            if False:
                x_max = F.relu(self.bn1_max(self.conv1_max(x)))
                x_max = F.relu(self.bn2_max(self.conv2_max(x_max)))
                x_max = F.relu(self.bn3_max(self.conv3_max(x_max)))
                x_max = torch.max(x_max, 2)[0]
        #without bn
        if False:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        
        
        #x = x

        #x = torch.max(x, 2)[0]
        # use mean pooling instead of max pooling
        x_mean = torch.mean(x_mean, 2)
        
        
        
        x = x_mean#+0.01*x_max
        # adaptive max pooling 
        #x2 = self.pooling(x).squeeze(-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc3_mean(x)
        logvar = self.fc3_logvar(x)
        return mean, logvar

class PointNetDecoder(nn.Module):
    """
    PointNet decoder for reconstructing point clouds from the latent space representation.
    
    Args:
        latent_dim (int): The dimension of the latent space.
        num_points (int): The number of points in the output point cloud.
        output_dim (int): The dimension of the output point cloud.
    """
    def __init__(self, latent_dim, num_points, output_dim=3):
        super(PointNetDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, num_points * output_dim)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
                                  
        self.num_points = num_points
        self.output_dim = output_dim

    def forward(self, z, mask):
        if False:
            x = F.relu(self.bn1(self.fc1(z)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
        if True:
            x = F.tanh(self.fc1(z))
            x = F.tanh(self.fc2(x))
            x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, self.output_dim, self.num_points)
        #x = x * mask.unsqueeze(1).float()
        return x

class PointNetVAE(nn.Module):
    """
    PointNet Variational Autoencoder for point clouds.
    
    Args:
        latent_dim (int): The dimension of the latent space.
        num_points (int): The number of points in the output point cloud.
        input_dim (int): The dimension of the input point cloud.
        output_dim (int): The dimension of the output point cloud.
        use_tnet (bool): Whether to use the T-Net module.
    """
    def __init__(self, latent_dim, num_points, input_dim=2, output_dim=2, use_tnet=False, variational=True):
        super(PointNetVAE, self).__init__()
        self.encoder = PointNetEncoder(latent_dim, input_dim, use_tnet)
        self.decoder = PointNetDecoder(latent_dim, num_points, output_dim)
        self.variational = variational
       
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, mask):
        mean, logvar = self.encoder(x, mask)
        if self.variational:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        recon_x = self.decoder(z, mask)
        return recon_x, mean, logvar

class PointNetDecoderV2(nn.Module):
    """
    Enhanced PointNet decoder for reconstructing point clouds from the latent space representation.
    
    Args:
        latent_dim (int): The dimension of the latent space.
        num_points (int): The number of points in the output point cloud.
        output_dim (int): The dimension of the output point cloud.
    """
    def __init__(self, latent_dim, num_points, output_dim=3):
        super(PointNetDecoderV2, self).__init__()
        
        # Fully connected layers with Leaky ReLU activation
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, num_points * output_dim)
        
        # Batch normalization layers to stabilize training
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Residual connection layer for enhanced feature propagation
        self.fc_res = nn.Linear(1024, 1024)
        
        # 1x1 Convolutional layers for feature transformation
        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1)
        
        # Self-attention mechanism for capturing global dependencies
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=4)
        
        self.num_points = num_points
        self.output_dim = output_dim

    def forward(self, z, mask):
        # Fully connected layers with tanh and Batch Normalization
        x = F.tanh((self.fc1(z)))
        x = F.tanh((self.fc2(x)))
        x = F.tanh((self.fc3(x)))
        
        # Adding residual connection to assist in gradient flow
        x_res = F.tanh(self.fc_res(x))
        x = x + x_res  # Residual connection

        # Reshaping for 1x1 convolution (Feature transformation)
        x = x.unsqueeze(2)  # Add a dimension for 1x1 Conv
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = x.squeeze(2)  # Remove the extra dimension
        
        # Applying self-attention mechanism (if applicable)
        x = x.unsqueeze(0)  # Add batch dimension for attention
        x, _ = self.attention(x, x, x)
        x = x.squeeze(0)  # Remove batch dimension after attention

        # Final fully connected layer to generate point cloud
        x = self.fc4(x)
        x = x.view(-1, self.output_dim, self.num_points)
        x = x * mask.unsqueeze(1).float()
        return x


class PointNetVAEV2(nn.Module):
    """
    PointNet Variational Autoencoder for point clouds.
    This is an enhanced version using more advanced techniques in
    both encoder and decoder.
    
    Args:
        latent_dim (int): The dimension of the latent space.
        num_points (int): The number of points in the output point cloud.
        input_dim (int): The dimension of the input point cloud.
        output_dim (int): The dimension of the output point cloud.
        use_tnet (bool): Whether to use the T-Net module.
    """
    def __init__(self, latent_dim, num_points, input_dim=2, output_dim=2, use_tnet=False):
        super(PointNetVAEV2, self).__init__()
        self.encoder = PointNetEncoder(latent_dim, input_dim, use_tnet=False)
        self.decoder = PointNetDecoderV2(latent_dim, num_points, output_dim)
       
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, mask):
        mean, logvar = self.encoder(x, mask)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decoder(z, mask)
        return recon_x, mean, logvar