# PCVAE: Point cloud Variational Autoencoder

PCVAE is a Point cloud variational autoencoder designed for analyzing single-molecule localization microscopy (SMLM) data. This model helps to detect continous structural heterogenity in SMLM datasets.

## 📌 Overview

Single-molecule localization microscopy (SMLM) provides super-resolution imaging at the nanometer scale. However, SMLM datasets often contain variability due to technical or biological factors. PCVAE introduces a contrastive variational autoencoder framework to:

- Preserve biologically meaningful latent features
- Enable interpretable and structured latent space analysis

## ✨ Features

- **Disentanglement**: Encourages interpretable latent dimensions.
- **Modular Codebase**: Easy to extend and integrate with your own data and models.
- **Visualization Tools**: Includes projections and binning for downstream biological analysis.

## 📦 Installation

1. **Clone the repository:**
   
   ```bash
   git clone https://github.com/Sobhanhaghparast/PCVAE.git
   cd PCVAE
   
2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate

3. Install dependencies:

   ```bash
   pip install -r requirements.txt


🚀 Usage
You can modify the configuration file and run the training script:

   ```bash
   python main.py


The input data is expected to be in CSV format with the first two columns representing x and y coordinates. Reconstructed outputs, latent spaces, and labels should be structured accordingly (see /data folder for examples).

🧠 Applications
Analyzing continuous structural heterogeneity in SMLM data

Dimensionality reduction for microscopy data

Interpretable representations in biomedical imaging


🔬 Developed by Yi zhang and Sobhan Haghparast
🧪 If you use this work in your research, please cite the repository and the associated publication when available.
