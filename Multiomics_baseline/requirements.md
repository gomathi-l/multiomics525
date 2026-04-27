# Tested with Python 3.10
# For GPU support, install PyTorch with CUDA first:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# For CPU only:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

scanpy>=1.10
squidpy>=1.4
anndata>=0.10
torch>=2.0
torch-geometric>=2.5
scikit-learn>=1.3
scipy>=1.12
matplotlib>=3.8
numpy>=1.24
pandas>=2.0
tqdm
seaborn

# Note: If you plan to use MAGPIE for coregistration, install its
# environment separately per https://github.com/Core-Bioinformatics/magpie
