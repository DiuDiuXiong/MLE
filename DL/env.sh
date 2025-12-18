# 1. Install the Python package installer (pip) if not already available
#    (Often installed with python3, but good to check)
sudo apt update
sudo apt install python3-pip -y

# 2. Install core dependencies for your NumPy MNIST project
#    You need NumPy for the math and Matplotlib for visualizing results later.
python3 -m pip install numpy matplotlib

# 3. Install the specific PyTorch version compatible with CUDA 12.1 or later
#    The official PyTorch website recommends this specific command for recent CUDA versions.
#    Note: This command is correct for CUDA 12.8 support in recent PyTorch versions.
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. (Optional but Recommended) Install Scikit-learn
#    This is useful for loading standard datasets like MNIST easily later.
python3 -m pip install scikit-learn

python3 -m pip install pandas

python3 -m pip install pyarrow fastparquet

python3 -m pip install plt
