import torch

def test_basic_gpu():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0))

if __name__ == "__main__":
    test_basic_gpu()


