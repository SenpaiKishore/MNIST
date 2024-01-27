
# MNIST Classification using PyTorch and MLX

This repo contains a clean implementaion of Image classification([MNIST Dataset](http://yann.lecun.com/exdb/mnist/)) using [PyTorch](https://github.com/pytorch/pytorch) and [Apple's MLX](https://github.com/ml-explore/mlx) frameworks.

This was implemented just for educational purpose. Only constructive criticism appreciated.

## Quick Start
**Make sure `python` is installed on your system.**

**1. Clone the repo:**
```bash
git clone https://github.com/SenpaiKishore/MNIST.git
cd MNIST
```

**2. Install required libraries:**
```bash
pip install -r requirements.txt
```

**4A. To run PyTorch implementation, run:**
```bash
python3 torch/main.py
```

**4B. To run PyTorch implementation, run:**
```bash
python3 mlx/main.py
```

## TODO
- [x] Remove `max_pooling2d` from PyTorch Implementation
- [ ] Fix the error in MLX implementation: `ValueError: [conv] Expect the input channels in the input and weight array to match but got shapes - input: (1,1,28,28) and weight: (32,6,6,1)`
- [ ] Add [Tinygrad](https://github.com/tinygrad/tinygrad) implementation
- [ ] Add MNIST from scratch (No frameworks used except `numpy`)

## Authors

**Kishore Gunalan** - [@SenpaiKishore](https://twitter.com/senpaikishore)

