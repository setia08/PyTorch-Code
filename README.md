# PyTorch Basics

A hands-on notebook covering the fundamentals of [PyTorch](https://pytorch.org/) tensors and operations — a personal learning reference for getting started with tensor computation before moving on to neural networks.

## Contents

`PyTorch_Basics.ipynb` walks through:

- **Setup** — checking the PyTorch version and GPU/CUDA availability
- **Creating Tensors** — `empty`, `zeros`, `ones`, `rand`, `manual_seed`, `tensor`, `arange`, `linspace`, `eye`, `full`
- **Tensor Shape** — `shape`, `reshape`, and the `*_like` family (`empty_like`, `zeros_like`, `ones_like`, `rand_like`)
- **Data Types** — inspecting and converting tensor `dtype`s with `.to()`
- **Mathematical Operations** — scalar arithmetic on tensors
- **Element-wise Operations** — arithmetic, `neg`, `abs`, `round`, `ceil`, `floor`, `clamp` between tensors
- **Reduction Operations** — `sum`, `mean`, `median`, `prod`, `std`, `var`, `argmax`, `argmin`
- **Matrix Operations** — `matmul`, `dot`, `transpose`, `det`, `inverse`, comparison operators
- **Special Functions** — `log`, `exp`, `sqrt`, `sigmoid`, `softmax`, `relu`, and in-place variants (`add_`, `relu_`)

## Requirements

- Python 3.x
- [PyTorch](https://pytorch.org/get-started/locally/)
- Jupyter Notebook / JupyterLab

Install dependencies:

```bash
pip install torch jupyter
```

## Usage

Clone the repo and launch the notebook:

```bash
git clone https://github.com/setia08/PyTorch-Code.git
cd PyTorch-Code
jupyter notebook PyTorch_Basics.ipynb
```

## License

This project is for personal learning and reference purposes.
