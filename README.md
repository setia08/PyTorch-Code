# PyTorch Basics

Hands-on notebooks covering the fundamentals of [PyTorch](https://pytorch.org/) — tensors, GPU acceleration, autograd, and a full training pipeline — a personal learning reference for building up to neural networks.

## Contents

### `PyTorch_Basics.ipynb`

- **Setup** — checking the PyTorch version and GPU/CUDA availability
- **Creating Tensors** — `empty`, `zeros`, `ones`, `rand`, `manual_seed`, `tensor`, `arange`, `linspace`, `eye`, `full`
- **Tensor Shape** — `shape`, `reshape`, and the `*_like` family (`empty_like`, `zeros_like`, `ones_like`, `rand_like`)
- **Data Types** — inspecting and converting tensor `dtype`s with `.to()`
- **Mathematical Operations** — scalar arithmetic on tensors
- **Element-wise Operations** — arithmetic, `neg`, `abs`, `round`, `ceil`, `floor`, `clamp` between tensors
- **Reduction Operations** — `sum`, `mean`, `median`, `prod`, `std`, `var`, `argmax`, `argmin`
- **Matrix Operations** — `matmul`, `dot`, `transpose`, `det`, `inverse`, comparison operators
- **Special Functions** — `log`, `exp`, `sqrt`, `sigmoid`, `softmax`, `relu`, and in-place variants (`add_`, `relu_`)

### `PyTorch_On_GPU.ipynb`

- **CUDA Basics** — checking `torch.cuda.is_available()`, creating tensors directly on the GPU, and moving tensors between CPU and GPU with `.to(device)`
- **CPU vs GPU Benchmarking** — timing `matmul` on large tensors to compare performance
- **Reshaping** — `unsqueeze` and `squeeze` for adding/removing dimensions
- **NumPy Interop** — converting between tensors and NumPy arrays with `.numpy()` and `torch.from_numpy()`

### `Auto_Grad_Back.ipynb`

- **Manual Gradients** — computing gradients of a binary cross-entropy loss by hand via the chain rule
- **Autograd** — the same computation using `requires_grad=True` and `loss.backward()`
- **Gradient Management** — disabling tracking with `requires_grad_(False)` and `torch.no_grad()` to avoid gradient accumulation across passes

### `pytorch_training_pipeline.ipynb`

- **Data Preparation** — loading the Breast Cancer dataset with pandas, splitting with `train_test_split`, scaling with `StandardScaler`, and label encoding
- **Tensor Conversion** — converting NumPy arrays to PyTorch tensors with `torch.from_numpy`
- **Model From Scratch** — a simple neural network (`SimpleNN`) implemented with raw tensors, manual forward pass, and a loss function
- **Training Loop** — gradient descent over multiple epochs using forward pass, loss computation, and manual weight updates

### `NN_Module.ipynb` / `NN_Module_updated.ipynb`

- **`nn.Module` Basics** — defining a model by subclassing `nn.Module`, using `nn.Linear` and `nn.Sigmoid`, and running a forward pass
- **Model Summary** — inspecting the model with `torchinfo.summary`

### `pytorch_training_pipeline_using_nn_module.ipynb` / `pytorch_training_pipeline_using_nn_module_updated.ipynb`

- Same end-to-end pipeline as `pytorch_training_pipeline.ipynb`, rebuilt with `torch.nn`:
  - Model defined via `nn.Module` with `nn.Linear` + `nn.Sigmoid`
  - Loss via `nn.BCELoss()` instead of a hand-written function
  - Training loop using `torch.optim.SGD` for parameter updates
  - Evaluation accuracy on the held-out test set

### `Full_pytorch_training_pipeline_using_nn_module.ipynb`

- The complete pipeline with batching via `torch.utils.data.Dataset` and `DataLoader`:
  - Custom `Dataset` class wrapping the feature/label tensors
  - Mini-batch training loop over a `DataLoader`
  - Model, loss (`nn.BCELoss`), and optimizer (`SGD`) as above
  - Evaluation loop computing accuracy over `test_loader` batches

### `ANN_using_Pytorch.ipynb`

- **Image Classification on MNIST** — a full artificial neural network pipeline on the MNIST digit dataset:
  - Loading MNIST, visualizing a 4x4 grid of sample digits
  - Scaling pixel values, `CustomDataset` + `DataLoader` for batching
  - A multi-layer `nn.Sequential` model (`MyNN`) trained with `nn.CrossEntropyLoss` and SGD
  - Evaluation accuracy on the test set
  - Falls back to `sklearn.datasets.fetch_openml('mnist_784')` when run outside Google Colab (no bundled `sample_data` CSV)

### `ANN_using_Pytorch_with_Dropout_and_others.ipynb`

- The same MNIST ANN pipeline extended with regularization and GPU support:
  - `nn.BatchNorm1d` and `nn.Dropout` layers between linear layers
  - `weight_decay` (L2 regularization) on the optimizer
  - GPU/CPU device handling — model and batches moved to `device` automatically
  - `pin_memory=True` on the `DataLoader`s for faster host-to-GPU transfer

### `ANN_using_Pytorch_with_Optuna.ipynb`

- Hyperparameter tuning of the MNIST ANN with [Optuna](https://optuna.org/):
  - A configurable `MyNN` built dynamically from `num_hidden_layers` / `neurons_per_layer`
  - An `objective()` function searching over layer count, width, dropout, batch size, learning rate, weight decay, epochs, and optimizer (`Adam` / `SGD` / `RMSprop`)
  - `optuna.create_study(direction="maximize")` + `study.optimize(...)` to find the best-performing configuration

### `transfer_learning_fashion_mnist_pytorch_gpu (1).ipynb`

- **Transfer Learning with VGG16** — fine-tuning a pretrained `torchvision.models.vgg16` on MNIST digits:
  - Grayscale images converted to 3-channel and resized/cropped/normalized to ImageNet stats for VGG16
  - Convolutional base frozen (`requires_grad = False`), classifier head replaced with a custom `nn.Sequential`
  - Training/evaluation loop on GPU (or CPU fallback)
  - Falls back to `sklearn.datasets.fetch_openml('mnist_784')` when run outside Google Colab

### `RNN.ipynb`

- **Simple RNN for Question Answering** — a single-word-answer QA model:
  - Custom tokenizer, vocabulary built from a question/answer dataset
  - `nn.Embedding` + `nn.RNN` + `nn.Linear` model (`SimpleRNN`) predicting one answer token
  - Training loop and a `predict()` function for inference
  - Falls back to a small built-in sample Q&A set when the bundled Colab dataset isn't present

### `pytorch_lstm_next_word_predictor.ipynb`

- **LSTM Next-Word Prediction** — trained on an inline text document (no external dataset):
  - Tokenization with `nltk`, vocabulary and padded training sequences
  - `nn.Embedding` + `nn.LSTM` + `nn.Linear` model (`LSTMModel`)
  - Training loop with GPU support, a `prediction()` function, and multi-token text generation
  - Accuracy evaluation over the full dataset

## Requirements

- Python 3.x
- [PyTorch](https://pytorch.org/get-started/locally/)
- Jupyter Notebook / JupyterLab
- pandas, scikit-learn (for the training pipeline notebooks)
- torchinfo (for `NN_Module.ipynb` / `NN_Module_updated.ipynb`)
- matplotlib (for the MNIST ANN notebooks)
- optuna (for `ANN_using_Pytorch_with_Optuna.ipynb`)
- torchvision, pillow (for the transfer learning notebook)
- nltk (for `pytorch_lstm_next_word_predictor.ipynb`)

Install dependencies:

```bash
pip install torch jupyter pandas scikit-learn torchinfo matplotlib optuna torchvision pillow nltk
```

## Usage

Clone the repo and launch the notebook:

```bash
git clone https://github.com/setia08/PyTorch-Code.git
cd PyTorch-Code
jupyter notebook
```

## License

This project is for personal learning and reference purposes.
