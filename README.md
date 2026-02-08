# MNIST OCR from Scratch

A neural network built from scratch using only NumPy to recognize handwritten digits from the MNIST dataset. No ML frameworks — just matrix math, forward propagation, backpropagation, and gradient descent.

## Project Structure

```
OCR-for-MNIST/
├── data/
│   └── mnist_test.csv        # MNIST dataset (CSV format)
├── src/
│   ├── data_loader.py        # Data loading and preprocessing
│   └── neural_network.py     # Neural network implementation
├── train/
│   └── train.py              # Training script
├── test/
│   ├── test.py               # Data exploration / debugging
│   └── array_data.csv        # Sample test data
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python train/train.py
```

This loads the MNIST data, trains the network for 100 iterations, and prints predictions alongside ground truth labels with an accuracy score.

## Architecture

3-layer fully connected network:

```
Input (784) → Hidden 1 (128, ReLU) → Hidden 2 (10, ReLU) → Output (10, Softmax)
```

- **Input**: 784 neurons (28x28 pixel images flattened)
- **Hidden layer 1**: 128 neurons with ReLU activation
- **Hidden layer 2**: 10 neurons with ReLU activation
- **Output**: 10 neurons with Softmax (one per digit class 0-9)

Weights are initialized using He initialization. Biases are initialized to zero.

## How It Works

### Data Loading (`src/data_loader.py`)

- `load_mnist(filepath)` — Reads the MNIST CSV, splits into training and test sets (first 1000 rows for test, remainder for training)
- `normalise(data_x)` — Scales pixel values from [0, 255] to [0, 1]

### Neural Network (`src/neural_network.py`)

The `neural_network` class implements:

- `generate_weights()` / `generate_biases()` — Initialize parameters
- `forward_prop(data)` — Forward pass through all 3 layers, caching intermediate values
- `backward_prop(output, labels)` — Computes gradients via backpropagation using the chain rule
- `update_params()` — Applies gradient descent to update weights and biases
- `predict(test_data, test_labels)` — Runs forward pass on test data and computes accuracy
- `train(data, labels, iterations)` — Full training loop: forward → backward → update

Standalone activation functions:
- `ReLU(nn_layer)` — max(0, x)
- `softmax(nn_layer)` — Converts logits to class probabilities

### Training (`train/train.py`)

Orchestrates the full pipeline:
1. Load MNIST data
2. Normalize pixel values
3. Create network (784 → 128 → 10 → 10, learning rate 0.1)
4. Train for 100 iterations
5. Predict on test set and print accuracy

## Dependencies

- `numpy` — Matrix operations
- `pandas` — CSV data loading
- `matplotlib` — Plotting
