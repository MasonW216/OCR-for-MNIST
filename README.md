# MNIST OCR from Scratch - A Learning Journey

Welcome! This project will teach you how to build a neural network from scratch using only numpy, pandas, and matplotlib. By the end, you'll understand how neural networks work at a fundamental level AND be comfortable with professional development tools like VSCode and Git.

This is designed for beginners who want to truly understand machine learning, not just use high-level frameworks.

## Table of Contents
- [Part 1: Getting Started (Development Environment)](#part-1-getting-started)
- [Part 2: Project Setup](#part-2-project-setup)
- [Part 3: Building the Neural Network](#part-3-building-the-neural-network)
- [Part 4: Understanding Your Code](#part-4-understanding-your-code)
- [Part 5: Resources for Learning](#part-5-resources-for-learning)

---

## Part 1: Getting Started

### 1.1 Environment Setup

**Install Python:**
- Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
- Verify installation: `python --version` or `python3 --version`

**Set up a virtual environment** (keeps project dependencies isolated):
```bash
# Navigate to your project folder
cd /path/to/OCR-for-MNIST

# Create virtual environment
python -m venv venv

# Activate it:
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

**Install VSCode:**
- Download from [code.visualstudio.com](https://code.visualstudio.com/)
- Install these essential extensions:
  - **Python** (by Microsoft) - Python language support
  - **Jupyter** (by Microsoft) - Run notebooks in VSCode
  - **GitLens** (optional but helpful) - Enhanced Git features

**Install project dependencies:**
```bash
# Make sure venv is activated!
pip install -r requirements.txt
```

### 1.2 VSCode Basics

**Opening your project:**
- File → Open Folder → Select `OCR-for-MNIST`
- This opens the entire project, letting you see all files in the sidebar

**Integrated Terminal:**
- View → Terminal (or Ctrl + \`)
- Run Python files: `python src/data_loader.py`
- Your virtual environment should auto-activate in VSCode terminal

**Running Python files:**
- Open any `.py` file
- Press **F5** (or click Run → Start Debugging)
- Or right-click in the file and select "Run Python File in Terminal"

**Jupyter Notebooks in VSCode:**
- Create or open `.ipynb` files
- Click on code cells and press **Shift + Enter** to run
- Add cells with the + button
- Switch between Code and Markdown cells

**Useful keyboard shortcuts:**
- `Ctrl + P` (or `Cmd + P` on Mac): Quick file search
- `Ctrl + Shift + P`: Command palette (search for any VSCode command)
- `Ctrl + /`: Comment/uncomment code
- `Shift + Alt + F`: Auto-format code

**Debugging basics:**
- Click left of line number to set a **breakpoint** (red dot)
- Press F5 to start debugging
- Code pauses at breakpoints - hover over variables to see values
- Use Step Over (F10) and Step Into (F11) to control execution
- This is super helpful when your code isn't working!

### 1.3 Git & GitHub Fundamentals

**Git concepts:**
- **Repository (repo)**: Your project folder with version history
- **Commit**: A snapshot of your code at a point in time
- **Branch**: A separate line of development (we'll mostly use `main`)
- **Remote**: The GitHub copy of your repository

**Using Git in VSCode:**
- Open the **Source Control** panel (Ctrl + Shift + G)
- See changed files, stage them (+ icon), write commit message, click ✓ to commit

**Basic Git commands** (in terminal):
```bash
# Check status
git status

# Stage files for commit
git add filename.py
git add .  # stages all changes

# Commit with message
git commit -m "Add data loader function"

# Push to GitHub
git push origin main

# Pull latest changes from GitHub
git pull origin main
```

**Your first commit:**
```bash
git add requirements.txt .gitignore README.md
git commit -m "Initial project setup with dependencies and gitignore"
git push origin main
```

**Making meaningful commits:**
- Commit often (after each small working change)
- Write clear messages: "Add normalization to data loader" not "stuff"
- Each commit should be one logical change

---

## Part 2: Project Setup

### 2.1 Project Structure

**Why organize code into modules?**
- **Maintainability**: Easy to find and update code
- **Reusability**: Import functions across files
- **Professionalism**: Industry standard practice
- **Testing**: Can test individual components

**Our structure:**
```
OCR-for-MNIST/
├── data/               # Dataset files (not tracked in Git)
│   ├── mnist_train.csv
│   └── mnist_test.csv
├── src/                # Your Python modules
│   ├── data_loader.py
│   └── neural_network.py
├── notebook.ipynb      # Your experimentation notebook
├── requirements.txt    # Dependencies
├── .gitignore          # Files Git should ignore
└── README.md           # This file!
```

**Create the src/ directory:**
```bash
mkdir src
```

### 2.2 MNIST Data

**Understanding the CSV format:**
- First row: headers (`label,1x1,1x2,...,28x28`)
- Each row after: one handwritten digit
- `label` column: the actual digit (0-9)
- 784 pixel columns: grayscale values 0-255 (28×28 = 784 pixels)

**Add your training data:**
- Place `mnist_train.csv` in the `data/` folder
- The `.gitignore` is already set up to not track these large files in Git

**Why we .gitignore data files:**
- CSV files are large (tens of MB)
- Git is designed for code, not data
- Keeps your repository size small and fast
- Data can be downloaded separately

---

## Part 3: Building the Neural Network

This is where the real learning happens! You'll implement everything from scratch.

### 3.1 Step 1: Data Loading (`src/data_loader.py`)

**What you'll learn:**
- Reading CSV files with pandas
- Converting to numpy arrays
- Normalizing data
- Splitting data for validation

**Create `src/data_loader.py` and add this code:**

```python
"""
Data loading and preprocessing for MNIST dataset.
"""

import pandas as pd
import numpy as np


def load_mnist(filepath):
    """
    Load MNIST data from CSV file.

    Args:
        filepath: Path to CSV file (e.g., 'data/mnist_train.csv')

    Returns:
        X: numpy array of shape (n_samples, 784) - pixel values
        y: numpy array of shape (n_samples,) - labels (0-9)
    """
    # Read CSV using pandas
    df = pd.read_csv(filepath)

    # Extract labels (first column)
    y = df['label'].values

    # Extract pixel values (all columns except first)
    X = df.drop('label', axis=1).values

    print(f"Loaded {len(X)} samples")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y


def normalize(X):
    """
    Normalize pixel values from [0, 255] to [0, 1].

    Args:
        X: numpy array of pixel values

    Returns:
        X_normalized: values scaled to [0, 1]
    """
    # TODO: Divide X by 255.0 to scale to [0, 1]
    # This makes training more stable
    return X / 255.0


def train_val_split(X, y, val_ratio=0.2, random_seed=42):
    """
    Split data into training and validation sets.

    Args:
        X: feature array
        y: label array
        val_ratio: fraction of data to use for validation
        random_seed: for reproducibility

    Returns:
        X_train, X_val, y_train, y_val
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Get number of samples
    n_samples = len(X)
    n_val = int(n_samples * val_ratio)

    # Create shuffled indices
    indices = np.random.permutation(n_samples)

    # Split indices
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Split data
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    return X_train, X_val, y_train, y_val


# Test the functions
if __name__ == "__main__":
    # Load data
    X, y = load_mnist('data/mnist_test.csv')

    # Normalize
    X = normalize(X)
    print(f"Pixel value range: [{X.min():.2f}, {X.max():.2f}]")

    # Split
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    print("\nData loading successful!")
```

**Run it to test:**
```bash
python src/data_loader.py
```

You should see output about shapes and ranges. If you get errors, debug them! This is part of learning.

### 3.2 Step 2: Neural Network Math

Before coding, let's understand what we're building.

**Architecture: 784 → 128 → 10**
- **Input layer**: 784 neurons (one per pixel)
- **Hidden layer**: 128 neurons (you can experiment with this)
- **Output layer**: 10 neurons (one per digit class 0-9)

**Forward Propagation:**

1. **Input to Hidden:**
   ```
   z1 = X @ W1 + b1        # Linear transformation
   a1 = ReLU(z1)           # Activation function
   ```
   - `W1` shape: (784, 128)
   - `b1` shape: (128,)
   - `a1` shape: (n_samples, 128)

2. **Hidden to Output:**
   ```
   z2 = a1 @ W2 + b2       # Linear transformation
   a2 = Softmax(z2)        # Output probabilities
   ```
   - `W2` shape: (128, 10)
   - `b2` shape: (10,)
   - `a2` shape: (n_samples, 10)

**Activation Functions:**

- **ReLU** (Rectified Linear Unit): `max(0, x)`
  - Introduces non-linearity
  - Simple and effective

- **Softmax**: Converts scores to probabilities
  ```
  softmax(x_i) = exp(x_i) / sum(exp(x_j))
  ```
  - Output sums to 1
  - Each output is probability of that class

**Loss Function (Cross-Entropy):**
```
Loss = -sum(y_true * log(y_pred))
```
- Measures how wrong our predictions are
- Lower is better
- Averaged over all samples

**Backward Propagation (Gradients):**

This is where the learning happens! We compute how to adjust weights to reduce loss.

1. **Output layer gradient:**
   ```
   dz2 = a2 - y_one_hot       # Softmax + cross-entropy gradient
   dW2 = a1.T @ dz2 / n       # Gradient for W2
   db2 = sum(dz2) / n         # Gradient for b2
   ```

2. **Hidden layer gradient:**
   ```
   dz1 = (dz2 @ W2.T) * ReLU'(z1)    # Chain rule
   dW1 = X.T @ dz1 / n                # Gradient for W1
   db1 = sum(dz1) / n                 # Gradient for b1
   ```

3. **Update rule (Gradient Descent):**
   ```
   W = W - learning_rate * dW
   b = b - learning_rate * db
   ```

Don't worry if this seems complex - the code will make it clearer!

### 3.3 Step 3: Neural Network Implementation (`src/neural_network.py`)

Now let's build it! Create `src/neural_network.py`:

```python
"""
Simple 2-layer neural network implemented from scratch.
"""

import numpy as np


class NeuralNetwork:
    """
    A 2-layer fully connected neural network for classification.
    Architecture: input -> hidden (ReLU) -> output (Softmax)
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10, learning_rate=0.1):
        """
        Initialize the neural network with random weights.

        Args:
            input_size: Number of input features (784 for MNIST)
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output classes (10 for digits 0-9)
            learning_rate: Step size for gradient descent
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights with small random values
        # He initialization for ReLU
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # For storing intermediate values during forward pass
        self.cache = {}

    def relu(self, z):
        """ReLU activation function: max(0, z)"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """Derivative of ReLU: 1 if z > 0, else 0"""
        return (z > 0).astype(float)

    def softmax(self, z):
        """
        Softmax activation function for output layer.
        Converts scores to probabilities.
        """
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward propagation: compute network output.

        Args:
            X: Input data of shape (n_samples, input_size)

        Returns:
            a2: Output probabilities of shape (n_samples, output_size)
        """
        # Layer 1: Input -> Hidden
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)

        # Layer 2: Hidden -> Output
        z2 = a1 @ self.W2 + self.b2
        a2 = self.softmax(z2)

        # Store values for backward pass
        self.cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

        return a2

    def compute_loss(self, y_true, y_pred):
        """
        Compute cross-entropy loss.

        Args:
            y_true: True labels (one-hot encoded) shape (n_samples, output_size)
            y_pred: Predicted probabilities shape (n_samples, output_size)

        Returns:
            loss: Average cross-entropy loss
        """
        n_samples = y_true.shape[0]

        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        loss = -np.sum(y_true * np.log(y_pred + epsilon)) / n_samples

        return loss

    def backward(self, y_true):
        """
        Backward propagation: compute gradients.

        Args:
            y_true: True labels (one-hot encoded) shape (n_samples, output_size)
        """
        n_samples = y_true.shape[0]

        # Get cached values from forward pass
        X = self.cache['X']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        a2 = self.cache['a2']

        # Output layer gradients
        # For softmax + cross-entropy, gradient is simply: prediction - truth
        dz2 = a2 - y_true
        dW2 = (a1.T @ dz2) / n_samples
        db2 = np.sum(dz2, axis=0, keepdims=True) / n_samples

        # Hidden layer gradients (using chain rule)
        dz1 = (dz2 @ self.W2.T) * self.relu_derivative(z1)
        dW1 = (X.T @ dz1) / n_samples
        db1 = np.sum(dz1, axis=0, keepdims=True) / n_samples

        # Store gradients
        self.grads = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2
        }

    def update_weights(self):
        """Update weights using computed gradients and learning rate."""
        self.W1 -= self.learning_rate * self.grads['dW1']
        self.b1 -= self.learning_rate * self.grads['db1']
        self.W2 -= self.learning_rate * self.grads['dW2']
        self.b2 -= self.learning_rate * self.grads['db2']

    def one_hot_encode(self, y):
        """
        Convert labels to one-hot encoded vectors.

        Args:
            y: Labels array of shape (n_samples,)

        Returns:
            one_hot: One-hot encoded array of shape (n_samples, output_size)
        """
        n_samples = y.shape[0]
        one_hot = np.zeros((n_samples, self.output_size))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot

    def train(self, X_train, y_train, X_val, y_val, epochs=10, verbose=True):
        """
        Train the neural network.

        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            verbose: Whether to print progress

        Returns:
            history: Dictionary with training/validation loss and accuracy
        """
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        # Convert labels to one-hot
        y_train_one_hot = self.one_hot_encode(y_train)
        y_val_one_hot = self.one_hot_encode(y_val)

        for epoch in range(epochs):
            # Forward pass
            y_train_pred = self.forward(X_train)

            # Compute loss
            train_loss = self.compute_loss(y_train_one_hot, y_train_pred)

            # Backward pass
            self.backward(y_train_one_hot)

            # Update weights
            self.update_weights()

            # Compute accuracy
            train_acc = self.accuracy(y_train, y_train_pred)

            # Validation
            y_val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val_one_hot, y_val_pred)
            val_acc = self.accuracy(y_val, y_val_pred)

            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Print progress
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
                      f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        return history

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Input data of shape (n_samples, input_size)

        Returns:
            predictions: Predicted class labels of shape (n_samples,)
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def accuracy(self, y_true, y_pred_probs):
        """
        Compute accuracy.

        Args:
            y_true: True labels
            y_pred_probs: Predicted probabilities

        Returns:
            accuracy: Fraction of correct predictions
        """
        y_pred = np.argmax(y_pred_probs, axis=1)
        return np.mean(y_pred == y_true)


# Test the neural network
if __name__ == "__main__":
    # Create dummy data
    X = np.random.randn(100, 784)
    y = np.random.randint(0, 10, size=100)

    # Create network
    nn = NeuralNetwork(hidden_size=64, learning_rate=0.01)

    # Test forward pass
    output = nn.forward(X)
    print(f"Output shape: {output.shape}")
    print(f"Output sum (should be ~1 per sample): {output.sum(axis=1)[:5]}")

    # Test training
    y_one_hot = nn.one_hot_encode(y)
    loss = nn.compute_loss(y_one_hot, output)
    print(f"Initial loss: {loss:.4f}")

    print("\nNeural network implementation successful!")
```

**Test it:**
```bash
python src/neural_network.py
```

This runs the test code at the bottom. You should see shapes and loss values.

### 3.4 Step 4: Training and Visualization (`notebook.ipynb`)

Now let's put it all together in an interactive notebook!

Create `notebook.ipynb` with these cells:

**Cell 1 (Markdown):**
```markdown
# MNIST Neural Network Training

In this notebook, we'll:
1. Load and visualize MNIST data
2. Train our neural network from scratch
3. Evaluate performance
4. Visualize results
```

**Cell 2 (Code):**
```python
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_mnist, normalize, train_val_split
from src.neural_network import NeuralNetwork

# Set random seed for reproducibility
np.random.seed(42)
```

**Cell 3 (Markdown):**
```markdown
## 1. Load and Explore Data
```

**Cell 4 (Code):**
```python
# Load training data
X_train_full, y_train_full = load_mnist('data/mnist_train.csv')
X_train_full = normalize(X_train_full)

# Split into train and validation
X_train, X_val, y_train, y_val = train_val_split(X_train_full, y_train_full, val_ratio=0.2)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
```

**Cell 5 (Code):**
```python
# Visualize some digits
def plot_digits(X, y, n=10):
    """Plot first n digits with their labels."""
    fig, axes = plt.subplots(1, n, figsize=(15, 2))
    for i in range(n):
        axes[i].imshow(X[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'Label: {y[i]}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

plot_digits(X_train, y_train, n=10)
```

**Cell 6 (Markdown):**
```markdown
## 2. Create and Train Neural Network
```

**Cell 7 (Code):**
```python
# Create neural network
nn = NeuralNetwork(
    input_size=784,
    hidden_size=128,
    output_size=10,
    learning_rate=0.1
)

print("Neural Network Architecture:")
print(f"Input: 784 neurons (28x28 pixels)")
print(f"Hidden: {nn.hidden_size} neurons (ReLU)")
print(f"Output: 10 neurons (Softmax)")
print(f"Learning rate: {nn.learning_rate}")
```

**Cell 8 (Code):**
```python
# Train the network
print("Training...")
history = nn.train(
    X_train, y_train,
    X_val, y_val,
    epochs=50,
    verbose=True
)
print("\nTraining complete!")
```

**Cell 9 (Markdown):**
```markdown
## 3. Visualize Training Progress
```

**Cell 10 (Code):**
```python
# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax1.plot(history['train_loss'], label='Training Loss', linewidth=2)
ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss over Training')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy curves
ax2.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy over Training')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
```

**Cell 11 (Markdown):**
```markdown
## 4. Test on Test Set
```

**Cell 12 (Code):**
```python
# Load test data
X_test, y_test = load_mnist('data/mnist_test.csv')
X_test = normalize(X_test)

# Make predictions
y_pred = nn.predict(X_test)

# Compute accuracy
test_acc = np.mean(y_pred == y_test)
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
```

**Cell 13 (Code):**
```python
# Visualize some predictions
def plot_predictions(X, y_true, y_pred, n=10, correct=True):
    """
    Plot predictions.
    If correct=True, show correct predictions. Else, show mistakes.
    """
    # Find indices
    if correct:
        indices = np.where(y_true == y_pred)[0]
        title_prefix = "Correct"
    else:
        indices = np.where(y_true != y_pred)[0]
        title_prefix = "Wrong"

    if len(indices) < n:
        n = len(indices)

    indices = indices[:n]

    fig, axes = plt.subplots(1, n, figsize=(15, 2))
    if n == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        axes[i].imshow(X[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'True: {y_true[idx]}\nPred: {y_pred[idx]}')
        axes[i].axis('off')

    plt.suptitle(f"{title_prefix} Predictions", fontsize=14)
    plt.tight_layout()
    plt.show()

# Show correct predictions
print("Correct predictions:")
plot_predictions(X_test, y_test, y_pred, n=10, correct=True)

# Show mistakes
print("\nIncorrect predictions (the network's mistakes):")
plot_predictions(X_test, y_test, y_pred, n=10, correct=False)
```

**Cell 14 (Code):**
```python
# Confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Add text annotations
for i in range(10):
    for j in range(10):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')

plt.xticks(range(10))
plt.yticks(range(10))
plt.tight_layout()
plt.show()
```

**Cell 15 (Markdown):**
```markdown
## 5. Experiment!

Try modifying these hyperparameters and see how they affect performance:

1. **Hidden layer size**: Change `hidden_size` (try 64, 256, 512)
2. **Learning rate**: Try 0.01, 0.05, 0.2
3. **Number of epochs**: Try 20, 100, 200
4. **Architecture**: Add another hidden layer (you'll need to modify the code)

What changes improve accuracy? What makes it worse?
```

---

## Part 4: Understanding Your Code

### 4.1 Numpy Broadcasting

**What is broadcasting?**
Broadcasting allows numpy to perform operations on arrays of different shapes.

```python
# Example 1: Add scalar to array
x = np.array([1, 2, 3])
y = x + 10  # Adds 10 to each element
# Result: [11, 12, 13]

# Example 2: Add 1D array to 2D array
X = np.array([[1, 2, 3],
              [4, 5, 6]])
y = np.array([10, 20, 30])
Z = X + y  # y is "broadcast" to each row
# Result: [[11, 22, 33],
#          [14, 25, 36]]
```

**In our neural network:**
```python
# When we do: z = X @ W + b
# X shape: (n_samples, 784)
# W shape: (784, 128)
# b shape: (1, 128) or (128,)
# Result z shape: (n_samples, 128)

# The bias b is automatically added to EACH sample!
```

**Why vectorization matters:**
```python
# Slow (loop):
for i in range(n_samples):
    output[i] = X[i] @ W + b  # One sample at a time

# Fast (vectorized):
output = X @ W + b  # All samples at once!

# Vectorized code is 10-100x faster!
```

### 4.2 Debugging Tips

**Print shapes everywhere:**
```python
print(f"X shape: {X.shape}")
print(f"W1 shape: {self.W1.shape}")
print(f"z1 shape: {z1.shape}")
```
Most bugs come from shape mismatches!

**Check for NaN/Inf:**
```python
if np.isnan(loss) or np.isinf(loss):
    print("WARNING: Loss is NaN or Inf!")
    # Usually means learning rate is too high
```

**Use VSCode debugger:**
1. Set breakpoint in `forward()` method
2. Run in debug mode (F5)
3. Inspect `z1`, `a1`, `z2`, `a2` shapes
4. Step through line by line

**Common errors and fixes:**

| Error | Cause | Fix |
|-------|-------|-----|
| Shape mismatch | Wrong matrix dimensions | Print all shapes |
| Loss is NaN | Learning rate too high | Reduce learning rate |
| Loss not decreasing | Learning rate too low or bug | Increase LR or debug |
| Low accuracy | Network too small or not enough epochs | Increase hidden size or train longer |

### 4.3 Hyperparameter Tuning

**Learning Rate:**
- **Too high**: Loss oscillates or becomes NaN
- **Too low**: Learning is very slow
- **Just right**: Smooth decrease in loss
- Try: 0.001, 0.01, 0.1, 1.0

**Hidden Layer Size:**
- **Too small**: Can't learn complex patterns (underfitting)
- **Too large**: Might overfit, slower training
- **Rule of thumb**: Start with 128-256 for MNIST
- Try: 64, 128, 256, 512

**Number of Epochs:**
- **Too few**: Model hasn't converged yet
- **Too many**: Might overfit (val accuracy drops)
- **Watch the curves**: Stop when validation accuracy plateaus
- Try: 20, 50, 100, 200

**Pro tip:** Use the validation set to tune hyperparameters, then report final results on the test set.

---

## Part 5: Resources for Learning

### 5.1 Python & Numpy

- **Numpy Documentation**: https://numpy.org/doc/stable/
- **Numpy Quickstart Tutorial**: https://numpy.org/doc/stable/user/quickstart.html
- **Python Virtual Environments**: https://docs.python.org/3/tutorial/venv.html
- **Real Python - Numpy Tutorial**: https://realpython.com/numpy-tutorial/

### 5.2 Machine Learning Theory

- **3Blue1Brown - Neural Networks** (YouTube): https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
  - Best visual explanation of neural networks!

- **Stanford CS231n**: http://cs231n.stanford.edu/
  - Comprehensive computer vision course

- **Understanding Backpropagation Visually**: https://colah.github.io/posts/2015-08-Backprop/
  - Clear explanation of chain rule

- **Neural Networks from Scratch in Python**: https://nnfs.io/
  - Book that goes even deeper

### 5.3 VSCode & Git

- **VSCode Python Tutorial**: https://code.visualstudio.com/docs/python/python-tutorial
- **VSCode Tips and Tricks**: https://code.visualstudio.com/docs/getstarted/tips-and-tricks
- **Git Handbook** (GitHub): https://guides.github.com/introduction/git-handbook/
- **Learn Git Branching** (Interactive): https://learngitbranching.js.org/
- **Oh My Git!** (Git learning game): https://ohmygit.org/

### 5.4 Next Steps

**After mastering this project:**

1. **Add more models:**
   - Logistic regression (simpler baseline)
   - 3-layer neural network (deeper network)
   - Experiment with different activation functions

2. **Implement optimizations:**
   - Mini-batch gradient descent (faster training)
   - Momentum (smoother convergence)
   - Learning rate scheduling

3. **Backdoor Learning (your research goal!):**
   - Implement trigger injection
   - Poison training data
   - Measure attack success rate
   - Explore defense mechanisms
   - Papers to read:
     - "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain"
     - "Neural Cleanse: Identifying and Mitigating Backdoor Attacks"

4. **Advanced Topics:**
   - Convolutional Neural Networks (CNNs)
   - Data augmentation
   - Regularization (L2, dropout)
   - Transfer learning

---

## Tips for Success

1. **Type the code yourself** - Don't copy-paste! You learn by doing.
2. **Experiment** - Change values, break things, see what happens.
3. **Read error messages** - They usually tell you exactly what's wrong.
4. **Use print statements liberally** - Understand what your code is doing.
5. **Commit often** - Save your progress frequently with Git.
6. **Ask questions** - Use Google, Stack Overflow, ChatGPT when stuck.
7. **Be patient** - Understanding neural networks takes time!

---

## Expected Results

With the default hyperparameters:
- **Training accuracy**: ~95-98%
- **Validation accuracy**: ~92-95%
- **Test accuracy**: ~92-94%

If you're getting much lower (< 85%), check:
- Is your data normalized?
- Is your learning rate reasonable?
- Did you train for enough epochs?
- Are there bugs in forward/backward pass?

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'src'"**
- Make sure you're running from the project root directory
- Or use: `python -m src.neural_network`

**"FileNotFoundError: data/mnist_train.csv"**
- Make sure you've added the training data to the `data/` folder
- Check the file path is correct

**"Loss is NaN"**
- Learning rate is too high, try 0.01 or 0.001
- Check for division by zero in your code

**Accuracy stuck at ~10%**
- Network is just guessing (10 classes = 10% random guess)
- Check your learning rate isn't too small
- Verify gradients are being computed correctly

**Training is very slow**
- Make sure you're using vectorized operations (no loops)
- Try reducing hidden layer size for faster iterations
- Consider using a smaller subset of data for testing

---

## Congratulations!

By completing this project, you've:
- ✅ Built a neural network completely from scratch
- ✅ Understood forward and backward propagation
- ✅ Learned professional development tools (VSCode, Git)
- ✅ Gained intuition for how neural networks learn
- ✅ Created a foundation for researching backdoor learning

You now have the skills to implement more complex models, understand research papers, and contribute to ML security research.

**Next challenge**: Add backdoor learning capabilities to study how neural networks can be compromised!

Good luck with your learning journey! 🚀
