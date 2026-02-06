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
- Navigate to your project folder
- Create virtual environment using `venv`
- Activate it (commands differ for Mac/Linux vs Windows)
- You should see (venv) in your terminal prompt

**Install VSCode:**
- Download from [code.visualstudio.com](https://code.visualstudio.com/)
- Install these essential extensions:
  - **Python** (by Microsoft) - Python language support
  - **Jupyter** (by Microsoft) - Run notebooks in VSCode
  - **GitLens** (optional but helpful) - Enhanced Git features

**Install project dependencies:**
- Make sure venv is activated
- Install dependencies using pip and the requirements.txt file

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
- `git status` - Check status
- `git add` - Stage files for commit (individual files or all changes)
- `git commit -m "message"` - Commit with message
- `git push origin main` - Push to GitHub
- `git pull origin main` - Pull latest changes from GitHub

**Your first commit:**
- Stage initial project files (requirements.txt, .gitignore, README.md)
- Commit with a descriptive message
- Push to GitHub

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
- **data/** - Dataset files (not tracked in Git): mnist_train.csv, mnist_test.csv
- **src/** - Your Python modules: data_loader.py, neural_network.py
- **notebook.ipynb** - Your experimentation notebook
- **requirements.txt** - Dependencies
- **.gitignore** - Files Git should ignore
- **README.md** - This file!

**Create the src/ directory:**
- Use mkdir command to create the src directory

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

**Create `src/data_loader.py` with the following functions:**

**Function 1: load_mnist(filepath)**
- Purpose: Load MNIST data from CSV file
- Returns: X (pixel values), y (labels)
- Implementation: Read CSV using pandas, extract labels and pixel values

**Function 2: normalize(X)**
- Purpose: Normalize pixel values from [0, 255] to [0, 1]
- Implementation: Divide X by 255.0 to scale values
- This makes training more stable

**Function 3: train_val_split(X, y, val_ratio=0.2, random_seed=42)**
- Purpose: Split data into training and validation sets
- Parameters: Feature array, label array, validation ratio, random seed
- Returns: X_train, X_val, y_train, y_val
- Implementation: Shuffle data and split according to val_ratio

**Test the functions:**
- Run the data_loader.py file to test the functions
- You should see output about shapes and ranges
- If you get errors, debug them! This is part of learning

### 3.2 Step 2: Neural Network Math

Before coding, let's understand what we're building.

**Architecture: 784 → 128 → 10**
- **Input layer**: 784 neurons (one per pixel)
- **Hidden layer**: 128 neurons (you can experiment with this)
- **Output layer**: 10 neurons (one per digit class 0-9)

**Forward Propagation:**

1. **Input to Hidden:**
   - Linear transformation: z1 = X @ W1 + b1
   - Activation function: a1 = ReLU(z1)
   - W1 shape: (784, 128)
   - b1 shape: (128,)
   - a1 shape: (n_samples, 128)

2. **Hidden to Output:**
   - Linear transformation: z2 = a1 @ W2 + b2
   - Output probabilities: a2 = Softmax(z2)
   - W2 shape: (128, 10)
   - b2 shape: (10,)
   - a2 shape: (n_samples, 10)

**Activation Functions:**

- **ReLU** (Rectified Linear Unit): Returns maximum of 0 and x
  - Introduces non-linearity
  - Simple and effective

- **Softmax**: Converts scores to probabilities
  - Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
  - Output sums to 1
  - Each output is probability of that class

**Loss Function (Cross-Entropy):**
- Formula: Loss = -sum(y_true * log(y_pred))
- Measures how wrong our predictions are
- Lower is better
- Averaged over all samples

**Backward Propagation (Gradients):**

This is where the learning happens! We compute how to adjust weights to reduce loss.

1. **Output layer gradient:**
   - dz2 = a2 - y_one_hot (Softmax + cross-entropy gradient)
   - dW2 = a1.T @ dz2 / n (Gradient for W2)
   - db2 = sum(dz2) / n (Gradient for b2)

2. **Hidden layer gradient:**
   - dz1 = (dz2 @ W2.T) * ReLU'(z1) (Chain rule)
   - dW1 = X.T @ dz1 / n (Gradient for W1)
   - db1 = sum(dz1) / n (Gradient for b1)

3. **Update rule (Gradient Descent):**
   - W = W - learning_rate * dW
   - b = b - learning_rate * db

Don't worry if this seems complex - implementing it will make it clearer!

### 3.3 Step 3: Neural Network Implementation (`src/neural_network.py`)

Now let's build it! Create `src/neural_network.py` with a NeuralNetwork class containing the following components:

**Class: NeuralNetwork**
- Architecture: 2-layer fully connected neural network (input -> hidden (ReLU) -> output (Softmax))

**1. __init__ method:**
- Initialize neural network with random weights
- Parameters: input_size (784), hidden_size (128), output_size (10), learning_rate (0.1)
- Use He initialization for weights
- Initialize biases to zeros
- Create cache for storing intermediate values

**2. Activation functions:**
- **relu(z)**: ReLU activation function (max(0, z))
- **relu_derivative(z)**: Derivative of ReLU (1 if z > 0, else 0)
- **softmax(z)**: Softmax activation for output layer (converts scores to probabilities)

**3. forward(X) method:**
- Forward propagation: compute network output
- Layer 1: Input -> Hidden (z1 = X @ W1 + b1, a1 = ReLU(z1))
- Layer 2: Hidden -> Output (z2 = a1 @ W2 + b2, a2 = Softmax(z2))
- Store intermediate values in cache
- Return output probabilities

**4. compute_loss(y_true, y_pred) method:**
- Compute cross-entropy loss
- Add small epsilon to prevent log(0)
- Return average loss

**5. backward(y_true) method:**
- Backward propagation: compute gradients
- Output layer gradients: dz2, dW2, db2
- Hidden layer gradients: dz1, dW1, db1 (using chain rule)
- Store gradients

**6. update_weights() method:**
- Update weights using computed gradients and learning rate
- W = W - learning_rate * dW, b = b - learning_rate * db

**7. Helper methods:**
- **one_hot_encode(y)**: Convert labels to one-hot encoded vectors
- **predict(X)**: Make predictions on new data
- **accuracy(y_true, y_pred_probs)**: Compute accuracy

**8. train() method:**
- Train the neural network for specified epochs
- For each epoch: forward pass, compute loss, backward pass, update weights
- Track training and validation metrics
- Return history dictionary with loss and accuracy

**Test the neural network:**
- Run the file to test with dummy data
- You should see shapes and loss values
- This validates that your implementation works correctly

### 3.4 Step 4: Training and Visualization (`notebook.ipynb`)

Now let's put it all together in an interactive notebook!

Create `notebook.ipynb` with these sections:

**Section 1: Introduction and Imports**
- Markdown cell: Describe what the notebook will do (load data, train network, evaluate, visualize)
- Code cell: Import necessary libraries (numpy, pandas, matplotlib, data_loader, NeuralNetwork)
- Set random seed for reproducibility

**Section 2: Load and Explore Data**
- Load training data from CSV file
- Normalize the pixel values
- Split into training and validation sets
- Visualize sample digits with their labels using matplotlib

**Section 3: Create and Train Neural Network**
- Create NeuralNetwork instance with appropriate hyperparameters
- Display the architecture details
- Train the network on training data
- Track training and validation metrics

**Section 4: Visualize Training Progress**
- Plot training curves showing loss over epochs
- Plot accuracy curves for both training and validation
- Display final validation accuracy

**Section 5: Test on Test Set**
- Load test data
- Make predictions using the trained model
- Compute and display test accuracy
- Visualize correct predictions
- Visualize incorrect predictions (the network's mistakes)
- Create confusion matrix to see which digits are most confused

**Section 6: Experiment!**
- Try modifying hyperparameters:
  1. Hidden layer size (try 64, 256, 512)
  2. Learning rate (try 0.01, 0.05, 0.2)
  3. Number of epochs (try 20, 100, 200)
  4. Architecture (add another hidden layer)
- Observe what changes improve or worsen accuracy

---

## Part 4: Understanding Your Code

### 4.1 Numpy Broadcasting

**What is broadcasting?**
Broadcasting allows numpy to perform operations on arrays of different shapes.

**Example 1: Add scalar to array**
- Adding a scalar (like 10) to an array [1, 2, 3]
- Result: Each element gets the scalar added to it [11, 12, 13]

**Example 2: Add 1D array to 2D array**
- Adding [10, 20, 30] to [[1, 2, 3], [4, 5, 6]]
- The 1D array is "broadcast" to each row
- Result: [[11, 22, 33], [14, 25, 36]]

**In our neural network:**
- When computing z = X @ W + b:
- X shape: (n_samples, 784)
- W shape: (784, 128)
- b shape: (1, 128) or (128,)
- Result z shape: (n_samples, 128)
- The bias b is automatically added to EACH sample!

**Why vectorization matters:**
- Slow approach (loop): Process one sample at a time
- Fast approach (vectorized): Process all samples at once using matrix operations
- Vectorized code is 10-100x faster!

### 4.2 Debugging Tips

**Print shapes everywhere:**
- Print shapes of X, W1, z1, etc. to understand data flow
- Most bugs come from shape mismatches!

**Check for NaN/Inf:**
- Check if loss becomes NaN or Inf during training
- Usually means learning rate is too high

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
