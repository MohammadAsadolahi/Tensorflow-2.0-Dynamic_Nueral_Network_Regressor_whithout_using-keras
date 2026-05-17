<div align="center">

# Dynamic Neural Network Regressor

### Pure TensorFlow 2.0 Low-Level API — No Keras Layers. No Shortcuts.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Educational-blueviolet?style=for-the-badge)]()

**Mohammad Asadolahi** — Senior Agentic AI Engineer | Agentic AI Architectures In The Wild

<br>

*A from-scratch implementation of a fully configurable feedforward neural network using TensorFlow's low-level primitives and `GradientTape` — demonstrating the raw mechanics of forward propagation and backpropagation without high-level model or layer abstractions.*

<br>

```
  Input          Hidden Layers          Output
  Layer          (Configurable)         Layer
                                      
  o---------+    +--o--+    +--o--+    +----o
       |    |     |    |     |    |
  o----+----+--o--+----+--o--+----+  y = Wx + b
       |    |     |    |     |    |
  o---------+    +--o--+    +--o--+    +----o
                                      
  x1,x2,x3    ReLU       ReLU      Linear
```

</div>

---

## Why This Exists

Most tutorials teach neural networks through Keras' `model.fit()` — a single line that hides everything interesting. **This project strips away the abstraction** to expose the raw engine underneath:

- **Manual weight initialization** with `tf.Variable`
- **Forward pass** computed as matrix multiplications + bias additions
- **Loss computation** using mean squared error from scratch
- **Backpropagation** via `tf.GradientTape` — TensorFlow's automatic differentiation engine
- **Gradient descent** applied manually to each weight and bias tensor

---

## Architecture

The `NN` class creates a **fully dynamic** neural network — you define the topology at instantiation:

```python
# Default: 3 inputs -> 3 neurons -> 2 neurons -> 1 output
nn = NN(layers=[3, 3, 2, 1])

# Custom: 5-layer deep network
nn = NN(layers=[10, 64, 32, 16, 1])

# Custom activations per layer
nn = NN(layers=[3, 4, 1], activations=[tf.nn.sigmoid, lambda x: x])
```

### Under the Hood

```
+---------------------------------------------------------------------+
|                        FORWARD PASS                                  |
|                                                                      |
|   Z(l) = W(l) . A(l-1) + B(l)       ->     A(l) = activation(Z(l)) |
|                                                                      |
+----------------------------------------------------------------------+
|                        LOSS                                          |
|                                                                      |
|   L = (1/m) sum (y_hat - y)^2        ->     Mean Squared Error      |
|                                                                      |
+----------------------------------------------------------------------+
|                        BACKWARD PASS                                 |
|                                                                      |
|   dL/dW, dL/dB = GradientTape()      ->     Auto-differentiation    |
|                                                                      |
|   W := W - alpha * dL/dW             ->     Gradient Descent        |
|   B := B - alpha * dL/dB                                            |
|                                                                      |
+----------------------------------------------------------------------+
```

---

## Quick Start

### Prerequisites

```bash
pip install tensorflow numpy
```

### Train & Predict

```python
import numpy as np
from Dynamic_NN_Regression import NN

# Define network: 3 inputs -> 3 neurons -> 2 neurons -> 1 output
nn = NN(layers=[3, 3, 2, 1])

# Training data (3 samples, 3 features each)
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

Y = np.array([11, 30, 48])

# Train for 3000 epochs
nn.fit(X=X, Y=Y, epoch=3000, lr=0.0005)

# Predict on training data
print(nn.predict(X).numpy())
```

---

## Training Convergence

Example training run on the included dataset (3 samples, `lr=0.0005`, 3000 epochs):

```
Epoch       Loss
-------------------------------
    0       1391.48       Starting
  100         63.79       Converging
  500         22.86
 1000          1.56
 1500          0.159
 2000          0.063       Plateauing
 2500          0.056
 2900          0.056       Converged
```

**Training predictions:** `[[11.16, 29.66, 48.16]]` vs targets `[11, 30, 48]` — close fit on training data.

> **Note:** With only 3 training samples this is a pedagogical demonstration, not a production model. Results vary across runs due to random weight initialization.

---

## API Reference

### `NN(layers, activations)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layers` | `list[int]` | `[3,3,2,1]` | Network topology — each element defines neuron count per layer |
| `activations` | `list[callable]` | `None` | Activation functions per layer. Defaults to ReLU (hidden) + Linear (output) |

### `nn.fit(X, Y, epoch, lr)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | — | Input features matrix |
| `Y` | `np.ndarray` | — | Target values |
| `epoch` | `int` | — | Number of training iterations |
| `lr` | `float` | `0.001` | Learning rate |

### `nn.predict(X)` -> `tf.Tensor`

Returns the forward pass output for input `X`.

### `nn.loss(Y_pred, Y_target)` -> `tf.Tensor`

Computes mean squared error between predictions and targets.

---

## Project Structure

```
.
├── Dynamic_NN_Regression.py          # Core implementation — the NN class
├── Tensorflow_2_0_*.ipynb            # Interactive Jupyter notebook walkthrough
└── README.md
```

---

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| **Weight Initialization** | Uniform random in `[-1, 1]` via `tf.random.uniform` |
| **Forward Propagation** | Sequential `Z = W*X + B` -> `A = activation(Z)` through all layers |
| **Automatic Differentiation** | `tf.GradientTape` records ops for gradient computation |
| **Gradient Descent** | Manual `W := W - lr*grad_W` update using `assign_sub` |
| **Dynamic Architecture** | Arbitrary depth and width — defined at instantiation |
| **Activation Functions** | Pluggable per-layer — ReLU, Sigmoid, Tanh, or custom |

---

## Author

**Mohammad Asadolahi** — Senior Agentic AI Engineer

Focus: Agentic AI Architectures In The Wild

---
