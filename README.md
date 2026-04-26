<div align="center">

# 🧠 Dynamic Neural Network Regressor

### Pure TensorFlow 2.0 — No Keras. No Shortcuts. Just Math.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Educational-blueviolet?style=for-the-badge)]()

<br>

*A from-scratch implementation of a fully configurable feedforward neural network using only TensorFlow's low-level primitives and `GradientTape` — demonstrating the raw mechanics of forward propagation and backpropagation without any high-level abstraction.*

<br>

```
  Input          Hidden Layers          Output
  Layer          (Configurable)         Layer
                                      
  ○─────┐    ┌──○──┐    ┌──○──┐    ┌────○
       │    │     │    │     │    │
  ○────┼────┼──○──┼────┼──○──┼────┤  ŷ = Wx + b
       │    │     │    │     │    │
  ○─────┘    └──○──┘    └──○──┘    └────○
                                      
  x₁,x₂,x₃    ReLU       ReLU      Linear
```

</div>

---

## 💡 Why This Exists

Most tutorials teach neural networks through Keras' `model.fit()` — a single line that hides everything interesting. **This project strips away the abstraction** to expose the raw engine underneath:

- **Manual weight initialization** with `tf.Variable`
- **Forward pass** computed as matrix multiplications + bias additions
- **Loss computation** using mean squared error from scratch
- **Backpropagation** via `tf.GradientTape` — TensorFlow's automatic differentiation engine
- **Gradient descent** applied manually to each weight and bias tensor

> *"The best way to understand deep learning is to build it from the ground up."*

---

## 🏗️ Architecture

The `NN` class creates a **fully dynamic** neural network — you define the topology at instantiation:

```python
# Default: 3 → 3 → 2 → 1 (input → hidden → hidden → output)
nn = NN(layers=[3, 3, 2, 1])

# Custom: 5-layer deep network
nn = NN(layers=[10, 64, 32, 16, 1])

# Custom activations per layer
nn = NN(layers=[3, 4, 1], activations=[tf.nn.sigmoid, tf.keras.activations.linear])
```

### Under the Hood

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FORWARD PASS                                 │
│                                                                     │
│   Z⁽ˡ⁾ = W⁽ˡ⁾ · A⁽ˡ⁻¹⁾ + B⁽ˡ⁾     →     A⁽ˡ⁾ = σ(Z⁽ˡ⁾)        │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        LOSS                                         │
│                                                                     │
│   ℒ = (1/m) Σ (ŷ - y)²              →     Mean Squared Error       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        BACKWARD PASS                                │
│                                                                     │
│   ∂ℒ/∂W, ∂ℒ/∂B = GradientTape()    →     Auto-differentiation     │
│                                                                     │
│   W := W - α · ∂ℒ/∂W                →     Gradient Descent         │
│   B := B - α · ∂ℒ/∂B                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow numpy
```

### Train & Predict

```python
import numpy as np
from Dynamic_NN_Regression import NN

# Define network: 3 inputs → 3 neurons → 2 neurons → 1 output
nn = NN(layers=[3, 3, 2, 1])

# Training data
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

Y = np.array([11, 30, 48])

# Train for 1000 epochs
nn.fit(X=X, Y=Y, epoch=1000, lr=0.001)

# Predict on unseen data
X_new = np.array([[10], [11], [12]])
print(nn.predict(X_new).numpy())
```

---

## 📉 Training Convergence

The network learns rapidly — loss drops by **6 orders of magnitude** within 3000 epochs:

```
Epoch       Loss              Status
─────────────────────────────────────────
    0       288.746           ████████████████████████████████  Starting
  100         0.547           █                                Converging
  500         0.053           ▏                                
 1000         0.003           ▏                                
 1500         0.000193        ▏                                Near zero
 2000         0.0000116       ▏                                
 2500         0.000000697     ▏                                
 2900         0.0000000740    ▏                                ✓ Converged
```

**Prediction on unseen input `[[10], [11], [12]]`:** `[[31.999481]]` — virtually perfect.

---

## 🧩 API Reference

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
| `lr` | `float` | `0.001` | Learning rate (α) |

### `nn.predict(X)` → `tf.Tensor`

Returns the forward pass output for input `X`.

### `nn.loss(Y_pred, Y_target)` → `tf.Tensor`

Computes mean squared error between predictions and targets.

---

## 📂 Project Structure

```
.
├── Dynamic_NN_Regression.py          # Core implementation — the NN class
├── Tensorflow_2_0_*.ipynb            # Interactive Jupyter notebook walkthrough
└── README.md                         # You are here
```

---

## 🔬 Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| **Weight Initialization** | Uniform random in `[-1, 1]` via `tf.random.uniform` |
| **Forward Propagation** | Sequential `Z = W·X + B` → `A = σ(Z)` through all layers |
| **Automatic Differentiation** | `tf.GradientTape` records ops for gradient computation |
| **Gradient Descent** | Manual `W := W - α·∇W` update using `assign_sub` |
| **Dynamic Architecture** | Arbitrary depth and width — defined at instantiation |
| **Activation Functions** | Pluggable per-layer — ReLU, Sigmoid, Tanh, or custom |

---

## 📚 Learning Resources

New to gradient descent? These are excellent starting points:

- 🎥 [Andrew Ng — Gradient Descent](https://www.youtube.com/watch?v=rIVLE3condE) — Intuitive visual walkthrough
- 📖 [TensorFlow GradientTape Guide](https://www.tensorflow.org/guide/autodiff) — Official documentation
- 📝 [Deep Learning Book (Goodfellow)](https://www.deeplearningbook.org/) — The definitive reference

---

<div align="center">

### Built with curiosity and `tf.GradientTape`

*Star ⭐ this repo if you believe the best way to learn neural networks is to build them from scratch.*

</div>
