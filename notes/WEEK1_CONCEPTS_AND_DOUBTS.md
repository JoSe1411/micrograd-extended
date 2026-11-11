# Week 1: Scalar Autograd, Computational Graphs & Systems Engineering
## Complete Learner's Notes, Concepts & Doubts Solved

> **Context:** Phase 1 (Week 1) of the AI Systems & Performance Engineering Roadmap.  
> **Repository:** `nano-autograd` (`micrograd-extended`)  
> **Author:** Learning in Public Proof-of-Work

---

## Table of Contents
1. [Computational Graph & Value Node Architecture](#1-computational-graph--value-node-architecture)
2. [Python Data Model & Operator Overloading Protocol](#2-python-data-model--operator-overloading-protocol)
3. [Loss Functions & Gradient Descent Dynamics](#3-loss-functions--gradient-descent-dynamics)
4. [Deconstructing Activations: tanh, ReLU, and GELU](#4-deconstructing-activations-tanh-relu-and-gelu)
5. [How GELU and Softmax Work Together](#5-how-gelu-and-softmax-work-together)
6. [The Softmax "Max Trick" (Numerical Stability)](#6-the-softmax-max-trick-numerical-stability)
7. [Numerical Differentiation vs. Analytical Autograd](#7-numerical-differentiation-vs-analytical-autograd)
8. [The Systems Deep-Dive: Compute vs. Memory-Bandwidth (Week 5 Setup)](#8-the-systems-deep-dive-compute-vs-memory-bandwidth-week-5-setup)
9. [Optimizers: Vanilla SGD vs. Adam](#9-optimizers-vanilla-sgd-vs-adam)
10. [The Next Frontier: Scalar Autograd vs. Tensor Autograd](#10-the-next-frontier-scalar-autograd-vs-tensor-autograd)

---

## 1. Computational Graph & Value Node Architecture

### What does a `Value` node store?
In a reverse-mode automatic differentiation Directed Acyclic Graph (DAG), each scalar node requires:
- `data`: The raw float computed during the forward pass.
- `grad`: The derivative of the scalar objective (Loss) with respect to this node ($\frac{\partial L}{\partial x}$). Initialized to `0.0`.
- `_children` / `_prev`: A set of parent nodes that generated this value, enabling reverse topological traversal.
- `_op`: A string representing the operation (`'+'`, `'*'`, `'tanh'`) for visualization.
- `label`: An optional string identifier (e.g. `'w1'`, `'x1'`) for visual debugging.

### Doubt 1.1: Why the leading underscore on `_op` and `_prev`?
In Python, a leading single underscore (`_variable`) is a convention indicating **internal implementation details / metadata**. These are used internally by the DAG traversal (`trace()`) and visualization (`draw_dot()`), not meant to be modified directly by the user.

### Doubt 1.2: Why `self._backward = lambda: None`?
Leaf nodes (like input features $x$ and trainable weights $w$) have no parent operations. Without an initial dummy function, walking the graph backwards and calling `node._backward()` would raise an `AttributeError`. Setting `_backward` to `lambda: None` turns leaf nodes into safe no-ops.

### Doubt 1.3: Why accumulate gradients with `+=` instead of `=`?
If a single variable is used multiple times in an expression (e.g., $f(x) = x + x$ or $y = x \cdot x$), the multivariate chain rule states:
$$\frac{\partial L}{\partial x} = \sum_{i} \frac{\partial L}{\partial \text{out}_i} \frac{\partial \text{out}_i}{\partial x}$$
Using `=` would overwrite the gradient from the first branch with the second. Using `+=` correctly accumulates gradients across all downstream paths.

---

## 2. Python Data Model & Operator Overloading Protocol

### Doubt 2.1: What are `__truediv__`, `__radd__`, and `__neg__`?
- **`__truediv__` (Division `/`):** In Python 2, `/` was mapped to `__div__`. In Python 3, standard floating-point division is always `__truediv__` (while `//` is `__floordiv__`).
- **`__neg__` (Unary Minus `-x`):** Operates on a single operand. Implementing `-self` as `self * -1` allows binary subtraction (`a - b`) to be built naturally as `a + (-b)`.
- **`__radd__` / `__rmul__` (Reflected / Right-side Operators):** Needed when a raw Python scalar is on the left side of an expression (e.g., `2 * x` or `1 + x`).

### Doubt 2.2: How does Python know to apply `__truediv__` when `/` is used?
When Python executes `a / b`:
1. It looks at the left operand `a` and calls `type(a).__truediv__(a, b)`.
2. If `a` is a built-in `int` or `float` that doesn't recognize our custom `Value` class, it returns the sentinel `NotImplemented`.
3. Python then falls back to the right operand `b` and calls `type(b).__rtruediv__(b, a)`.
4. If neither handles it, Python raises `TypeError: unsupported operand type(s) for /`.

### Doubt 2.3: Reusing Power for Division
Rather than writing dedicated quotient rule derivative logic ($\frac{u'v - uv'}{v^2}$), we decompose division:
$$\frac{a}{b} = a \cdot b^{-1}$$
By implementing `__pow__` ($x^n$ with derivative $n x^{n-1}$) and `__mul__`, division and its derivatives emerge automatically.

---

## 3. Loss Functions & Gradient Descent Dynamics

### Doubt 3.1: Why sum the loss instead of averaging it?
$$\text{Sum Loss} = \sum_{i=1}^N L_i \quad \text{vs.} \quad \text{Mean Loss} = \frac{1}{N} \sum_{i=1}^N L_i$$
- Summing and averaging produce the **exact same gradient direction**.
- They only differ by a constant scale factor of $\frac{1}{N}$.
- Multiplying the loss by $\frac{1}{N}$ scales all gradients by $\frac{1}{N}$, which is equivalent to dividing the learning rate by $N$: $\eta_{\text{eff}} = \frac{\eta}{N}$.
- In small toy scripts with fixed datasets, `sum()` is simpler. In production mini-batch training, `mean()` is standard so the gradient scale is invariant to batch size changes.

### Doubt 3.2: What does moving in the negative gradient direction actually mean?
- The derivative $\frac{\partial L}{\partial w}$ indicates the slope of the loss curve with respect to weight $w$:
  - If $\frac{\partial L}{\partial w} > 0$: Increasing $w$ increases the loss. (Therefore, we must **decrease** $w$).
  - If $\frac{\partial L}{\partial w} < 0$: Increasing $w$ decreases the loss. (Therefore, we must **increase** $w$).
- The gradient vector $\nabla L$ always points in the direction of **steepest ascent** (maximum increase).
- Therefore, subtracting the gradient:
  $$w \leftarrow w - \eta \frac{\partial L}{\partial w}$$
  is mathematically guaranteed locally to move in the direction of **steepest descent**, minimizing the loss.

---

## 4. Deconstructing Activations: tanh, ReLU, and GELU

### 4.1 Deconstructing tanh
Instead of implementing $\tanh(x)$ as an opaque black box, it can be decomposed using its exponential definition:
$$\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$$
```python
e = (2 * x).exp()
out = (e - 1) / (e + 1)
```

### 4.2 The Dying ReLU Problem
- **ReLU:** $f(x) = \max(0, x)$ with derivative $f'(x) = 1$ if $x > 0$ else $0$.
- **The Failure Mode:** If large gradients knock a neuron's weights such that it outputs negative values for all inputs, its gradient becomes permanently $0.0$. The neuron "dies" and cannot recover.

### 4.3 GELU (Gaussian Error Linear Unit)
Used in modern Transformers (GPT-2/3/4, BERT, LLaMA). It weights inputs by their probability under a standard Gaussian distribution:
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$
- **Curvature:** As $x \to -0.5$, the function dips smoothly to $\approx -0.17$ and curves back up.
- **Continuous Derivative:**
  $$\frac{d}{dx} \text{GELU}(x) = \Phi(x) + x \cdot \phi(x) = \Phi(x) + x \cdot \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$$
- Non-zero gradients everywhere eliminate the dying neuron problem and accelerate Transformer convergence.

---

## 5. How GELU and Softmax Work Together

| Dimension | GELU | Softmax |
| :--- | :--- | :--- |
| **Location** | **Hidden Layers** (inside the network) | **Output Layer** (or Attention heads) |
| **Scope** | **Element-wise** (operates on 1 scalar at a time) | **Vector-wide** (requires sum of all exponents) |
| **Purpose** | Adds non-linear curvature for feature extraction | Normalizes unconstrained scores into probabilities |
| **Output Range** | $\approx [-0.17, +\infty)$ | $(0.0, 1.0)$, with $\sum p_i = 1.0$ |

### Pipeline in a Classifier:
$$\text{Input } \mathbf{x} \longrightarrow \underbrace{\text{Linear}_1 \to \text{GELU}}_{\text{Feature Extraction}} \longrightarrow \underbrace{\text{Linear}_2}_{\text{Logits } z} \longrightarrow \underbrace{\text{Softmax}}_{\text{Probabilities } p} \longrightarrow \text{Cross-Entropy Loss}$$

---

## 6. The Softmax "Max Trick" (Numerical Stability)

### The Problem: IEEE 754 Float Overflow
In standard 64-bit floats:
$$e^{709} \approx 8.2 \times 10^{307} \quad \text{but} \quad e^{710} \implies \textbf{OverflowError: math range error}$$
In untrained networks with random initialization, logits can easily exceed $800$, crashing the forward pass.

### The Mathematical Fix
Subtracting a constant $c = \max(\mathbf{z})$ from all logits:
$$\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i} \cdot e^{-c}}{\sum_j (e^{z_j} \cdot e^{-c})} = \frac{e^{-c} \cdot e^{z_i}}{e^{-c} \cdot \sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
- **Mathematical equivalence:** The factor $e^{-c}$ cancels out exactly.
- **Stability guarantee:** The largest exponent becomes $z_{\max} - z_{\max} = 0$, and $e^0 = 1.0$. All other exponents are negative ($e^{\le 0} \in (0, 1]$). Overflow is impossible.

---

## 7. Numerical Differentiation vs. Analytical Autograd

### The Finite Difference Quotient
To verify analytical backprop, we compute central finite differences:
$$\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + h\mathbf{e}_i) - f(\mathbf{x} - h\mathbf{e}_i)}{2h} \quad (h = 10^{-6})$$

### Relative Error Metric
$$\text{rel\_error} = \frac{|g_{\text{analytical}} - g_{\text{numerical}}|}{\max(|g_{\text{analytical}}|, |g_{\text{numerical}}|, 10^{-8})}$$
Our test suite verifies that across all operations (`+`, `-`, `*`, `/`, `**`, `exp`, `log`, `tanh`, `gelu`), the relative error is $< 10^{-10}$.

---

## 8. The Systems Deep-Dive: Compute vs. Memory-Bandwidth (Week 5 Setup)

### The Benchmark Comparison

```text
Num Params (N)   | Analytical Backprop  | Numerical Diff (2N)  | Speedup   
---------------------------------------------------------------------------
5                |    0.035 ms          |    0.630 ms          |   18.0x
10               |    0.062 ms          |    0.897 ms          |   14.4x
20               |    0.111 ms          |    2.935 ms          |   26.4x
40               |    0.169 ms          |    9.594 ms          |   56.9x
```

### The Architectural Tradeoff

| Property | Numerical Differentiation | Reverse-Mode Autograd (Backprop) |
| :--- | :--- | :--- |
| **Compute Complexity** | **$O(N \times \text{Forward FLOPs})$** — $2N$ full forward passes. | **$O(1 \times \text{Forward FLOPs})$** — All $N$ gradients in $\approx 2\times$ forward passes. |
| **Memory Footprint** | **$O(1)$ Minimal** — Intermediate values are discarded immediately. | **$O(\text{graph size})$ High** — Must **retain all intermediate activations** in memory. |
| **Hardware Regimes** | **Compute-Bound** (swamped by FLOPs). | **Memory-Bandwidth Bound** (reading/writing activations to HBM/DRAM). |

### Why This Matters for Modern LLMs (Week 5 Bridge)
1. On modern GPUs (H100/B200), compute (FLOPs) is practically free, but **Memory Bandwidth** (moving activations between High-Bandwidth Memory and on-chip SRAM) is the hard physical bottleneck.
2. Backpropagation trades arithmetic compute for memory bandwidth.
3. This bottleneck directly explains why modern AI systems use:
   - **Activation Checkpointing (Recomputation):** Intentionally discarding forward activations and recomputing them during backward pass (spending extra compute to save memory bandwidth).
   - **FlashAttention:** Fusing Softmax and Attention into SRAM tiles to avoid round-trips to GPU global memory.

---

## 9. Optimizers: Vanilla SGD vs. Adam

### The Ravine Problem in Vanilla SGD
In narrow valleys with steep walls and gentle floors:
- Vanilla SGD ($w \leftarrow w - \eta g$) takes massive, oscillating steps between the steep walls, while making almost zero progress along the gentle floor.

### How Adam Solves It
Maintains running averages for every parameter:
1. **First Moment $m$ (Momentum - Direction):**
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
   Averages out high-frequency oscillations across steep canyon walls.
2. **Second Moment $v$ (RMSProp - Scale / Friction):**
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   Divides updates by $\sqrt{v_t}$: damps down steps along steep walls and boosts steps along flat slopes.
3. **Bias Correction:**
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
   Corrects for initial $0.0$ initialization bias during the first few training iterations.

---

## 10. The Next Frontier: Scalar Autograd vs. Tensor Autograd

### The Limitation of Micrograd
Every single scalar number is an independent Python object with pointers, sets, and closures. A tiny $64 \times 64$ matrix multiplication requires $131{,}072$ separate Python object allocations. Python interpreter overhead dominates execution time.

### The Tensor Leap
A `Tensor` wraps a contiguous block of memory (`np.ndarray`) and uses **Vector-Jacobian Products (VJPs)**:
- **Matrix Multiplication:** $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} @ B^T$ and $\frac{\partial L}{\partial B} = A^T @ \frac{\partial L}{\partial C}$.
- **Broadcasting:** When dimensions are broadcasted in the forward pass, gradients must be **summed along the broadcasted axis** during the backward pass.
- Vectorized execution unlocks SIMD, GPU tensor cores, and orders-of-magnitude speedups.

