"""
micrograd.nn
============

Neural network modules, layers, and optimization routines built on top of the
Value autograd engine.

Learner's Notes & Conceptual Questions
--------------------------------------
1. Why did we sum the loss instead of averaging it?
   - Mathematically, sum(loss) and mean(loss) have the exact same gradient
     direction; they only differ by a constant scale factor of (1 / N).
   - Multiplying or dividing the total loss by N simply scales the gradients by
     (1 / N), which is equivalent to tuning the learning rate (eta / N).
   - In small toy examples with fixed batch size, sum is simpler. In production
     mini-batch SGD, mean is preferred so the loss and gradient magnitude do not
     change when the batch size changes.

2. What does moving in the negative gradient direction actually mean?
   - The derivative dL/dw tells us the rate of change of the loss with respect
     to weight w:
       * If dL/dw > 0: Increasing w INCREASES the loss. (So we must DECREASE w).
       * If dL/dw < 0: Increasing w DECREASES the loss. (So we must INCREASE w).
   - The gradient always points in the direction of steepest ASCENT.
   - Therefore, subtracting the gradient: w = w - lr * (dL/dw) always moves in
     the direction of steepest DESCENT, reducing the loss.
"""

import math
import random
from micrograd.engine import Value


class Module:
    """Base class for all neural network modules."""

    def zero_grad(self):
        """Reset gradients of all parameters to zero before backward pass."""
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        """Return a list of all Value parameters in this module."""
        return []


class Neuron(Module):
    """
    A single artificial neuron computing:
        out = activation(sum(w_i * x_i) + b)
    """

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1), label=f"w{i}") for i in range(nin)]
        self.b = Value(0.0, label="b")
        self.nonlin = nonlin

    def __call__(self, x):
        # 1. Linear combination: w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # 2. Non-linear activation (tanh for hidden layers, linear for output)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'Tanh' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """
    A layer of independent neurons operating in parallel on the same input vector.
    """

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer([{', '.join(str(n) for n in self.neurons)}])"


class MLP(Module):
    """
    Multi-Layer Perceptron (feedforward network of chained layers).
    """

    def __init__(self, nin, nouts):
        # sz concatenates input dimension with layer output dimensions: e.g. [3] + [4, 4, 1] -> [3, 4, 4, 1]
        sz = [nin] + nouts
        # Non-linear activation for hidden layers, linear for the final output layer
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP([{', '.join(str(layer) for layer in self.layers)}])"


# -----------------------------------------------------------------------------
# Classification & Optimization Utilities
# -----------------------------------------------------------------------------


def softmax(logits):
    """
    Numerically stable Softmax using the max-subtraction trick:
        softmax(z_i) = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
    Prevents float overflow when logits are large.
    """
    max_val = max(z.data for z in logits)
    counts = [(z - max_val).exp() for z in logits]
    total = sum(counts)
    return [c / total for c in counts]


def cross_entropy(logits, target_idx):
    """
    Negative Log-Likelihood (Cross-Entropy) loss:
        loss = -log(softmax(logits)[target_idx])
    """
    probs = softmax(logits)
    return -probs[target_idx].log()


class Adam:
    """
    Adaptive Moment Estimation (Adam) optimizer.
    Combines momentum (first moment m) and RMSProp scaling (second moment v)
    with startup bias correction.
    """

    def __init__(self, parameters, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0 for _ in parameters]
        self.v = [0.0 for _ in parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad == 0.0:
                continue

            # 1. Update biased 1st moment estimate (momentum)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * p.grad

            # 2. Update biased 2nd raw moment estimate (uncentered variance)
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (p.grad ** 2)

            # 3. Compute bias-corrected estimates
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            # 4. Parameter update with per-parameter adaptive learning rate
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0


