"""
Demo: Training an MLP with Adam and Stable Softmax
==================================================

Demonstrates:
1. Training an MLP on a classification dataset using Cross-Entropy loss.
2. Adam optimizer vs. vanilla SGD dynamics.
3. Benchmark comparing Analytical Autograd vs. Numerical Differentiation.
"""

import math
import random
import time
import sys
import os

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from micrograd.engine import Value
from micrograd.nn import MLP, Adam, cross_entropy, softmax


def run_classification_demo():
    print("=" * 60)
    print("DEMO: Training MLP on 4-Class Classification using Adam & Softmax")
    print("=" * 60)

    # 4 training points in 2D space, each belonging to one of 4 classes
    X = [
        [2.0, 3.0],    # Class 0 (top-right)
        [-2.0, 3.0],   # Class 1 (top-left)
        [-2.0, -3.0],  # Class 2 (bottom-left)
        [2.0, -3.0],   # Class 3 (bottom-right)
    ]
    Y = [0, 1, 2, 3]

    # MLP with 2 inputs, hidden layer of 8 neurons with GELU/Tanh, and 4 output logits
    model = MLP(2, [8, 4])
    optimizer = Adam(model.parameters(), lr=0.05)

    print(f"Model Architecture: {model}")
    print(f"Total Parameters  : {len(model.parameters())}\n")

    # Training loop
    for step in range(40):
        # Forward pass: compute total cross-entropy loss over the dataset
        total_loss = Value(0.0)
        correct = 0

        for x, y_true in zip(X, Y):
            logits = model(x)
            loss = cross_entropy(logits, y_true)
            total_loss = total_loss + loss

            # Check predicted class
            probs = softmax(logits)
            pred_class = max(range(len(probs)), key=lambda i: probs[i].data)
            if pred_class == y_true:
                correct += 1

        loss_avg = total_loss * (1.0 / len(X))

        # Backward pass
        optimizer.zero_grad()
        loss_avg.backward()

        # Optimizer update step
        optimizer.step()

        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step+1:2d} | Loss: {loss_avg.data:.4f} | Accuracy: {correct}/{len(X)} ({correct/len(X)*100:.0f}%)")

    print("\nFinal Predictions:")
    for i, (x, y_true) in enumerate(zip(X, Y)):
        logits = model(x)
        probs = [f"{p.data:.3f}" for p in softmax(logits)]
        pred = max(range(len(probs)), key=lambda idx: float(probs[idx]))
        print(f"  Input: {x} -> True Class: {y_true} | Predicted: {pred} | Probabilities: {probs}")
    print()


def run_benchmark():
    print("=" * 60)
    print("BENCHMARK: Analytical Autograd vs. Numerical Differentiation")
    print("=" * 60)
    print(f"{'Num Params (N)':<16} | {'Analytical Backprop':<20} | {'Numerical Diff (2N)':<20} | {'Speedup':<10}")
    print("-" * 75)

    param_sizes = [5, 10, 20, 40]
    h = 1e-6

    for N in param_sizes:
        raw_vals = [0.1 * (i + 1) for i in range(N)]

        def evaluate(p):
            total = Value(0.0)
            for i, val in enumerate(p):
                total = total + val * (0.05 * (i + 1))
            return total.tanh() * (p[0] ** 2 + 1.0) / (p[1].exp() + 0.5)

        # Time analytical backprop
        reps = 30
        t0 = time.perf_counter()
        for _ in range(reps):
            params = [Value(v) for v in raw_vals]
            out = evaluate(params)
            out.backward()
        t_analytical = (time.perf_counter() - t0) / reps

        # Time numerical differentiation (2 * N forward passes)
        t0 = time.perf_counter()
        for _ in range(reps):
            for i in range(N):
                p_plus = [Value(v + (h if j == i else 0.0)) for j, v in enumerate(raw_vals)]
                p_minus = [Value(v - (h if j == i else 0.0)) for j, v in enumerate(raw_vals)]
                f_plus = evaluate(p_plus).data
                f_minus = evaluate(p_minus).data
                _ = (f_plus - f_minus) / (2.0 * h)
        t_numerical = (time.perf_counter() - t0) / reps

        speedup = t_numerical / t_analytical if t_analytical > 0 else 0.0
        print(f"{N:<16} | {t_analytical*1000:8.3f} ms           | {t_numerical*1000:8.3f} ms           | {speedup:6.1f}x")

    print("-" * 75)
    print("Key Takeaway: Numerical diff scales as O(N * FLOPs), while Backprop is O(1) passes.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_classification_demo()
    run_benchmark()
