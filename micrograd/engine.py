"""
micrograd.engine
================

A scalar autograd engine built from scratch.

Learner's Notes & Architecture Thoughts
---------------------------------------
These notes document the first-principles reasoning, design questions, and
conceptual doubts encountered while building this autograd engine:

1. What does a Value node need to store?
   - `data`: The scalar float computed during the forward pass.
   - `grad`: The derivative of the final loss/output with respect to this node
     (initialized to 0.0, accumulating gradients during backprop).
   - `_children` / `_prev`: The parent nodes in the computation graph that
     produced this value, allowing backward topological traversal.
   - `_op`: The operation string (e.g. '+', '*', 'exp') for visualization.
   - `label`: An optional name for debugging and graph plotting.

   > Question: Why the leading underscore on `_op` and `_prev`?
   > Convention: It indicates internal graph metadata used by the engine and
   > visualization tools, rather than configuration meant to be modified directly.

2. Why `self._backward = lambda: None`?
   - Leaf nodes (input data and model weights) have no parents or operations to
     backprop into. Initializing `_backward` to a no-op lambda ensures that
     calling `node._backward()` during the topological walk safely does nothing
     instead of raising an AttributeError or needing an `if` check.

3. Operator Overloading & Scalar Wrapping:
   - Wrapped operands with `other if isinstance(other, Value) else Value(other)`
     so expressions like `x + 1` or `x * 2` work without manually wrapping raw numbers.
   - Implemented reflected dunder methods (`__radd__`, `__rmul__`) so that
     expressions with scalars on the left (e.g. `2 * x` or `1 + x`) work seamlessly.

4. Deconstructing tanh using Primitive Operations:
   - Rather than treating tanh as a single monolithic black-box function, we can
     decompose it into primitive operations using its mathematical definition:
       tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
   - To make `(e - 1) / (e + 1)` work, we need:
     * `exp()`: Computes e^x with local derivative d/dx(e^x) = e^x.
     * `__pow__`: Computes x^n with local derivative n * x^(n - 1).
     * `__truediv__`: In Python 3, `/` maps to `__truediv__`. We express division
       as a * (b ** -1) so we get division derivatives for free from multiplication & power.
     * `__neg__` & `__sub__`: Unary minus (-x) allows subtraction to be `self + (-other)`.

5. Python Data Model & Operator Protocol:
   - When Python encounters `a / b`:
     1. It first calls `type(a).__truediv__(a, b)`.
     2. If `a` returns `NotImplemented` (e.g. `2 / Value(...)`), Python falls
        back to the right operand: `type(b).__rtruediv__(b, a)`.
     3. If neither operand handles it, Python raises a `TypeError`.
   - By implementing `__radd__`, `__rmul__`, and `__rtruediv__`, `Value` objects
     interoperate naturally with standard Python ints and floats.
"""

import math

try:
    from graphviz import Digraph
except ModuleNotFoundError:
    Digraph = None


class Value:
    """Stores a single scalar value and its gradient for automatic differentiation."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        # Default no-op for leaf nodes (inputs/weights with no parents)
        self._backward = lambda: None
        self.children = tuple(_children)
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data:{self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        # Fallback when scalar is on the left: other + self
        return self + other

    def __neg__(self):
        # Unary minus: -self
        return self * -1

    def __sub__(self, other):
        # Subtraction via addition and negation: self + (-other)
        return self + (-other)

    def __rsub__(self, other):
        # Fallback when scalar is on the left: other - self
        return Value(other) + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        # Fallback when scalar is on the left: other * self
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # d/dx (x^n) = n * x^(n - 1)
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        # Division via power: self * (other ** -1)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        # Fallback when scalar is on the left: other / self
        return Value(other) * (self ** -1)

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            # d/dx (e^x) = e^x = out.data
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            # d/dx (ln(x)) = 1 / x
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        e = (2 * self).exp()
        return (e - 1) / (e + 1)

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Perform reverse-mode automatic differentiation from this node."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def trace(root):
    """Build the set of nodes and edges in the computation graph."""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, format="svg", rankdir="LR"):
    """
    Visualize a Value computation graph with graphviz.
    rankdir: "LR" (left-to-right) or "TB" (top-to-bottom)
    """
    if Digraph is None:
        raise ModuleNotFoundError(
            "graphviz is not installed. In a normal notebook: %pip install graphviz "
            "(and install the system `dot` binary). JupyterLite cannot run graphviz."
        )
    assert rankdir in ("LR", "TB")
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
