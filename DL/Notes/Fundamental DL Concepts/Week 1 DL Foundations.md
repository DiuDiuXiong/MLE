# Week 1 — Deep Learning Foundations

# 1. Tensors and Shape Thinking

In Deep Learning (DL), mastering "shape thinking" is arguably more important than memorizing calculus. If you can track the shapes of your tensors, you can usually build the architecture.

## 1.1 What a tensor is (in DL terms, not textbook terms)



Forget the physics definition ("an object that transforms according to..."). In Deep Learning, a **Tensor** is simply a **generic n-dimensional container for numbers**.

Think of it as a generalized spreadsheet:
* **0-D Tensor (Scalar):** A single value (e.g., loss). Shape: `[]`.
* **1-D Tensor (Vector):** A single row or column. Shape: `[d]`.
* **2-D Tensor (Matrix):** A grayscale image or a spreadsheet. Shape: `[h, w]`.
* **3-D Tensor:** A stack of matrices (e.g., an RGB image). Shape: `[c, h, w]`.
* **4-D Tensor:** A batch of RGB images. Shape: `[n, c, h, w]`.

**Key Attributes in PyTorch/TensorFlow:**
1.  **Data:** The actual numbers.
2.  **Shape:** The dimensions (the most critical attribute for debugging).
3.  **Dtype:** Precision (e.g., `float32`, `float16`, `int64`).
4.  **Device:** Where the data lives (CPU vs. GPU/CUDA).

> **Note:** A tensor is rigid. Every row must have the same length; every channel must have the same resolution. You cannot have a "jagged" tensor natively.

---

## 1.2 “Shape = meaning” (why DL is mostly shape transformations)

In classical software engineering, we use variable names to denote meaning (`user_age`, `image_width`). In Deep Learning, **position in the shape determines meaning**.

If you have a tensor of shape `(64, 10, 256)`, the raw numbers tell you nothing about what is happening. You must know the semantic mapping:
* **Dim 0 (64):** Batch size (64 different samples).
* **Dim 1 (10):** Sequence length (10 words in a sentence).
* **Dim 2 (256):** Embedding dimension (each word represented by 256 features).

**The Core DL Loop:**
Deep Learning is essentially piping data through a series of geometric transformations.
1.  We reshape data to fit an operation.
2.  We apply the operation (Matrix multiplication, Convolution).
3.  We reshape the output for the next layer.

If you mess up the shape (e.g., swap logic and swap dimensions 0 and 1), you corrupt the **meaning** of the data, even if the code runs without crashing.

---

## 1.3 Canonical tensor shapes in DL

While shapes can be anything, specific domains have agreed-upon standards. Memorizing these helps you read code faster.

### Computer Vision (CV)
* **NCHW (PyTorch Default):** $(N, C, H, W)$
    * $N$: Batch Size
    * $C$: Channels (3 for RGB)
    * $H$: Height
    * $W$: Width
* **NHWC (TensorFlow/Keras Default):** $(N, H, W, C)$

### Natural Language Processing (NLP) / Time Series
* **Batch First:** $(N, L, D)$ or $(N, T, D)$
    * $N$: Batch Size
    * $L/T$: Sequence Length / Time steps
    * $D$: Feature dimension / Embedding size
* **Sequence First (Old PyTorch RNNs):** $(L, N, D)$
    * *Why?* It was slightly faster for memory access in recurrent loops, though Transformers mostly strictly use Batch First now.

### Simple Tabular (MLP)
* **Flat:** $(N, D)$
    * $N$: Samples
    * $D$: Features

---

## 1.4 Linear layer shape reasoning

The Linear Layer (or Dense/Fully Connected layer) is the workhorse of DL. It projects data from one feature space to another.

**The Operation:**
Mathematically, for a single input vector $x$ and weight matrix $W$:
$$y = xW^T + b$$

**Shape Transformation:**
Given an input batch $X$ of shape $(N, D_{in})$:

1.  **Weight Matrix ($W$):** Shape is $(D_{out}, D_{in})$.
    * *Note:* PyTorch stores weights as `(out_features, in_features)` so it can perform the dot product easily.
2.  **Bias ($b$):** Shape is $(D_{out})$.
3.  **The Calculation:**
    $$(N, D_{in}) \times (D_{in}, D_{out}) \rightarrow (N, D_{out})$$

**Reasoning Rule:**
If you want to transform a representation from size 512 to size 128:
1.  Input shape ends in 512.
2.  Layer defines `nn.Linear(512, 128)`.
3.  Output shape ends in 128.
4.  **Batch dimensions are preserved.** If input is $(N, T, 512)$, output is $(N, T, 128)$.

---

## 1.5 Broadcasting



Broadcasting is the magic that allows numpy/PyTorch to perform math on tensors of *different shapes*. It is both a powerful tool and a source of silent bugs.

### 1.5.1 The rule: align from the right, pad 1s on the left

When operating on two tensors, say $A$ and $B$, the system aligns their shapes **starting from the last dimension (the right)**.

**The Algorithm:**
1.  Align shapes on the right side.
2.  If one tensor has fewer dimensions than the other, prepend dimensions of size **1** to the smaller tensor until ranks match.
3.  Iterate through dimensions from right to left. The dimensions are compatible if:
    * They are equal.
    * **OR** one of them is 1.
4.  If a dimension is 1, the data is conceptually "stretched" (copied) to match the other dimension.

### 1.5.2 Interpreting broadcast meaning

Broadcasting implies applying an operation **across** a dimension.

* **Case A: Per-feature Bias**
    * Input: $(N, C)$ (Batch of $N$ items, $C$ features)
    * Bias: $(1, C)$
    * **Meaning:** Add the *same* bias vector to every sample in the batch. The dimension of size 1 (Batch dim) is broadcasted over.

* **Case B: Per-sample Offset**
    * Input: $(N, C)$
    * Offset: $(N, 1)$
    * **Meaning:** Add a specific scalar value to *every feature* of a specific sample. The dimension of size 1 (Feature dim) is broadcasted over.

### 1.5.3 Worked examples

**Example 1: The standard bias add**
$$A: (32, 128)$$
$$B: (128)$$
1.  Align right: $A$ is $(32, 128)$, $B$ aligns to $128$.
2.  Pad left: $B$ becomes $(1, 128)$.
3.  Compare: $32$ vs $1$ (OK, stretch $B$), $128$ vs $128$ (OK).
4.  **Result:** $(32, 128)$.

**Example 2: The "Outer Product" creation**
$$A: (3, 1)$$
$$B: (1, 3)$$
1.  Align: Both are 2D.
2.  Dim 0: $3$ vs $1$ $\rightarrow$ Result $3$.
3.  Dim 1: $1$ vs $3$ $\rightarrow$ Result $3$.
4.  **Result:** $(3, 3)$.
    * *Visual:* This creates a grid where every element of A is added/multiplied with every element of B.

**Example 3: The Failure**
$$A: (4, 3)$$
$$B: (4)$$
1.  Align right: $A$ ends in $3$, $B$ ends in $4$.
2.  $3 \neq 4$ and neither is 1.
3.  **Result:** `RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1`.

### 1.5.4 Common broadcasting mistakes and how to debug them

**1. The "Column Vector" Trap**
You predict a value $y_{pred}$ of shape $(N)$ and compare it to targets $y_{true}$ of shape $(N, 1)$.
* $(N) - (N, 1) \rightarrow (1, N) - (N, 1) \rightarrow (N, N)$ matrix!
* You calculated loss on an $N \times N$ matrix instead of a vector of size $N$. Your loss will be massive and wrong.
* **Fix:** Ensure shapes match exactly. Use `.squeeze()` or `.unsqueeze()` or `.view()`.

**2. Implicit vs Explicit**
Implicit broadcasting is concise but dangerous.
* *Dangerous:* `return x + bias` (relies on shape of x matching bias expectations).
* *Safer:* `return x + bias.view(1, -1)` (Explicitly tells the reader and the compiler you expect the bias to be a row vector).

**How to Debug:**
When in doubt, print the shapes immediately before the crash.
```python
# The Debug Print
print(f"Tensor A: {A.shape}")
print(f"Tensor B: {B.shape}")
# calculate C = A + B
```

---

# 2. Autograd and Computation Graphs

If shape thinking is the structural logic of DL, **Autograd** is the engine. It is the reason we don't have to derive complex derivatives by hand for every new network architecture we invent.

## 2.1 What autograd is (automatic differentiation)

**Autograd** stands for **Auto**matic **Grad**ient calculation.

In classical machine learning (like linear regression in closed form), we solved equations. In Deep Learning, the functions are too complex to solve. Instead, we use an iterative "hot and cold" game called Gradient Descent. To play this game, we need to know which direction to move our weights to lower the error.

Autograd is a "record keeper." As you perform forward math (addition, multiplication, ReLU), Autograd quietly watches. It records the operations in a history tape so that it can later replay the tape backward to calculate the slopes (gradients).

---

## 2.2 Computation graph: nodes, edges, and what is stored from forward pass



Deep Learning frameworks represent your code as a **Directed Acyclic Graph (DAG)**.

* **Nodes:** The operations (Functions) like `Mul`, `Add`, `ReLU`, `Conv2d`.
* **Edges:** The Tensors (Data) flowing between operations.

**The "Forward" Pass:**
When you run `output = model(input)`, two things happen simultaneously:
1.  **Calculation:** The actual numbers are computed (e.g., $2 \times 3 = 6$).
2.  **Graph Construction:** PyTorch builds a dynamic graph linking the output tensor back to the input tensor via the operation (e.g., `grad_fn=<MulBackward0>`).

**Critical Storage:**
To calculate the derivative later, the graph often needs to save parts of the forward data.
* *Example:* For $y = x^2$, the derivative is $2x$. To compute $2x$ during the backward pass, the system must **save** the value of $x$ in memory during the forward pass.
* *Implication:* This is why "training" uses much more VRAM than "inference." Inference doesn't need to store these intermediate values.

---

## 2.3 What `.backward()` actually does (reverse-mode autodiff)

When you call `loss.backward()`, you kickstart the **Reverse-Mode Automatic Differentiation**.

1.  **Start:** It starts at the `loss` node (usually a scalar).
2.  **Seed:** It implicitly creates a gradient of `1.0` for the loss (meaning $\frac{\partial Loss}{\partial Loss} = 1$).
3.  **Traverse:** It follows the `grad_fn` pointers backwards through the graph.
4.  **Chain Rule:** At every node, it multiplies the incoming gradient (from the future/output) by the local gradient (the derivative of that specific operation).
    $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$
5.  **Deposit:** When it reaches a "Leaf Tensor" (like your Weights), it deposits the calculated gradient into that tensor's `.grad` attribute.

---

## 2.4 Leaf vs non-leaf tensors

Not all tensors are created equal. Understanding this distinction saves you from `None` gradient bugs.

### 2.4.1 Why leaf tensors store `.grad`
A **Leaf Tensor** is a tensor that you created directly (not the result of an operation). In DL, these are usually your **Model Weights** and **Biases**.
* These are the inputs to the graph.
* We need their gradients to update them (optimizer step).
* PyTorch preserves their `.grad` attribute after `.backward()` finishes.

### 2.4.2 Why non-leaf tensors usually do not store `.grad`
A **Non-Leaf Tensor** (or intermediate tensor) is the result of an operation.
* *Example:* If $w$ is a weight and $x$ is input, $h = w \cdot x$ is a non-leaf tensor.
* **Memory Efficiency:** Once the backward pass flows *through* $h$ to get to $w$, we technically don't need the gradient of $h$ anymore. To save memory, PyTorch aggressively frees these intermediate gradients. If you try to access `h.grad`, you will get `None` or a warning.

### 2.4.3 When and why to use `.retain_grad()`
Sometimes (usually for debugging or visualization), you *do* want to see the gradient of an intermediate layer.
* **The Fix:** Call `h.retain_grad()` on the tensor *before* calling `.backward()`. This tells Autograd: "Please don't delete this specific gradient during cleanup."

---

## 2.5 Why gradients accumulate (the “sum of paths” idea)



One of the most confusing features for beginners is that gradients **accumulate** (add up) by default; they do not overwrite.

### 2.5.1 Multiple dependency paths and gradient addition
Imagine a variable $x$ is used in two different places in your network:
$$y = x^2$$
$$z = 3x$$
**Loss Function:** $L = y \times z$ (or $y + z$, it doesn't matter)

According to the **Multivariate Chain Rule (Total Derivative)**, the total change in $L$ with respect to $x$ is the **sum** of the changes through all paths, regardless of how $y$ and $z$ are combined later:
$$\frac{dL}{dx} = \frac{\partial L}{\partial y}\frac{dy}{dx} + \frac{\partial L}{\partial z}\frac{dz}{dx}$$

**Why Addition?**
Calculus assumes that for tiny changes, effects are **locally linear**. If you push $x$, it sends a "ripple" down path $y$ and a "ripple" down path $z$. The total effect on the Loss is the sum of these independent ripples.

Because Autograd processes the graph node-by-node, it calculates the "y-path" first and deposits the gradient into `x.grad`. When it calculates the "z-path", it must **add** to the existing `x.grad` to satisfy the summation in the formula above.

### 2.5.2 Why overwriting would be wrong
If PyTorch overwrote gradients instead of adding, you would lose the contribution of the earlier paths. The gradient would represent only the *last* branch computed, which is mathematically incorrect.

### 2.5.3 The role of `zero_grad()` in training loops
In a training loop, we reuse the same weight tensors for every batch.
1.  **Batch 1:** Gradients are calculated and **added** to `.grad` (which starts at 0).
2.  **Optimizer Step:** Weights are updated.
3.  **Batch 2:** If we don't zero out `.grad`, the new gradients are **added to the old gradients** from Batch 1.
    * *Result:* Your updates get larger and larger, mixing history inappropriately.
    * *Fix:* `optimizer.zero_grad()` clears the buffers so Batch 2 starts fresh.

---

## 2.6 Tiny autograd examples (intuition builders)

The best way to trust Autograd is to verify it against high-school calculus. Below are four specific scenarios that cover 90% of the mechanics you will encounter.

### 2.6.1 Mathematical Reasoning

**Case 1: The Scalar Chain Rule**
* **Function:** $f(x) = (3x + 4)^2$
* **Logic:** We use the standard chain rule. Let $u = 3x+4$, so $f = u^2$.
* **Derivative:** $f'(x) = \frac{df}{du} \cdot \frac{du}{dx} = 2u \cdot 3 = 6(3x+4)$.
* **Test Value:** If $x=1$, then $f'(1) = 6(3(1)+4) = 42$.

**Case 2: The Vector Input (Summation)**
* **Function:** $y = x^2$ where $x$ is a vector $[1, 2]$. We then calculate scalar loss $z = \sum y$.
* **Logic:** Autograd calculates the gradient of the scalar $z$ with respect to each element of the vector $x$.
* **Derivative:** $\frac{\partial z}{\partial x_i} = \frac{\partial z}{\partial y}*\frac{\partial y}{\partial x_i} = 1 * 2x_i$.
* **Test Value:**
    * For $x_1=1$, grad is $2(1) = 2$.
    * For $x_2=2$, grad is $2(2) = 4$.
    * Expected gradient vector: $[2, 4]$.

**Case 3: Branching (Accumulation)**
* **Function:** $y = x^2 + 3x$. The variable $x$ splits into two paths that merge back at addition.
* **Logic:** The total derivative is the sum of derivatives from both paths.
* **Derivative:** $\frac{dy}{dx} = 2x + 3$.
* **Test Value:** If $x=2$, then $y' = 2(2) + 3 = 7$.

**Case 4: ReLU (Gradient Gating)**
* **Function:** $y = \text{ReLU}(x)$ summed to a scalar.
* **Logic:** ReLU acts as a gate. If input $> 0$, gradient is 1 (pass). If input $< 0$, gradient is 0 (block).
* **Test Value:**
    * Input $x = [-1, 2]$.
    * Element 1 is negative $\rightarrow$ Grad is $0$.
    * Element 2 is positive $\rightarrow$ Grad is $1$.
    * Expected gradient vector: $[0, 1]$.

### 2.6.2 PyTorch Verification

Run this block to confirm the math above matches the Autograd engine.

```python
import torch

def verify_autograd():
    print("--- 1. Scalar Chain ---")
    x = torch.tensor(1.0, requires_grad=True)
    y = (3 * x + 4) ** 2
    y.backward()
    print(f"Expected: 42.0 | Actual: {x.grad.item()}")

    print("\n--- 2. Vector Input ---")
    x = torch.tensor([1., 2.], requires_grad=True)
    y = x ** 2
    z = y.sum()
    z.backward()
    print(f"Expected: [2., 4.] | Actual: {x.grad.tolist()}")

    print("\n--- 3. Branching ---")
    x = torch.tensor(2.0, requires_grad=True)
    y = x**2 + 3*x
    y.backward()
    print(f"Expected: 7.0 | Actual: {x.grad.item()}")

    print("\n--- 4. ReLU Gating ---")
    x = torch.tensor([-1.0, 2.0], requires_grad=True)
    y = torch.relu(x)
    y.sum().backward()
    print(f"Expected: [0., 1.] | Actual: {x.grad.tolist()}")

if __name__ == "__main__":
    verify_autograd()
```

---

# 3. Manual Backprop: The "Local Gradient" Mental Model

Most tutorials treat Backpropagation as a giant mess of chain rules. This is overwhelming.
As an engineer, it is better to view Backprop as a modular system where every layer implements a standard interface.

## 3.1 The Universal Algorithm: "Hot Potato"

Backprop is just a game of passing a "Hot Potato" (the gradient) backwards.
Every layer in a network, regardless of its type (Linear, Conv, ReLU), has two responsibilities during the backward pass:

1.  **Calculate Gradients for Weights:** If the layer has learnable parameters ($W, b$), it uses the incoming gradient to calculate how to update them.
2.  **Pass Gradient to Input:** It must calculate the gradient with respect to its own input ($x$) to pass it "down" to the previous layer.

**The Golden Formula:**
$$\text{Grad}_{\text{input}} = \text{Grad}_{\text{incoming}} \times \text{Local Derivative}$$

---

## 3.2 Component A: The Linear Layer

Let's dissect the workhorse of DL: $y = xW^T + b$.
* **Forward:** We project data from $D_{in}$ to $D_{out}$.
* **Backward:** We receive a gradient tensor $\delta$ (delta) from the next layer.

### 3.2.1 Setup and Notation
* **Input ($x$):** Shape $(N, D_{in})$. Batch of $N$ vectors.
* **Weights ($W$):** Shape $(D_{out}, D_{in})$. *Note: PyTorch stores weights as (out, in).*
* **Bias ($b$):** Shape $(D_{out})$.
* **Incoming Gradient ($\delta$):** Shape $(N, D_{out})$. This represents $\frac{\partial L}{\partial y}$.

### 3.2.2 Derivation 1: Bias Gradient ($\partial L/\partial b$)
**Intuition:** The bias $b$ was added to *every* sample in the batch during the forward pass (broadcasting).
**The Rule:** Because the forward pass *cloned* $b$ across $N$ samples, the backward pass must *sum* the gradients across those $N$ samples.

$$\frac{\partial L}{\partial b} = \sum_{i=1}^{N} \delta_i$$

**Shape Check:**
* $\delta$ is $(N, D_{out})$.
* Summing over dim 0 yields $(D_{out})$.
* Matches shape of $b$ $(D_{out})$. ✅

### 3.2.3 Derivation 2: Weight Gradient ($\partial L/\partial W$)
**Intuition:** Weights connect input features to output features. The strength of the connection depends on the "strength" of the input ($x$) and the "strength" of the required change ($\delta$).
**The Math:** This is the **Outer Product**.

$$\frac{\partial L}{\partial W} = \delta^T \cdot x$$

**Shape Check:**
* $\delta^T$ is $(D_{out}, N)$.
* $x$ is $(N, D_{in})$.
* Result is $(D_{out}, D_{in})$.
* Matches shape of $W$. ✅

### 3.2.4 Derivation 3: Input Gradient ($\partial L/\partial x$)
**Intuition:** We need to tell the previous layer how much its output contributed to the error. We project the error "backwards" through the weights.
**The Math:** Multiply by the Transpose of the weights.

$$\frac{\partial L}{\partial x} = \delta \cdot W$$

**Shape Check:**
* $\delta$ is $(N, D_{out})$.
* $W$ is $(D_{out}, D_{in})$.
* We need output $(N, D_{in})$.
* Operation: $(N, D_{out}) \times (D_{out}, D_{in}) \rightarrow (N, D_{in})$. ✅

---

## 3.3 Component B: The Activation (ReLU)



The ReLU layer has no weights, so it has no parameters to update. Its only job is to modify the gradient before passing it down.

### 3.3.1 Forward
$$y = \text{max}(0, x)$$

### 3.3.2 Backward (The "Mask")
The derivative of ReLU is a simple switch:
* If $x > 0$, derivative is 1.
* If $x \leq 0$, derivative is 0.

The backward pass applies a **Binary Mask** to the incoming gradient $\delta$.
$$\frac{\partial L}{\partial x} = \delta \odot \mathbb{I}(x > 0)$$
*(Where $\odot$ is element-wise multiplication and $\mathbb{I}$ is the indicator function)*

**Intuition:** If a neuron didn't fire (output 0) during the forward pass, it doesn't get blamed for the error during the backward pass. The gradient is "killed."

---

## 3.4 Putting it Together: A 2-Layer Network Walkthrough

Let's trace a full backprop for: **Input $\rightarrow$ Linear1 $\rightarrow$ ReLU $\rightarrow$ Linear2 $\rightarrow$ Loss**

### Phase 1: Forward Pass (Cache Storage)
1.  **L1:** $z_1 = xW_1^T + b_1$ (Store $x$ for backward).
2.  **Act:** $a_1 = \text{ReLU}(z_1)$ (Store mask of $z_1 > 0$).
3.  **L2:** $z_2 = a_1W_2^T + b_2$ (Store $a_1$).
4.  **Loss:** Calculate $L$.

### Phase 2: Backward Pass (The Chain)

**Step 1: Start at the end**
* We have $\frac{\partial L}{\partial z_2}$ (let's call it $\delta_2$).

**Step 2: Backprop Linear 2**
* **Update Params:**
    * $\text{grad}_{W2} = \delta_2^T \cdot a_1$
    * $\text{grad}_{b2} = \text{sum}(\delta_2)$
* **Pass Down:**
    * $\delta_{a1} = \delta_2 \cdot W_2$

**Step 3: Backprop ReLU**
* **Update Params:** None.
* **Pass Down:**
    * $\delta_{z1} = \delta_{a1} \odot \text{Mask}(z_1 > 0)$
    * *(Effectively zeros out gradients for neurons that were dead).*

**Step 4: Backprop Linear 1**
* **Update Params:**
    * $\text{grad}_{W1} = \delta_{z1}^T \cdot x$
    * $\text{grad}_{b1} = \text{sum}(\delta_{z1})$
* **Pass Down:**
    * $\delta_{x} = \delta_{z1} \cdot W_1$ (If we had another layer before this).

---

## 3.5 Consolidated Mental Models & Debug Checklist

### 3.5.1 The "Transpose" Rule
If the forward pass involves multiplication by $W$, the backward pass almost always involves multiplication by $W^T$. You are literally projecting the error back into the input space.

### 3.5.2 The "Summation" Rule
If a variable is reused or broadcasted (like a bias vector used on every row, or a residual connection splitting to two paths), its gradient is the **sum** of all incoming gradients from those usages.

### 3.5.3 Debug Checklist
If your training crashes or loss doesn't move, check these:
1.  **Shape Matching:** Does `weight.grad.shape == weight.shape`?
2.  **Broadcasting Bugs:** Did you accidentally broadcast a vector $(N)$ against a matrix $(N, N)$ in your loss function?
3.  **Dead ReLU:** Did you initialize weights such that all your neurons are negative? If so, gradients are all 0 (The "Dying ReLU" problem).
4.  **Zero Grad:** Did you forget `optimizer.zero_grad()`? Your gradients are accumulating to infinity.

---

## 4. Initialization & Normalization: Keeping the Signal Alive
### 4.1 The Goal: Mean=0, Variance=1 (Signal Preservation)
### 4.2 Failure Modes: What happens when Init goes wrong?
#### 4.2.1 The "Symmetry Trap" (Why Zero Init = Single Neuron Network)
#### 4.2.2 The "Vanishing Signal" (Small weights $\rightarrow$ Activations decay to 0)
#### 4.2.3 The "Exploding Signal" (Large weights $\rightarrow$ NaNs and Inf)
### 4.3 The Modern Standards: Xavier (Glorot) vs Kaiming (He)
### 4.4 Normalization Layers: Forcing statistics during training
#### 4.4.1 BatchNorm (Vertical/Batch-wise) vs LayerNorm (Horizontal/Sample-wise)
#### 4.4.2 The "Running Stats" gotcha in BatchNorm (Train vs Eval mode)

## 5. Activation Functions: The Non-Linearity
### 5.1 Why we need them (The "Collapsing Linear" Proof)
### 5.2 The "S-Curves": Sigmoid & Tanh (and why they vanish)
### 5.3 The Modern Standards: ReLU & GELU
### 5.4 The "Dead Neuron" problem (and how LeakyReLU fixes it)

## 6. Loss Functions & Numerical Stability
### 6.1 Softmax + Cross-Entropy: The "Logits" Pipeline
### 6.2 The "Log-Sum-Exp" Trick (preventing `NaN` in production)
### 6.3 The "First Iteration" Math Proof ($\ln(C)$ sanity check)
### 6.4 Metrics vs Loss (Why we can't optimize Accuracy directly)

## 7. Regularization: Fighting Overfitting
### 7.1 The Overfitting intuition (Memorization vs Generalization)
### 7.2 Dropout: The "Ensemble" interpretation (Training sub-networks)
### 7.3 Weight Decay vs L2 Regularization (The "Shrinking" intuition)

## 8. Optimizers: The Update Step
### 8.1 SGD: The basic "Step Downhill" ($\theta = \theta - \eta \cdot \nabla$)
### 8.2 Momentum: The "Heavy Ball" intuition (dampening oscillation)
### 8.3 Adam: The "Adaptive" intuition (why it's the default choice)

## 9. The Training Loop Anatomy (Engineering View)
### 9.1 The Standard Boilerplate (Forward $\rightarrow$ Loss $\rightarrow$ Backward $\rightarrow$ Step $\rightarrow$ Zero)
### 9.2 `model.train()` vs `model.eval()` (The global switch)
### 9.3 Gradient Accumulation (Simulating large batches on small GPUs)
### 9.4 Device Management: CPU vs GPU
#### 9.4.1 The `.to(device)` pattern (Device Agnostic code)
#### 9.4.2 The Bottleneck: CPU-to-GPU Data Transfer (Pinned Memory)

## 10. Final Boss: Minimal NN in Pure NumPy
### 10.1 Setting up parameters (Xavier Init from scratch)
### 10.2 Forward pass (Linear $\rightarrow$ ReLU $\rightarrow$ Linear)
### 10.3 Computing Loss and Gradients manually
### 10.4 The Update loop
