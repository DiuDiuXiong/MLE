# CNN

## 1. The Intuition & Principles

### Why CNNs over MLPs?
* **Parameter Efficiency:** MLPs are fully connected. A 224x224 RGB image with 1,000 hidden units = 150M+ params. CNNs use significantly fewer.
* **Spatial Topology:** MLPs flatten images, losing neighbor relationships. CNNs preserve 2D/3D structure.
* **Translation Invariance:** An object should be recognized regardless of its position. CNNs achieve this through weight sharing.

### Key Concepts
* **Local Receptive Fields:** Neurons only look at a small local patch (e.g., 3x3) rather than the whole image.
* **Weight Sharing:** The same filter (kernel) slides across the entire image.
    * **Benefit:** Acts as a regularizer, reducing **Variance** (overfitting).
    * **Trade-off:** Increases **Bias** by assuming spatial stationarity. (a combo that is an edge in top is also an edge in bottom)

### Parameter Calculation Formula
For any Conv layer:
`Params = ((filter_w * filter_h * in_channels) + 1(bias)) * num_filters`

---

## 2. Convolution Mechanics

### 2.1 The Dot Product Operation
The "Convolution" is actually a sliding window of dot products.
1. **Align:** Place the Filter (Kernel) over a patch of the image.
2. **Multiply:** Multiply each weight in the filter by the corresponding pixel value.
3. **Sum:** Add all products together + a bias term to get **one single value**.
4. **Slide:** Move the filter by the **Stride** and repeat.

**Example Calculation (3x3 Filter on Grayscale):**
Input Patch:          Filter Weights:
[1, 2, 0]             [1, 0, -1]
[0, 1, 1]      dot    [1, 0, -1]
[5, 0, 1]             [1, 0, -1]

Result = (1*1 + 2*0 + 0*-1) + (0*1 + 1*0 + 1*-1) + (5*1 + 0*0 + 1*-1)
       = (1) + (-1) + (4) = **4** (This becomes one pixel in the output map).

### 2.2 The Dimensionality Formula
To determine the output size ($O$) of a layer based on Input ($I$), Filter ($K$), Padding ($P$), and Stride ($S$):

$$O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$$

* **Design Tip:** It is best practice to choose $P$ and $S$ such that the result is an **integer**. 
* **The "Floor" Problem:** If the result is not an integer, the framework will "floor" the value, meaning the filter cannot complete its last stride and the right/bottom edge of your data is **dropped (ignored)**.

### 2.3 Padding ($P$)
* **Valid Padding:** $P=0$. The image shrinks.
* **Same Padding:** We add zero-padding so the output size equals the input size (when $S=1$).
* **Formula for "Same":** To keep dimensions identical, set $P = (K - 1) / 2$.

### 2.4 Stride ($S$)
* The number of pixels the filter skips after each operation.
* **Higher Stride = Smaller Output.** This is a learned way to downsample an image instead of using a fixed method like Pooling.

### 2.5 The 3D Volume Rule
* Filters are always 3D. If an image is $H \times W \times C$ (Channels), a $3 \times 3$ filter is actually $3 \times 3 \times C$.
* **1 Filter always produces 1 2D Feature Map**, regardless of input depth.
* To get an output with 64 channels, you must apply 64 separate filters.
* But the number of filter will become the number of channel of next layer.

---

## 3. Pooling & Receptive Field Dynamics

### 3.1 Spatial Downsampling via Pooling
Pooling layers are non-parametric operations designed to reduce the spatial resolution of feature maps, providing a form of **local translation invariance**.

* **Max Pooling:** Selects the maximum activation within a kernel window.
    * **Mechanism:** Acts as a "strongest signal" filter.
    * **Invariance:** By discarding the exact location of a feature within the window, it makes the representation robust to small spatial perturbations (shifts).
* **Average Pooling:** Computes the arithmetic mean of activations.
    * **Mechanism:** Acts as a low-pass filter, smoothing the feature map and capturing the global context of a region.
* **Global Average Pooling (GAP):** A specialized case where the kernel size equals the input dimensions ($H \times W$), reducing each feature map to a single scalar. 
    * **MLE Insight:** GAP is often used to replace Fully Connected layers at the network head, drastically reducing parameter count and mitigating overfitting. (head here means last few layers)

### 3.2 Effective Receptive Field (ERF)
The ERF is the spatial extent of the input image that influences a specific neuron's activation in a deeper layer.

* **The Power of Stacking:** Stacking multiple small kernels (e.g., $3 \times 3$) allows for a high ERF while maintaining computational efficiency.
* **The Parameter Advantage:** * One $7 \times 7$ layer has $C^2 \times 49$ parameters.
    * Three $3 \times 3$ layers have $3 \times (C^2 \times 9) = C^2 \times 27$ parameters.
    * **Result:** Stacking provides a **45% reduction in parameters** for the same receptive field, with the added benefit of three non-linearities (e.g., ReLU) instead of one.



### 3.3 Advanced Architecture: Parallel Concatenated Pooling
Modern architectures sometimes leverage both Max and Average signals simultaneously to preserve both high-frequency (peak) and low-frequency (background) features.

#### Implementation: `ConcatPool2d` (PyTorch)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatPool2d(nn.Module):
    """
    Concatenates Max and Average Pooling along the channel dimension.
    Commonly used in architectures like FastAI's Head or custom attention blocks.
    """
    def __init__(self, kernel_size=2, stride=2):
        super(ConcatPool2d, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride)
        self.avg_pool = nn.AvgPool2d(kernel_size, stride)

    def forward(self, x):
        # Resulting tensor will have 2 * input_channels
        return torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)

# Usage Example:
# input_tensor shape: (Batch, 64, 28, 28)
# output_tensor shape: (Batch, 128, 14, 14)
# dim means stack in that dim, torch.stack create a new dim
```

| Operation | Dimension Change | Common Use Case |
| :--- | :--- | :--- |
| **`torch.cat`** | Joins along **existing** dimension. | Concatenating feature maps/channels (e.g., DenseNet, Inception, or ConcatPool). |
| **`torch.stack`** | Joins along a **new** dimension. | Organizing a sequence of separate images or tensors into a single batch/volume. |

---

## 4. Non-Linearity (Activation Functions)

### 4.1 The Necessity of Non-Linearity
Without non-linear activation functions, a multi-layer CNN remains a **linear transformation**. Mathematically, the composition of multiple linear layers $W_2(W_1x)$ collapses into a single linear layer $W_{total}x$. Non-linearity allows the network to approximate complex, high-dimensional functions (the Universal Approximation Theorem).

### 4.2 The Evolution of Activations: Why ReLU?
In modern CNN architectures, **Sigmoid** and **Tanh** have been deprecated for hidden layers due to the **Vanishing Gradient Problem**.

* **The Sigmoid Constraint:** Sigmoid saturates at 0 and 1. Its derivative peaks at 0.25. In deep networks, multiplying these small gradients during backpropagation causes the gradient to decay exponentially, preventing early layers from updating.
* **The ReLU (Rectified Linear Unit) Advantage:** $f(x) = \max(0, x)$
    * **Non-Saturating Gradient:** For all $x > 0$, the gradient is a constant **1.0**. This eliminates vanishing gradients in the positive domain.
    * **Computational Efficiency:** Involves only a simple thresholding operation (no exponentials).
    * **Sparsity:** Mimics biological neurons by "turning off" neurons with negative inputs, leading to a more efficient representation.

### 4.3 The "Dying ReLU" Phenomenon
An MLE interview favorite. If a large gradient update pushes a neuron's weights into a state where it outputs negative values for all inputs in the training set, the gradient becomes permanently **0**.
* **Result:** The neuron is "dead" and cannot be revived via standard backpropagation because no gradient flows back through it.
* **Symptom:** You may observe high percentages of zero-valued activations in your feature maps.

### 4.4 Advanced Variants (The "Leaky" Family)
To mitigate the Dying ReLU problem, researchers introduced slopes into the negative domain:

1. **Leaky ReLU:** $f(x) = \max(\alpha x, x)$ where $\alpha$ is a small constant (e.g., 0.01). It ensures a small gradient always exists.
2. **ELU (Exponential Linear Unit):** Uses an exponential curve for negative values, making the mean activation closer to zero, which speeds up convergence.
3. **GELU (Gaussian Error Linear Unit):** Used in state-of-the-art models (Transformers, Vision Transformers). It scales the input by the cumulative distribution function of the Gaussian distribution, providing a smoother, probabilistic non-linearity.

### 4.5 Implementation (PyTorch)
```python
import torch.nn as nn

# Standard ReLU
self.relu = nn.ReLU()

# Leaky ReLU (to prevent dying neurons)
self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

# GELU (Modern State-of-the-art)
self.gelu = nn.GELU()
```

---

## 5. The Fully Connected (FC) Head & Classification

### 5.1 The Architecture "Handshake"
By the end of the Convolutional base, the data is a 3D volume (e.g., $7 \times 7 \times 512$). The "Head" is the final part of the network that translates these abstract spatial features into a categorical decision (e.g., "Dog" vs. "Cat").

### 5.2 Transitioning from 3D to 1D
There are two primary ways to "bridge" the spatial feature maps to the classifier:

#### A. Flattening (The Traditional Way)
- **Mechanism:** Collapses the entire 3D volume into a single long 1D vector. 
- **The "Fixed Size" Trap:** Because the resulting vector must match the input size of the next Dense layer, **Flattening forces the entire CNN to have a fixed input image size** (e.g., exactly 224x224).
- **The Parameter Explosion:** This is where most parameters live. 
    - *Example:* A $7 \times 7 \times 512$ map flattens to **25,088** units. A single Dense layer with 4,096 neurons following this would require $(25,088 \times 4,096) \approx \mathbf{102 \text{ million parameters}}$.

#### B. Global Average Pooling / GAP (The Modern Way)
- **Mechanism:** Takes the average of each channel's $H \times W$ grid. A $7 \times 7 \times 512$ volume becomes a $1 \times 1 \times 512$ vector.
- **Benefits:** 1. **Parameter Efficiency:** Reduces the input to the classifier from 25,088 down to 512.
    2. **Size Agnostic:** Since it averages the whole grid regardless of size, a model with GAP can theoretically accept **any** image resolution during inference.

### 5.3 The Feature Hierarchy (How the "Brain" Builds)
The network doesn't see a "dog" immediately. It builds the concept layer by layer:
1. **Low-Level (Layers 1-2):** Detects "Gabor Filters"—edges, lines, and color blobs.
2. **Mid-Level (Layers 3-5):** Combines lines into textures (circles, stripes) and simple shapes (honeycombs, squares).
3. **High-Level (Late Layers):** Combines shapes into semantic parts (eyes, wheels, bird beaks).
4. **The Head (FC Layers):** Performs high-level "reasoning." It asks: *"If I see a beak and feathers, what is the probability this is a Pelican?"*

### 5.4 Peering Inside: Visualization & Verification
How do we know if a layer has "understood" a concept?
- **Activation Maximization:** Using gradient descent to find an image that makes a specific neuron fire the most. 
- **Linear Probing:** Freezing the model at an intermediate layer and training a simple linear classifier on those features. If it performs well, that layer has already "captured" the object's identity.
- **Saliency / Grad-CAM:** Heatmaps that show which pixels in the original image the model is "looking at" to make its final decision.



### 5.5 Final Activation: Softmax
To output a probability distribution across $K$ classes:
$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$
- **Logits:** The raw output numbers from the last Dense layer.
- **Softmax:** Turns logits into values between 0 and 1 that sum to 1.0. This is the standard output for multi-class classification.