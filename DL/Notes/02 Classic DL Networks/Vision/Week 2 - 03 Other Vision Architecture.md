# Computer Vision: From Fundamentals to Architectural Mastery

This section of the repository documents the theoretical deep-dive into "Modern Traditional" Computer Vision. The focus is on the architectural evolution of Convolutional Neural Networks (CNNs), emphasizing mathematical proofs for gradient flow, parameter efficiency, and hardware-specific optimizations.

## Section 0: The Roadmap to Interview Immunity

We focus on the "Leaps" rather than the "Steps." Each chapter covers a model that introduced a fundamental shift in how we handle deep neural networks.

### 🎯 Objective
To understand the *why* behind each architecture, reaching a level of depth where I can mathematically derive gradient flows and explain hardware trade-offs during MLE system design interviews.

### 🗺️ The Curriculum Plan

| Chapter | Model | Core Learning Pillar | Key Interview Concept |
| :--- | :--- | :--- | :--- |
| **01** | **ResNet & Highway Nets** | Solving the Degradation Problem | Identity Mappings & Gradient Highways |
| **02** | **Inception (GoogLeNet)** | Multi-scale Feature Extraction | $1 \times 1$ Convolutions & Factorization |
| **03** | **MobileNets (v1-v3)** | Hardware-Efficient Inference | Depthwise Separable Convolutions |
| **04** | **DenseNet** | Feature Reuse & Concatenation | Growth Rate & Parameter Efficiency |
| **05** | **U-Net & FCNs** | Pixel-level Reasoning | Skip Connections for Spatial Recovery |

---

### 🛠️ Hardware & Data Considerations (SRE Notes)
* **Profiling:** Every model is analyzed for its memory footprint and GPU utilization (Compute-bound vs. Memory-bound).
* **Data I/O:** Optimization strategies for large datasets, transitioning from standard Image Loaders to high-throughput formats like **Parquet**, **WebDataset**, or **NVIDIA DALI**.
* **Numerical Stability:** Analysis of vanishing/exploding gradients and the impact of weight initialization/normalization techniques.