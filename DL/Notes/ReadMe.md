# 🛠️ Inference Engineer Technical Roadmap (Theory + Implementation)
**Owner:** DiuDiuXiong
**Strategy:** "Learn the Physics (Theory) -> Build the Engine (Code)."
**Hardware:** Local CPU (Theory/Debugging) + Cloud GPU (Training/Serving).

---

## 🏗️ Epic 1: The Low-Level Mechanics (Debug Capability)
**Goal:** Master the raw materials. If you understand shapes and gradients, you can debug anything.

### 📘 Phase 1.0: Theory & Intuition (The Physics)
* **1.0.1 The Tensor Mental Model**
    * **Math:** A tensor is just a data container. A 0D tensor is a scalar ($x$), 1D is a vector ($\vec{v}$), 2D is a matrix ($A$).
    * **Intuition:** "Shape = Semantic Meaning."
        * `[32, 768]` means "Batch of 32 items, each described by 768 numbers."
        * If the shapes don't align, the math is physically impossible.
* **1.0.2 Matrix Multiplication (The Engine)**
    * **Formula:** $C_{ik} = \sum_j A_{ij} B_{jk}$.
    * **The "K-Dimension" Rule:** To multiply matrices, the inner dimensions must match (Column of A = Row of B).
    * **Visual:** `[Batch, Input_Dim] @ [Input_Dim, Hidden_Dim] = [Batch, Hidden_Dim]`. This "transforms" the data from one representation to another.
* **1.0.3 Calculus (The Learning Signal)**
    * **The Chain Rule:** "If I change input $x$, how does it ripple through function $f$, then function $g$, to change loss $L$?"
    * **Math:** $\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}$.
    * **Intuition:** Backpropagation is just assigning blame. If the error is high, which weight contributed most to it?

### 🛠️ Phase 1.1: Implementation Spec (The Code)
* **1.1.1 Tensor Internals Spec**
    * **Concept:** Strides vs. Contiguous memory. Why `.view()` raises errors but `.reshape()` copies data.
    * **Task:** Create a non-contiguous tensor (transpose a matrix) and try to `.view()` it. Watch it fail. Fix it with `.contiguous()`.
* **1.1.2 The "Micro-Autograd"**
    * **Task:** Implement a class `Value` in Python.
    * **Requirement:** Support `__add__`, `__mul__`, `__pow__`, and `backward()`.
    * **Test:** Implement $f(x) = x^2 + 3x$ and verify your manual derivative against PyTorch's.

---

## 🏗️ Epic 2: The Classics (Performance Profiling)
**Goal:** Understand how different architectures stress hardware differently (Compute-Bound vs. Memory-Bound).

### 📘 Phase 2.0: Theory & Intuition
* **2.0.1 Convolution (The "Scanner")**
    * **Math:** A dot product that slides across the image.
    * **Intuition:** **Spatial Invariance.** A cat in the top-left corner is the same as a cat in the bottom-right. We share the same weights (filters) across the whole image.
    * **Receptive Field:** How one pixel in the output depends on a 3x3 patch, which depends on a 5x5 patch... eventually seeing the whole image.
* **2.0.2 Recurrence (The "Memory")**
    * **Math:** $h_t = \tanh(W x_t + U h_{t-1})$.
    * **Intuition:** The "Hidden State" $h_t$ is a compressed summary of everything the network has seen so far.
    * **The Problem:** **Vanishing Gradient.** If you multiply a number $< 1$ (like 0.9) by itself 100 times, it becomes zero. This means the network "forgets" what happened 100 steps ago.

### 🛠️ Phase 2.1: Implementation Spec
* **2.1.1 The Bottleneck Hunt (CNNs)**
    * **Task:** Train ResNet-18 on CIFAR-10.
    * **SRE Focus:** Use `torch.profiler`.
    * **Deliverable:** A "Flame Graph" showing the GPU waiting for the CPU (DataLoader bottleneck). Fix it by increasing `num_workers`.
* **2.1.2 Vanishing Gradients (RNNs)**
    * **Task:** Train a vanilla RNN on a long sequence.
    * **Debug:** Print the `.grad` norm of the first layer's weights. Watch it decay to zero.

---

## 🏗️ Epic 3: Transformer Architecture (The Core Spec)
**Goal:** Build the engine that powers ChatGPT.

### 📘 Phase 3.0: Theory & Intuition
* **3.0.1 The "Search Engine" Analogy (Attention)**
    * **Concept:** Attention is a "soft" lookup in a database.
    * **Query ($Q$):** "What am I looking for?" (e.g., current token).
    * **Key ($K$):** "What defines this item?" (e.g., previous tokens).
    * **Value ($V$):** "What is the content?" (e.g., the vector to retrieve).
    * **Math:** $\text{softmax}(\frac{Q \cdot K^T}{\sqrt{d}}) \cdot V$. The dot product measures similarity between Query and Key.
* **3.0.2 Positional Encoding**
    * **Intuition:** Attention is "permutation invariant" (a bag of words). It doesn't know order. We must mathematically "stamp" the order onto the vectors.
* **3.0.3 Layer Normalization**
    * **Math:** $\frac{x - \mu}{\sigma}$.
    * **Intuition:** It stabilizes the learning by ensuring the numbers don't get too big or too small between layers.

### 🛠️ Phase 3.1: Implementation Spec
* **3.1.1 Manual Attention**
    * **Task:** Implement `def self_attention(x): ...` using raw matrix multiplication.
    * **Requirement:** Implement the "Causal Mask" (ensure token 5 cannot see token 6).
    * 
* **3.1.2 The Fine-Tuning Pipeline**
    * **Hardware:** Rent GPU (Vast.ai).
    * **Task:** Load `Llama-3-8B` with `bitsandbytes` (4-bit). Apply `LoRA`.
    * **Success Metric:** The model output changes style (e.g., starts answering like a pirate) after training.

---

## 🏗️ Epic 4: Inference Engineering (Productionization)
**Goal:** This is your interview winner. Latency, Throughput, and Memory.

### 📘 Phase 4.0: Theory & Intuition
* **4.0.1 The Autoregressive Bottleneck**
    * **Concept:** To generate token 100, you need token 99. You cannot parallelize generation (unlike training).
    * **Memory Bound:** The GPU spends more time *loading* weights from VRAM than *calculating* the math.
* **4.0.2 KV Caching Math**
    * **Problem:** In step 10, calculating Attention requires $Q_{10}$ against $K_1...K_9$. In step 11, you need $K_1...K_{10}$.
    * **Solution:** Don't recalculate $K_1...K_9$. Store them in VRAM.
    * **Cost:** Huge VRAM usage.
* **4.0.3 The Roofline Model**
    * **Graph:** Performance (FLOPS) vs Arithmetic Intensity (Math/Byte).
    * **Intuition:** Are we limited by the speed of the math (Compute Bound) or the speed of the memory wires (Memory Bound)?

### 🛠️ Phase 4.1: Implementation Spec
* **4.1.1 vLLM & PagedAttention**
    * **Task:** Deploy your fine-tuned model using `vLLM`.
    * **Experiment:** Set `block_size` to different values. Observe memory fragmentation.
* **4.1.2 The Load Test**
    * **Task:** Hammer your API with `Locust`.
    * **Graph:** Plot **TTFT** (Time To First Token) vs **RPS** (Requests Per Second). Identify the "Knee of the Curve" (where latency explodes).

---

## 🏗️ Epic 5: Specialized Hardware & Hybrid Routing
**Goal:** Optimize for Cost vs Performance.

### 📘 Phase 5.0: Theory & Intuition
* **5.0.1 Chip Architecture**
    * **GPU (NVIDIA):** Uses HBM (High Bandwidth Memory). Great for massive parallel data, but latency is limited by memory transfer.
    * **LPU (Groq):** Uses SRAM (Static RAM). It's physically closer to the logic gates. Insanely fast, but tiny capacity (can't fit big models easily).
* **5.0.2 Quantization Theory**
    * **Math:** Converting `float32` (4 bytes) to `int8` (1 byte).
    * **Trade-off:** You lose precision (accuracy) to gain speed and save memory.

### 🛠️ Phase 5.1: Implementation Spec
* **5.1.1 The "Smart Router"**
    * **Task:** Build a Gateway.
        * **Route A (Hard):** $\to$ Groq (Llama-3-70B).
        * **Route B (Easy):** $\to$ Local vLLM (Llama-3-8B).
* **5.1.2 Multi-Modal Pipelines**
    * **Task:** Text $\to$ Summary (LLM) $\to$ Image (Flux) $\to$ Audio (TTS).
    * **SRE Focus:** Handle async failures (circuit breaker pattern) if one API fails.

---

## ✅ Capstone: "The SRE Copilot"
*Combine Phase 1-5.*
1.  **Ingest:** Webhook logs.
2.  **Analyze:** Local vLLM (Quantized 8B model).
3.  **Escalate:** If low confidence, call Groq (70B model).
4.  **Visualize:** Dashboard of Latency and Cost.