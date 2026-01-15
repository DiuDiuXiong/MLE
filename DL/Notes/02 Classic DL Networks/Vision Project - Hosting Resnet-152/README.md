# VisionNode-152: Large-Scale Image Classification Pipeline

**VisionNode-152** is an end-to-end Machine Learning Engineering project focused on the training, optimization, and production-grade deployment of a deep residual network (**ResNet-152**) using the **ImageNet-1K** dataset. 

The project follows a "Local Validation, Remote Scale" strategy: validating model convergence and data pipelines on high-performance local hardware (RTX 4090) before streaming workloads to remote, resource-constrained compute nodes for final training and high-throughput inference.

---

## 📖 Project Overview

This project implements a full-cycle vision pipeline designed to categorize images into 1,000 distinct classes. By utilizing the 152-layer ResNet architecture, the system is capable of learning complex feature hierarchies while maintaining stability during deep training through residual learning.

[Image of ResNet-152 architecture diagram and bottleneck blocks]

The project is structured to handle the **150GB ImageNet-1K** dataset efficiently, moving from a local "Offline Validation" phase to a "Remote Production" phase.

---

## 🛠 Technical Challenges

* **Local-First Validation:** Ensuring the model converges and gradients flow correctly across 152 layers before committing to a long-running remote training job.
* **Distributed Data Streaming:** Saturating international network links to move optimized data shards without starving the remote training process.
* **Production Serving & QPS:** Optimizing a heavyweight model to run on standard cloud CPUs while meeting aggressive throughput requirements.


---

## 🎯 Final Project Goals

### 1. Offline Validation (Local 4090 Phase)
* **Smoke Testing:** Verify the training loop with a subset of ImageNet to ensure the loss decreases.
* **Architecture Integrity:** Confirm the ResNet-152 implementation successfully handles 1,000-class output.
* **Pipeline Profiling:** Identify local I/O bottlenecks during data loading before going remote.

### 2. Accuracy & Classification (The Science)
* **Performance:** Target competitive Top-1 and Top-5 accuracy benchmarks on the full 1.2M image set.

### 3. Production Serving (The SRE-MLE Hybrid)
* **High QPS Target:** Achieve **100+ Queries Per Second (QPS)** on the remote VPS.
* **Low Latency:** Maintain **P99 Latency < 150ms** for inference requests.
* **Efficiency:** Use **INT8 Quantization** to fit the 60M+ parameter model within an 8GB RAM footprint.

---

## 🏗 Project Roadmap

1.  **Phase 1: Local Offline Training & Validation**
    Running a "mini-training" session on the 4090 using the existing 150GB local dataset to validate model architecture and convergence.
2.  **Phase 2: Data Sharding & Transfer**
    Packaging local data into optimized binary shards and establishing a multi-threaded TCP transfer link to the Singapore VPS.
3.  **Phase 3: Remote Model Training**
    Scaling the training loop to the remote node, utilizing checkpointing and mixed-precision logic.
4.  **Phase 4: Serving & Benchmarking**
    Quantizing the model and performing stress