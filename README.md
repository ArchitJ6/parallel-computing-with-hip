# 🚀 GPU Parallel Computing: HIP & CUDA

> A structured, hands-on exploration of GPU programming using **HIP (portable)** and **CUDA (NVIDIA)** — built from fundamentals to real-world parallel workloads.

---

<p align="center">
  <img src="https://img.shields.io/badge/GPU-CUDA%20%7C%20HIP-blue" />
  <img src="https://img.shields.io/badge/Language-C%2B%2B-orange" />
  <img src="https://img.shields.io/badge/Parallel-Computing-green" />
  <img src="https://img.shields.io/badge/Docker-Enabled-blue" />
</p>

---

# 📌 Overview

This repository demonstrates:

* ⚡ GPU parallel programming fundamentals
* 🧠 Thread hierarchy & execution model
* 📊 Data-parallel algorithms
* 🔁 Reduction & synchronization
* 🧮 Matrix computations
* 🧠 Memory access optimization
* 🖼️ Image processing on GPU
* 🔄 Portability from HIP → CUDA

---

# 📁 Project Structure

```bash
.
├── 01_basics/
├── 02_vector_ops/
├── 03_reduction/
├── 04_matrix/
├── 05_memory_patterns/
├── 06_image_processing/
│
├── cuda_versions/
│   ├── (same structure as above)
│
├── Dockerfile
├── docker-compose.yml
```

---

# 🧠 GPU Execution Model

All programs follow this pipeline:

```text
CPU (Host)
   ↓
Memory Transfer (Host → Device)
   ↓
GPU Kernel Execution (Parallel Threads)
   ↓
Memory Transfer (Device → Host)
```

---

# 📂 Module Breakdown (with File Explanation)

## 🔹 01_basics/

### ✔ `hello_threads`

* Prints thread IDs from GPU
* Demonstrates **parallel execution**

### ✔ `thread_indexing`

* Displays block and thread indices
* Helps understand **thread hierarchy**

---

## 🔹 02_vector_ops/

### ✔ `vector_add`

* Each thread adds one element
  👉 Demonstrates **data parallelism**

### ✔ `vector_square`

* Each thread squares one value
  👉 Fully independent computation

---

## 🔹 03_reduction/

### ✔ `dot_product_atomic`

* Uses `atomicAdd` to accumulate results

👉 Teaches:

* Race conditions
* Synchronization
* Reduction patterns

---

## 🔹 04_matrix/

### ✔ `matrix_add`

* Uses 2D thread mapping
  👉 Introduces **grid-based parallelism**

### ✔ `matrix_mul`

* Performs matrix multiplication

👉 Core concept used in:

* Machine Learning
* Deep Learning (GEMM)

---

## 🔹 05_memory_patterns/

### ✔ `reverse_array`

* Demonstrates reverse indexing
  👉 Memory remapping logic

### ✔ `strided_access`

* Accesses memory with stride

👉 Shows:

* Non-coalesced memory access
* Performance implications

---

## 🔹 06_image_processing/

### ✔ `grayscale`

* Converts RGB image data → grayscale

👉 First **real-world GPU workload**

---

# 🔄 HIP vs CUDA

Both implementations follow the same logic:

| Concept           | HIP                  | CUDA                  |
| ----------------- | -------------------- | --------------------- |
| Memory Allocation | hipMalloc            | cudaMalloc            |
| Memory Copy       | hipMemcpy            | cudaMemcpy            |
| Synchronization   | hipDeviceSynchronize | cudaDeviceSynchronize |
| Compiler          | hipcc                | nvcc                  |

👉 Only API names differ — execution model is identical.

---

# ⚙️ How to Run

## 🟣 HIP (AMD / Portable)

```bash
hipcc file.cpp -o output
./output
```

---

## 🟢 CUDA (NVIDIA)

```bash
nvcc file.cu -o output
./output
```

---

# 🐳 Docker Setup (CUDA Only)

This project includes Docker support for **CUDA execution using NVIDIA GPU**

## ⚠️ Important

* Works only with **NVIDIA GPUs**
* Uses CUDA base image
* HIP (ROCm) is **not supported in this container**

---

## ▶️ Start Container

```bash
docker compose up -d
```

---

## ▶️ Enter Container

```bash
docker compose exec cuda-dev bash
```

---

## ▶️ Run Example

```bash
cd cuda_versions/01_basics
nvcc hello_threads.cu -o hello
./hello
```

---

## 🧪 Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
```

---

# 📊 When GPU Helps

| Workload         | CPU        | GPU       |
| ---------------- | ---------- | --------- |
| Vector Ops       | Moderate   | Fast      |
| Matrix Ops       | Slow       | Very Fast |
| Image Processing | Sequential | Parallel  |

👉 GPU excels when:

* Work is independent
* Data size is large

---

# 🧠 Key Learnings

* GPU requires **parallel thinking**, not sequential logic
* Memory transfer can be a bottleneck
* Thread mapping impacts performance
* Memory access patterns matter

---

# 🚀 Future Improvements

* Shared memory optimization
* CPU vs GPU benchmarking
* Real image input/output
* Multi-GPU scaling

---

# 🏆 Why This Repo Stands Out

* Structured learning progression
* Covers both HIP & CUDA
* Includes Docker-based execution
* Demonstrates real-world GPU usage

---

# ⭐ Final Thought

> “Parallel computing isn’t just faster computing —
> it’s smarter computation at scale.”
