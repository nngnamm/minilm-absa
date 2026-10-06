# MiniLM Knowledge Distillation for Aspect-Based Sentiment Analysis (ABSA)

This repository implements a mathematically rigorous **MiniLM knowledge distillation pipeline** to compress a 12-layer BERT expert into a lightweight 4-layer student model.

The pipeline is explicitly designed for **continuous regression (0.0 to 1.0)** on Aspect-Based Sentiment Analysis (ABSA) tasks using multi-GPU environments.

---

## Architecture & Mathematical Implementation

Unlike standard hidden-state distillation, this implementation replicates the original **Microsoft MiniLM** specification by distilling deep self-attention relationships.

Because the student (384d) and teacher (768d) have different hidden dimensions, the implementation bypasses the dimension mismatch by transferring the **attention distribution matrices directly**.

### Key Components

- **Task:** Aspect-Based Sentiment Analysis (ABSA)
- **Target:** Continuous Regression using Mean Squared Error (MSE)
  - `1.0` = Positive
  - `0.5` = Neutral / Conflict
  - `0.0` = Negative
- **Attention Transfer (A):** KL Divergence between Teacher and Student softmax attention probability distributions.
- **Value-Relation Transfer (VR):** Manual extraction of the Value projection layers (`attention.self.value`) to compute scaled dot-product `VVᵀ` relation matrices, optimized using Temperature-scaled KL Divergence.
- **Hardware Setup:** Distributed Data Parallel (DDP) multi-GPU execution through Hugging Face `accelerate`.

### Eager Attention

Hugging Face Transformers defaults to `sdpa` (Scaled Dot-Product Attention), which uses fused C++ kernels and does not expose attention weights in the form required for relation distillation.

Therefore, this repository explicitly forces:

```python
attn_implementation="eager"
```

This allows the training pipeline to materialize and extract the attention matrices required for MiniLM-style relation distillation.

---

# Two-Stage Training Pipeline

To ensure that the student learns from a competent teacher, training is divided into two stages.

## Stage 1: Expert Teacher Fine-Tuning

The base `bert-base-uncased` model is fine-tuned on the:

`tomaarsen/setfit-absa-semeval-restaurants`

dataset.

Inputs are pair-tokenized in the following format:

```text
[CLS] Sentence [SEP] Aspect [SEP]
```

The teacher is optimized using **Mean Squared Error (MSE)** to predict a continuous sentiment score between `0.0` and `1.0`.

### Training

```bash
accelerate launch --num_processes=2 train_teacher.py
```

---

## Stage 2: Relational Distillation

A randomly initialized **4-layer / 384-dimensional student model** is trained using both the original dataset targets and the relational knowledge extracted from the frozen teacher.

The total objective combines:

1. **Task Loss:** MSE between the student's prediction and the ground-truth sentiment score.
2. **Attention Distillation Loss:** KL Divergence between teacher and student attention distributions.
3. **Value-Relation Distillation Loss:** Temperature-scaled KL Divergence between the teacher's and student's `VVᵀ` relation matrices.

The teacher remains frozen throughout this stage.

### Training

```bash
accelerate launch --num_processes=2 train_distill.py
```

---

# Performance Benchmarks

Results were measured on an **NVIDIA T4** using `benchmark_full.py`.

**Benchmark configuration:**

- Batch Size: `32`
- Sequence Length: `128`

| Metric | Teacher (12-Layer BERT) | Student (4-Layer MiniLM) | Improvement |
|---|---:|---:|---:|
| **Parameters** | 109.5M | 19.2M | **82.5% Smaller** |
| **Latency (ms/batch)** | 188.4 ms | 20.2 ms | **9.3× Faster** |
| **Test MSE (↓)** | 0.2024 | 0.0590 | **Regularized & Supervised** |
| **Test MAE (↓)** | 0.4200 | 0.2285 | **Lower Error** |

The distilled student contains only **19.2M parameters**, representing an **82.5% reduction** compared with the 109.5M-parameter teacher.

At the same time, the student achieves approximately **9.3× lower inference latency per batch**, while also achieving lower test MSE and MAE on the evaluated regression task.

---

# Project Structure

```text
.
├── config.py
├── dataset.py
├── minilm.py
├── model_teacher.py
├── model_student.py
├── train_teacher.py
├── train_distill.py
├── benchmark_full.py
└── README.md
```

### File Descriptions

| File | Description |
|---|---|
| `config.py` | Centralized hyperparameter and architectural dimension configuration. |
| `dataset.py` | ABSA pair-tokenization and dynamic continuous target mapping. |
| `minilm.py` | Core implementation of Attention (`A`) and Value-Relation (`VVᵀ`) extraction and distillation loss calculation. |
| `model_teacher.py` | Teacher model initialization with a 1D regression head and eager attention. |
| `model_student.py` | Student model initialization with a 1D regression head and eager attention. |
| `train_teacher.py` | Stage 1 teacher fine-tuning loop. |
| `train_distill.py` | Stage 2 dual-loss relational distillation training loop. |
| `benchmark_full.py` | Evaluation script for latency, parameter count, and regression metrics. |

---

# Knowledge Distillation Overview

The overall pipeline can be summarized as:

```text
                    ┌──────────────────────┐
                    │   ABSA Dataset       │
                    │ Sentence + Aspect    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Stage 1             │
                    │  BERT-base Teacher   │
                    │  12 Layers / 768d    │
                    └──────────┬───────────┘
                               │
                         Frozen Teacher
                               │
              ┌────────────────┴────────────────┐
              │                                 │
       Attention Relations              Value Relations
              │                                 │
              ▼                                 ▼
        Teacher A                         Teacher VVᵀ
              │                                 │
              │        Knowledge               │
              │        Distillation            │
              └───────────────┬─────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Stage 2           │
                    │  MiniLM Student    │
                    │  4 Layers / 384d   │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Continuous ABSA    │
                    │ Sentiment Score     │
                    │       0.0 – 1.0     │
                    └────────────────────┘
```

The key idea is that the student does not need to reproduce the teacher's hidden representations directly. Instead, it learns the **relational knowledge encoded within the teacher's self-attention mechanisms**.

This makes it possible to distill knowledge between models with different hidden dimensions while maintaining a significantly smaller student architecture.

---

# Requirements

The project is designed to run with:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Accelerate
- CUDA-compatible NVIDIA GPUs

For multi-GPU training, configure Accelerate before launching the training scripts:

```bash
accelerate config
```

Then run:

```bash
accelerate launch --num_processes=2 train_teacher.py
```

and:

```bash
accelerate launch --num_processes=2 train_distill.py
```

---

# Training Workflow

The recommended workflow is:

```text
1. Prepare environment
        ↓
2. Configure Accelerate
        ↓
3. Fine-tune BERT teacher
        ↓
4. Freeze trained teacher
        ↓
5. Initialize MiniLM student
        ↓
6. Distill Attention + Value Relations
        ↓
7. Evaluate student
        ↓
8. Benchmark latency and parameter count
```

---

# Objective

The goal of this project is to investigate whether **MiniLM-style relational knowledge distillation** can compress a large BERT-based ABSA model into a substantially smaller architecture while maintaining or improving task performance.

The final student model provides:

- **82.5% fewer parameters**
- **9.3× lower batch inference latency**
- Lower measured **MSE**
- Lower measured **MAE**
- A substantially smaller architecture suitable for more resource-constrained inference environments

The implementation focuses on **relational knowledge transfer through self-attention**, rather than conventional hidden-state matching, making it particularly suitable for studying the principles behind MiniLM-style model compression.