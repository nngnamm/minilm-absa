# MiniLM Knowledge Distillation for Aspect-Based Sentiment Analysis (ABSA)

This repository implements a mathematically rigorous MiniLM knowledge distillation pipeline to compress a 12-layer BERT expert into a lightweight 4-layer student model. 

The pipeline is explicitly designed for continuous regression (0.0 to 1.0) on Aspect-Based Sentiment Analysis (ABSA) tasks using multi-GPU environments.

## Architecture & Mathematical Implementation

Unlike standard hidden-state distillation, this implementation replicates the original Microsoft MiniLM specification by distilling deep self-attention relationships. Because the student (384d) and teacher (768d) have different hidden dimensions, we bypass the dimension mismatch by transferring the attention distribution matrices directly.

*   **Task:** Aspect-Based Sentiment Analysis (ABSA). 
*   **Target:** Continuous Regression (Mean Squared Error). 1.0 = Positive, 0.5 = Neutral/Conflict, 0.0 = Negative.
*   **Attention Transfer (A):** KL Divergence between Teacher and Student Softmax attention probability distributions.
*   **Value-Relation Transfer (VR):** Manual extraction of Value projection layers (`attention.self.value`) to compute scaled dot-product $VV^T$ relation matrices, optimized via Temperature-scaled KL Divergence.
*   **Hardware Setup:** Distributed Data Parallel (DDP) Multi-GPU execution via Hugging Face `accelerate`.

> **Note on Eager Attention:** Hugging Face defaults to `sdpa` (Scaled Dot-Product Attention) which fuses C++ kernels and drops attention weights. This repository explicitly forces `attn_implementation="eager"` to successfully materialize and extract the matrices required for relation distillation.

## Two-Stage Training Pipeline

To ensure the student learns from a competent master, the training is split into two stages:

### Stage 1: Expert Teacher Fine-Tuning
The `bert-base-uncased` base model is trained on the `tomaarsen/setfit-absa-semeval-restaurants` dataset. Inputs are pair-tokenized (`[CLS] Sentence [SEP] Aspect [SEP]`). The model is optimized using Mean Squared Error (MSE) to accurately output a 0.0 - 1.0 sentiment score.
```bash
accelerate launch --num_processes=2 train_teacher.py