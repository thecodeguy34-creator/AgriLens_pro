# Agrilens Pro
### Two-Stage Deep Learning & Out-of-Distribution (OOD) Protected Diagnostic Pipeline

**Author:** Khizar Hayat | **Domain:** Computer Vision & Applied Machine Learning  
**Tech Stack:** `Python 3.13` • `TensorFlow 2.21` • `Keras 3.13` • `MobileNetV2` • `Gradio`

---

## Overview

Single-stage plant disease models frequently output high-confidence false diagnoses when given unrelated crops, everyday objects, or low-quality photos. **Agrilens Pro** solves this by separating **leaf verification (Out-of-Distribution filtering)** from **disease classification** using a calibrated two-stage Computer Vision pipeline.

### System Workflow

```text
[Input Image]
      │
      ▼
┌────────────────────────────────────────┐
│  0. Pre-Inference Quality Guardrails   │ ──► Filters dark, glaring, blurry, or multi-leaf shots
└────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────┐
│  1. Stage 1: Binary Tomato Leaf Gate   │ ──► Rejects non-tomato leaves & objects (Threshold: 0.80)
└────────────────────────────────────────┘
      │ (Verified Tomato Leaf)
      ▼
┌────────────────────────────────────────┐
│  2. Stage 2: 10-Class Disease Model    │ ──► Diagnoses disease or requests retake (Threshold: 0.60)
└────────────────────────────────────────┘
```

---

## Key Technical Contributions

* **Two-Stage `MobileNetV2` Architecture:** Pairs a binary leaf gate (`98.29%` validation accuracy, `0.9967` AUC) with a 10-class pathogen classifier (`92.69%` validation accuracy), fine-tuned on an NVIDIA Tesla T4 GPU.
* **Zero-Leakage Multi-Dataset Engineering:** Curated **28,084 unique images** across six laboratory and field datasets using **MD5 cryptographic hashing** to guarantee `0%` overlap between training and test splits.
* **Statistical Threshold Calibration:** Calibrated operating cutoffs (`0.80` Gate / `0.60` Classifier) on a held-out calibration split using Wilson Score 95% confidence intervals.
* **Automated Abstention & Quality Checks (`test_app.py`):** Evaluates brightness, edge sharpness, and 6 multi-crop views prior to classification, routing uncertain field shots to a retake prompt.

---

## Dataset Corpus (28,084 Unique Images)

| Source Dataset | Environment | Role in Pipeline |
| :--- | :--- | :--- |
| **Olazo Tomato Diseases** | Controlled Lab | 10-class tomato benchmark (healthy + 9 foliar diseases) |
| **FieldPlant-11 (COCO)** | Real-World Field | Bounding-box cropped field tomato and cassava leaves |
| **PlantDoc** | Real-World Field | Field tomato leaves and unseen holdout crops (apple, corn, grape) |
| **Emmarex PlantDisease** | Controlled Lab | Hard-negative foliar controls (potato and bell pepper) |
| **Intel Image Classification** | Outdoor Scenes | Negative background controls (forests, mountains, streets, sea, buildings) |
| **Natural Images** | General Objects | Negative object controls (vehicles, animals, flowers, fruit, people) |

---

## Benchmark Performance

Evaluated on held-out test partitions with Wilson Score 95% confidence intervals:

| Evaluation Partition / Control Group | Samples (`n`) | Rejected (OOD) | Retake Prompt | Diagnostic Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Tomato Lab Test (`Olazo`)** | 4,437 | 1.1% | 3.6% | **96.2%** |
| **Tomato Field Test (`FieldPlant`)** | 276 | 4.7% | 37.3% | **93.0%** |
| **Bell Pepper Lab (Negative)** | 248 | **100.0%** | 0.0% | — |
| **Potato Lab (Negative)** | 215 | **99.1%** | 0.5% | — |
| **Cassava Field (Negative)** | 251 | **99.2%** | 0.4% | — |
| **Scenes & Objects (Negative)** | 198 | **100.0%** | 0.0% | — |

---

## Repository Contents

* **`Agrilens_pro.ipynb.ipynb`** — End-to-end Google Colab training, MD5 deduplication, threshold calibration, and statistical evaluation notebook.
* **`test_app.py`** — Standalone Gradio diagnostic web application implementing image quality pre-checks and multi-crop leaf inference.
* **`class_names.json`** — Index-to-label mapping across all 10 target tomato leaf classes.
* **`gate_threshold.json`** — Serialized Stage 1 OOD gate threshold (`0.80`) and calibration metadata.
* **`classifier_threshold.json`** — Serialized Stage 2 selective classification threshold (`0.60`).
* **`frozen_fingerprints.json`** — MD5 partition hashes verifying reproducible, leakage-free dataset splits.
