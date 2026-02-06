# ECG-TransNet: Architecting Clinical Intelligence 🫀

### A State-of-the-Art Hybrid CNN–Transformer Framework for Arrhythmia Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.13026%2FC2F305-green.svg)](https://doi.org/10.13026/C2F305)

---

## 🧬 Overview
**ECG-TransNet** represents a paradigm shift in automated cardiac diagnostics. By synthesizing the **spatial feature extraction** power of 1D-Residual Convolutional Neural Networks (ResNet) with the **global sequence modeling** capabilities of Transformer Encoders, the framework deciphers complex cardiac morphologies and rhythmic variations with physician-level precision.

Optimized for the **AAMI EC57 standard**, ECG-TransNet specifically addresses the "Long-Tail" problem in medical datasets, ensuring high sensitivity for rare but critical arrhythmic events.

---

## 🧠 The Hybrid Philosophy: Morphology + Context
ECG signals are inherently multi-scalar. ECG-TransNet models both domains simultaneously:

1. **Morphological Domain (CNN):** Captures high-frequency micro-structures like QRS complex widening, ST-segment elevation, and T-wave inversions.
2. **Temporal Domain (Transformer):** Captures macro-rhythmic context, such as R-R interval variability and premature beats that only become apparent across multiple heartbeat cycles.



---

## 🔥 Key Technical Innovations
* **Synergistic Architecture:** 1D-CNN front-end serves as a learnable tokenizer for the Transformer backbone.
* **Clinical XAI (Explainable AI):** Native integration of **Grad-CAM** and **SHAP** to provide heatmaps of diagnostic evidence, ensuring "Black-Box" models are clinically verifiable.
* **Imbalance Resilience:** Implements weighted cross-entropy and specialized data augmentation to handle the massive prevalence of 'Normal' beats vs. 'Fusion' beats.
* **Edge-Ready Deployment:** Engineered for sub-15ms inference latency and a compact ~28MB footprint via post-training quantization.

---

## 🏗️ Architecture Deep-Dive

### 1. Convolutional Feature Encoder (The Eye)
Utilizes deep 1D-Residual blocks to learn shift-invariant patterns. These blocks identify the specific "signature" of various heartbeats regardless of baseline wander or noise.

### 2. Transformer Temporal Encoder (The Brain)
Implements Multi-Head Self-Attention (MHSA) to compute dependencies between distant beats. This allows the model to "remember" previous beat timings to identify irregular rhythms.



### 3. Interpretability & Classification
A global average pooling layer feeds into a high-density linear head, supervised by an interpretability layer that maps neural activations back to the original ECG signal.

---

## 📊 Dataset & Standards: MIT-BIH
We utilize the **MIT-BIH Arrhythmia Database**, the global benchmark for ECG signal processing.

| AAMI Class | Description | Clinical Significance |
| :--- | :--- | :--- |
| **N** | Normal / Bundle Branch Block | Baseline cardiac function |
| **S** | Supraventricular Ectopic | Atrial irregularities |
| **V** | Ventricular Ectopic | High-risk ventricular events |
| **F** | Fusion Beats | Hybrid morphological triggers |
| **Q** | Unknown / Paced | Artifacts or paced rhythms |

**Preprocessing Pipeline:**
* **Filtration:** Butterworth Bandpass (0.5 – 35 Hz) for baseline wander removal.
* **Normalization:** Z-Score standardization per patient record.
* **Segmentation:** Pan-Tompkins derived beat-centering.

---

## 📈 Performance Benchmarks
* **Overall Accuracy:** `~98.2%`
* **Avg. F1-Score:** `~0.94` (AAMI-balanced)
* **Inference Latency:** `< 12ms` (CPU-optimized)
* **Memory Footprint:** `27.4 MB` (Quantized)

---

## 📂 Modular Repository Structure
```bash
ECG-TransNet/
├── data/                  # Ingestion & Segmented Tensors
├── preprocessing/         # Signal DSP & Pan-Tompkins Logic
├── models/                # Architecture Definitions
│   ├── cnn_encoder.py     # ResNet1D Backbone
│   ├── transformer.py     # MHSA Modules
│   └── ecg_transnet.py    # Hybrid Fusion Model
├── training/              # Optimization & Schedulers
│   ├── train.py           # Distributed Training Logic
│   └── evaluate.py        # K-Fold & Metric Aggregation
├── interpretability/      # Explainable AI Suite
│   ├── gradcam.py         # Saliency Mapping
│   └── shap_analysis.py   # Feature Attribution
├── results/               # Generated Visuals & Weights
└── README.md

```markdown
## ⚙️ Installation & Usage

### 🛠️ Setup
```bash
git clone [https://github.com/prakulhiremath/ECG-TransNet-Arrhythmia-Detection](https://github.com/prakulhiremath/ECG-TransNet-Arrhythmia-Detection)
cd ECG-TransNet-Arrhythmia-Detection
pip install -r requirements.txt

```

### 🚄 Training

```bash
python training/train.py --config configs/production_run.yaml

```

### 🔍 Explaining Model Decisions

```bash
# Generate Grad-CAM heatmaps
python interpretability/gradcam.py --sample_id 100_beat_5

# Run SHAP feature attribution
python interpretability/shap_analysis.py

```

---

## 📝 Citations

### Research Reference

> Hiremath, P., Bagawan, M. (2026). **ECG-TransNet: Hybrid CNN-Transformer for Arrhythmia Detection.** GitHub: [prakulhiremath/ECG-TransNet-Arrhythmia-Detection](https://github.com/prakulhiremath/ECG-TransNet-Arrhythmia-Detection)

### Dataset & Ecosystem

> 1. Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE EB, 2001.
> 2. Goldberger AL, et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 2000.
> **DOI:** [10.13026/C2F305](https://doi.org/10.13026/C2F305)
> 
> 

---

## 🤝 Contributing & License

Distributed under the **MIT License**. Contributions that push the boundaries of cardiac AI are welcome via Pull Requests.

**Acknowledgment:** Supported by the PhysioNet research ecosystem. Built with ❤️ for Clinical Cardiology.
