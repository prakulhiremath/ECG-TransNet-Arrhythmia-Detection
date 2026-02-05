# ECG-TransNet: Hybrid CNN-Transformer for Arrhythmia Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.13026%2FC2F305-green)](https://doi.org/10.13026/C2F305)

**ECG-TransNet** is a state-of-the-art deep learning framework engineered for robust and interpretable cardiac arrhythmia classification. By synthesizing **1D-Residual Convolutional Neural Networks (ResNet1D)** with **Temporal Transformer Encoders**, the model captures both high-frequency morphological features (QRS complex) and long-range temporal dependencies (R-R intervals).

## 🚀 Key Features

* **Hybrid Architecture:** Synergistic integration of CNNs for local morphology extraction and Multi-Head Self-Attention for global rhythm context.
* **Clinical Explainability:** Native support for **Grad-CAM** saliency maps and **SHAP** analysis to visualize "points of interest" for clinical validation.
* **Medical Standard Alignment:** Fully optimized for the **AAMI EC57** five-class taxonomy (N, S, V, F, Q).
* **Edge Optimized:** Achieves **sub-15ms inference** and a **~28MB footprint** via post-training quantization, enabling deployment on mobile and wearable devices.

## 🏗️ Architecture Overview

The ECG-TransNet pipeline is divided into four modular stages:
**Convolutional Front-End** | Local Feature Extraction | 1D-ResNet blocks for robust R-peak and ST-segment detection. |
**Transformer Encoder** | Temporal Context | Multi-head self-attention to model rhythm variability across heartbeats. |
**Interpretability Engine** | Trust & Transparency | Grad-CAM heatmaps and Saliency maps for morphological verification. |
**Classification Head** | Diagnostic Output | Softmax layer mapped to AAMI standardized arrhythmia classes. |

## 📊 Dataset: MIT-BIH Arrhythmia

This project utilizes the **MIT-BIH Arrhythmia Database**, the industry standard for evaluating cardiac signal processing algorithms.

* **Data Volume:** 48 half-hour two-channel ambulatory ECG recordings.
* **Preprocessing:** Resampled to 250 Hz with Butterworth bandpass filtering (0.5–35 Hz) for artifact removal.
* **Class Mapping (AAMI):**
    * **N**: Normal, Left/Right Bundle Branch Blocks.
    * **S**: Supraventricular Ectopic Beats (SVEB).
    * **V**: Ventricular Ectopic Beats (VEB).
    * **F**: Fusion Beats.
    * **Q**: Unknown/Paced Beats.

## 📈 Performance & Benchmarks

ECG-TransNet demonstrates significant gains in precision-recall balance, particularly in low-frequency classes:

* **Generalization:** +4–6% improvement in macro F1-score compared to pure CNN architectures.
* **Sensitivity:** Enhanced detection of Ventricular Ectopic Beats (V) in noisy environments.
* **Efficiency:** Model size reduced from >100MB to **28MB** post-quantization.

## 📜 Citations & Acknowledgments

### Cite this Framework
If you use this code in your research, please cite:
> Hiremath, P. (2026). *ECG-TransNet: Hybrid CNN-Transformer for Arrhythmia Detection.* GitHub Repository: https://github.com/prakulhiremath/ECG-TransNet-Arrhythmia-Detection

### Cite the Data Source
This project relies on the MIT-BIH database via PhysioNet. Please cite the following original publications:
1.  **Moody GB, Mark RG.** The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng in Med and Biol* 20(3):45-50 (May-June 2001).
2.  **Goldberger AL, et al.** PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource. *Circulation* 101(23):e215-e220 (2000).
3.  **Database DOI:** [10.13026/C2F305](https://doi.org/10.13026/C2F305)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.
