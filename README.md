# ECG-TransNet 🫀
Transformer-based ECG Arrhythmia Classification using the MIT-BIH Dataset

## 📌 Overview
ECG-TransNet is a deep learning framework for automated heartbeat classification that combines
1D Convolutional Neural Networks (CNNs) with Transformer Encoders.  
The model captures local ECG morphology (QRS complexes) and long-range temporal dependencies,
making it suitable for arrhythmia detection under the AAMI EC57 standard.

## 🚀 Key Features
- Hybrid CNN–Transformer architecture
- AAMI 5-class heartbeat classification (N, S, V, F, Q)
- Robust performance on imbalanced ECG data
- Clean training & evaluation pipeline in PyTorch

## 🏗️ Model Architecture
1. **CNN Feature Extractor**  
   Extracts local morphological patterns from ECG segments.

2. **Transformer Encoder**  
   Models long-range temporal dependencies using multi-head self-attention.

3. **Classification Head**  
   Fully connected layer with Softmax activation for AAMI class prediction.

## 📊 Dataset
**MIT-BIH Arrhythmia Database (PhysioNet)**

- Sampling rate: 250 Hz
- Preprocessing: normalization & segmentation
- AAMI Class Mapping:
  - N: Normal & Bundle Branch Blocks
  - S: Supraventricular Ectopic Beats
  - V: Ventricular Ectopic Beats
  - F: Fusion Beats
  - Q: Unknown/Paced Beats

> Dataset is not included due to licensing restrictions.

## 📈 Results
- Training Accuracy: **~97–98%**
- Improved detection of minority arrhythmia classes
- Evaluation includes confusion matrix and classification report

## 📂 Project Structure
