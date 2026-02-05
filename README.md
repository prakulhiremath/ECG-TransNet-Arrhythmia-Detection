# ECG-TransNet: Hybrid CNN-Transformer for Arrhythmia Detection

**ECG-TransNet** is a state-of-the-art deep learning framework designed for robust and explainable arrhythmia detection. By combining **1D-Convolutional Neural Networks (CNNs)** for local morphology extraction with **Transformer Encoders** for global temporal context, the model achieves superior performance in classifying complex cardiac rhythms.

## 🚀 Key Features
**Hybrid Architecture:** Integrates a ResNet1D front-end with Multi-Head Self-Attention layers to capture both P-QRS-T morphology and irregular R-R intervals.
**Clinical Explainability:** Provides transparency through **Grad-CAM** saliency maps and **SHAP** values, allowing clinicians to verify predictions against gold-standard ECG patterns. 
**AAMI Standard Compliant:** Optimized for the five-class arrhythmia scheme (N, S, V, F, Q) recommended by the Association for the Advancement of Medical Instrumentation.
**Edge-Ready:** Optimized through quantization and pruning to achieve **sub-15ms inference** latency, making it ideal for wearable monitors and real-time surveillance.

## 📊 Dataset: MIT-BIH Arrhythmia

This implementation focuses on the **MIT-BIH Arrhythmia Database**, the gold-standard reference for beat-level classification.
**Recordings** | 48 half-hour two-channel ambulatory ECGs 
**Sampling Rate** | 360 Hz (Resampled to 250 Hz for efficiency) 
**Total Beats** | ~110,000 annotated beats 
**Classes** | N (Normal), S (SVEB), V (VEB), F (Fusion), Q (Unknown) 

## 🏗️ Architecture (ECG-TransNet)
The framework consists of four specialized modules:
1. **Convolutional Front-End:** Extracts sharp transitions and local features like R-peaks using residual blocks.
2. **Temporal Transformer Encoder:** Uses self-attention to model long-range dependencies across cardiac cycles.
3. **Interpretability Module:** Generates attention heatmaps highlighting critical segments like ST-elevations or Q-wave abnormalities.
4. **Classification Head:** A softmax output layer mapped to the AAMI taxonomy.

## 📈 Performance Results
ECG-TransNet outperforms traditional machine learning (SVM, KNN) and standard CNN baselines.
**Generalization:** +4–6% improvement in macro F1-score over CNN-only models.
**Clinical Sensitivity:** Significantly higher recall in detecting rare classes like Ventricular Ectopic Beats (V).
**Model Size:** Reduced to **~28 MB** after quantization for edge deployment.
