import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
from scipy.io import savemat

def plot_professional_results(history, y_true, y_pred, folder='results/'):
    """
    Generates high-fidelity visualizations and statistical exports.
    """
    os.makedirs(folder, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    classes = ['N', 'S', 'V', 'F', 'Q']
    
    # --- 1. Learning Curves (High Resolution) ---
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # Loss Plot: Log scale often helps show convergence detail
    ax[0].plot(history['loss'], label='Train Loss', linewidth=2.5, color='#1f77b4')
    ax[0].plot(history['val_loss'], label='Val Loss', linewidth=2.5, color='#ff7f0e', linestyle='--')
    ax[0].set_title('A. Cross-Entropy Loss Convergence', fontweight='bold', pad=15)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend(frameon=True)
    
    # Accuracy Plot
    ax[1].plot(np.array(history['accuracy']) * 100, label='Train Acc', linewidth=2.5, color='#2ca02c')
    ax[1].plot(np.array(history['val_accuracy']) * 100, label='Val Acc', linewidth=2.5, color='#d62728', linestyle='--')
    ax[1].set_title('B. Classification Accuracy (%)', fontweight='bold', pad=15)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].set_ylim([0, 105])
    ax[1].legend(frameon=True)
    
    plt.tight_layout()
    plt.savefig(f'{folder}learning_curves_final.png', bbox_inches='tight')

    # --- 2. Advanced Confusion Matrix ---
    # We use a square figure to prevent class label stretching
    plt.figure(figsize=(9, 8), dpi=300)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    
    # Using 'magma' as it's perceptually uniform and handles low values well
    ax_cm = sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap='magma', 
                        xticklabels=classes, yticklabels=classes,
                        cbar_kws={'label': 'Sensitivity / Recall'},
                        annot_kws={"size": 11, "weight": "bold"})
    
    plt.title('Normalized Confusion Matrix: AAMI Standards', fontweight='bold', size=16, pad=20)
    plt.ylabel('Physician Label (Actual)', fontweight='bold')
    plt.xlabel('ECG-TransNet Prediction', fontweight='bold')
    plt.savefig(f'{folder}confusion_matrix_final.png', bbox_inches='tight')

    # --- 3. Statistical Export for MATLAB/Excel ---
    # Save as CSV for quick reading
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f'{folder}performance_metrics.csv')

    # Save as .MAT for MATLAB plotting (High-End engineering plots)
    mat_data = {
        'train_loss': history['loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['accuracy'],
        'val_acc': history['val_accuracy'],
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred
    }
    savemat(f'{folder}final_results_data.mat', mat_data)

    print(f"✅ Success! Generated:")
    print(f"   - {folder}learning_curves_final.png")
    print(f"   - {folder}confusion_matrix_final.png")
    print(f"   - {folder}performance_metrics.csv")
    print(f"   - {folder}final_results_data.mat")
