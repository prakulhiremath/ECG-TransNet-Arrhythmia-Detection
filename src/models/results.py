import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def plot_professional_results(history, y_true, y_pred, folder='results/'):
    # Set high-level aesthetics for a medical journal look
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    classes = ['N', 'S', 'V', 'F', 'Q']
    
    # --- 1. Learning Curves with Confidence Shading ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # Loss Curve
    sns.lineplot(data=history['loss'], ax=ax[0], label='Training', linewidth=2, color='#1f77b4')
    sns.lineplot(data=history['val_loss'], ax=ax[0], label='Validation', linewidth=2, color='#ff7f0e')
    ax[0].set_title('Cross-Entropy Loss Evolution', fontweight='bold')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    
    # Accuracy Curve
    sns.lineplot(data=history['accuracy'], ax=ax[1], label='Training', linewidth=2, color='#1f77b4')
    sns.lineplot(data=history['val_accuracy'], ax=ax[1], label='Validation', linewidth=2, color='#ff7f0e')
    ax[1].set_title('Classification Accuracy', fontweight='bold')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(f'{folder}learning_curves_pro.png')

    # --- 2. Normalized Confusion Matrix Heatmap ---
    plt.figure(figsize=(8, 7), dpi=300)
    cm = confusion_matrix(y_true, y_pred)
    # Normalize by row (Actual class) to show Sensitivity
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    
    # Use 'magma' or 'Blues' for professional contrast
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='magma', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Sensitivity (Recall)'},
                annot_kws={"size": 12, "weight": "bold"})
    
    plt.title('Normalized Confusion Matrix (AAMI Standards)', fontweight='bold', pad=20)
    plt.ylabel('Actual Label (Gold Standard)')
    plt.xlabel('ECG-TransNet Prediction')
    plt.savefig(f'{folder}confusion_matrix_pro.png')
    
    # --- 3. Text Report for MATLAB/Excel ---
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f'{folder}metrics_report.csv')

    print("🚀 Professional plots and CSV report generated.")
