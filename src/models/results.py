print("🔥 result.py loaded successfully")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
from scipy.io import savemat


def plot_professional_results(history, y_true, y_pred, folder='results/'):
    """
    Generates high-quality plots and exports performance metrics.
    """
    os.makedirs(folder, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

    classes = ['N', 'S', 'V', 'F', 'Q']

    # -------------------------------
    # 1. Learning Curves
    # -------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    ax[0].plot(history['loss'], label='Train Loss', linewidth=2.5)
    ax[0].plot(history['val_loss'], label='Val Loss', linewidth=2.5, linestyle='--')
    ax[0].set_title('Cross-Entropy Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(np.array(history['accuracy']) * 100, label='Train Accuracy', linewidth=2.5)
    ax[1].plot(np.array(history['val_accuracy']) * 100, label='Val Accuracy', linewidth=2.5, linestyle='--')
    ax[1].set_title('Classification Accuracy (%)')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].set_ylim(0, 100)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(f'{folder}learning_curves_final.png', bbox_inches='tight')
    plt.close()

    # -------------------------------
    # 2. Confusion Matrix
    # -------------------------------
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure(figsize=(9, 8), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".3f",
        cmap='magma',
        xticklabels=classes,
        yticklabels=classes
    )

    plt.title('Normalized Confusion Matrix (AAMI Classes)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{folder}confusion_matrix_final.png', bbox_inches='tight')
    plt.close()

    # -------------------------------
    # 3. Metric Export
    # -------------------------------
    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f'{folder}performance_metrics.csv')

    savemat(
        f'{folder}final_results_data.mat',
        {
            'train_loss': history['loss'],
            'val_loss': history['val_loss'],
            'train_acc': history['accuracy'],
            'val_acc': history['val_accuracy'],
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred
        }
    )

    print("\n✅ Results successfully saved to /results/")
