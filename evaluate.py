# ─────────────────────────────────────────────────────────────────
#  evaluate.py  —  Load best weights, plot results, classification report
# ─────────────────────────────────────────────────────────────────

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config  import (CLASSES, NUM_CLASSES, WEIGHTS_PATH,
                     HISTORY_PLOT, CM_PLOT, TRAIN_DIR, TEST_DIR)
from dataset import build_pipelines
from model   import ExpressionCNN


# ── Plot training history ─────────────────────────────────────────

def plot_history(history: dict, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(history['train_acc'], label='Train',      linewidth=2)
    ax1.plot(history['val_acc'],   label='Validation', linewidth=2)
    ax1.set_title('Accuracy over epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_loss'], label='Train',      linewidth=2)
    ax2.plot(history['val_loss'],   label='Validation', linewidth=2)
    ax2.set_title('Loss over epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[Evaluate] Training history saved → {save_path}")


# ── Confusion matrix ──────────────────────────────────────────────

def plot_confusion_matrix(y_true: list, y_pred: list,
                          classes: list, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=classes, yticklabels=classes,
                cmap='Blues', linewidths=0.5)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[Evaluate] Confusion matrix saved → {save_path}")


# ── Per-class metrics ─────────────────────────────────────────────

def print_classification_report(y_true: list, y_pred: list,
                                 classes: list):
    print("\n[Evaluate] Classification Report:")
    print("─" * 55)
    print(classification_report(y_true, y_pred,
                                 target_names=classes,
                                 digits=4))


# ── Run full evaluation ───────────────────────────────────────────

def evaluate(model: ExpressionCNN,
             test_dataset: tf.data.Dataset,
             history: dict | None = None):

    os.makedirs(os.path.dirname(CM_PLOT), exist_ok=True)

    # Load best saved weights
    if os.path.exists(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH)
    else:
        print("[Evaluate] WARNING: No saved weights found. "
              "Run train.py first.")
        return

    # Collect predictions
    all_preds, all_true = [], []
    for imgs, lbls in test_dataset:
        logits = model.forward(imgs, training=False)
        all_preds.extend(tf.argmax(logits, axis=1).numpy())
        all_true.extend(tf.argmax(lbls,   axis=1).numpy())

    # Reports
    print_classification_report(all_true, all_preds, CLASSES)
    plot_confusion_matrix(all_true, all_preds, CLASSES, CM_PLOT)

    # Training curves (if history file available)
    if history is not None:
        plot_history(history, HISTORY_PLOT)
    elif os.path.exists('outputs/history.json'):
        with open('outputs/history.json') as f:
            history = json.load(f)
        plot_history(history, HISTORY_PLOT)
    else:
        print("[Evaluate] No history found — skipping training curves. "
              "Run train.py first to generate history.json.")


# ── Entry point ───────────────────────────────────────────────────

if __name__ == '__main__':
    _, test_ds, _ = build_pipelines(TRAIN_DIR, TEST_DIR, NUM_CLASSES)
    model = ExpressionCNN(num_classes=NUM_CLASSES)
    evaluate(model, test_ds)
