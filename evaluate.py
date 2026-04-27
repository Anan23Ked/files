# ─────────────────────────────────────────────────────────────────
#  evaluate.py  —  Evaluate model + auto-log results
# ─────────────────────────────────────────────────────────────────

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report,
                              confusion_matrix)

from config   import (CLASSES, NUM_CLASSES, WEIGHTS_PATH,
                      HISTORY_PLOT, CM_PLOT,
                      TRAIN_DIR, TEST_DIR,
                      BATCH_SIZE, IMG_SIZE, LR_INITIAL,
                      TFLITE_PATH)
from dataset  import build_pipelines
from model    import ExpressionCNN
from logger   import log_experiment, print_comparison_table


# ── Plots ─────────────────────────────────────────────────────────

def plot_history(history: dict, save_path: str, exp_id: str = ''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f'Training History  [{exp_id}]', fontsize=11)

    ax1.plot(history['train_acc'], label='Train',      linewidth=2)
    ax1.plot(history['val_acc'],   label='Validation', linewidth=2)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_loss'], label='Train',      linewidth=2)
    ax2.plot(history['val_loss'],   label='Validation', linewidth=2)
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save timestamped copy to logs/
    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', f'history_{exp_id}.png')
    plt.savefig(log_path,  dpi=150)
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
    print(f"[Evaluate] History plot -> {log_path}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path,
                           exp_id: str = ''):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=classes, yticklabels=classes,
                cmap='Blues', linewidths=0.5)
    plt.title(f'Confusion Matrix  [{exp_id}]')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', f'confusion_{exp_id}.png')
    plt.savefig(log_path,  dpi=150)
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
    print(f"[Evaluate] Confusion matrix -> {log_path}")
    return cm


# ── Main evaluate ─────────────────────────────────────────────────

def evaluate(model: ExpressionCNN,
             test_dataset: tf.data.Dataset,
             history: dict | None = None,
             dataset_info: dict   = None,
             tflite_size_kb: float = 0.0,
             pi_fps: float         = 0.0,
             notes: str            = '') -> str:
    """
    Run evaluation, print report, save plots, and log everything.

    Returns
    -------
    exp_id : str — the experiment ID for cross-referencing logs
    """
    os.makedirs(os.path.dirname(CM_PLOT), exist_ok=True)

    if os.path.exists(WEIGHTS_PATH.replace('.npy', '.weights.h5')):
        model.load_weights(WEIGHTS_PATH)
    else:
        print("[Evaluate] No saved weights found. Run train.py first.")
        return ''

    # ── Collect predictions ───────────────────────────────────────
    all_preds, all_true = [], []
    for imgs, lbls in test_dataset:
        logits = model.forward(imgs, training=False)
        all_preds.extend(tf.argmax(logits, axis=1).numpy())
        all_true.extend(tf.argmax(lbls,   axis=1).numpy())

    # ── Classification report ─────────────────────────────────────
    report_str  = classification_report(
        all_true, all_preds, target_names=CLASSES, digits=4)
    report_dict = classification_report(
        all_true, all_preds, target_names=CLASSES,
        digits=4, output_dict=True)

    print("\n[Evaluate] Classification Report:")
    print("─" * 55)
    print(report_str)

    # ── Auto-detect TFLite size ───────────────────────────────────
    if tflite_size_kb == 0.0 and os.path.exists(TFLITE_PATH):
        tflite_size_kb = os.path.getsize(TFLITE_PATH) / 1024

    # ── Build dataset_info if not passed in ──────────────────────
    if dataset_info is None:
        dataset_info = {'total_train': 'unknown',
                        'total_test': 'unknown'}

    # ── Build config snapshot ─────────────────────────────────────
    config_snapshot = {
        'filters'    : '64/128/256',
        'img_size'   : IMG_SIZE,
        'batch_size' : BATCH_SIZE,
        'lr_initial' : LR_INITIAL,
        'dropout'    : '0.25/0.5',
        'classes'    : CLASSES,
    }

    # ── Plots & confusion matrix ──────────────────────────────────
    # Use a temp exp_id for filenames before logger assigns the real one
    temp_id = 'latest'

    cm = plot_confusion_matrix(
        all_true, all_preds, CLASSES, CM_PLOT, exp_id=temp_id)

    if history is not None:
        plot_history(history, HISTORY_PLOT, exp_id=temp_id)
    elif os.path.exists('outputs/history.json'):
        with open('outputs/history.json') as f:
            history = json.load(f)
        plot_history(history, HISTORY_PLOT, exp_id=temp_id)
    else:
        history = {'train_loss':[], 'train_acc':[],
                   'val_loss':[], 'val_acc':[]}

    # ── Log everything ────────────────────────────────────────────
    exp_id = log_experiment(
        history        = history,
        report         = report_dict,
        conf_matrix    = cm,
        classes        = CLASSES,
        config         = config_snapshot,
        dataset_info   = dataset_info,
        tflite_size_kb = tflite_size_kb,
        pi_fps         = pi_fps,
        notes          = notes,
    )

    # ── Print comparison so far ───────────────────────────────────
    print_comparison_table()

    return exp_id


# ── Entry point ───────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--notes',  default='',  help='Experiment description')
    parser.add_argument('--pi-fps', default=0.0, type=float,
                        help='FPS measured on Pi (if available)')
    args = parser.parse_args()

    _, test_ds, counts = build_pipelines(TRAIN_DIR, TEST_DIR, NUM_CLASSES)

    dataset_info = {
        'total_train': sum(counts.values()),
        'total_test' : 'from test dir',
        'per_class'  : counts,
    }

    model = ExpressionCNN(num_classes=NUM_CLASSES)

    evaluate(
        model,
        test_ds,
        dataset_info   = dataset_info,
        pi_fps         = args.pi_fps,
        notes          = args.notes,
    )