# ─────────────────────────────────────────────────────────────────
#  logger.py  —  Experiment logger with timestamped JSON + CSV logs
#                Captures: model config, training history, eval metrics,
#                          confusion matrix, TFLite size, Pi FPS
# ─────────────────────────────────────────────────────────────────

import os
import json
import csv
import numpy as np
from datetime import datetime


LOG_DIR      = 'logs'
JSON_LOG     = os.path.join(LOG_DIR, 'experiments.json')
CSV_SUMMARY  = os.path.join(LOG_DIR, 'summary.csv')
CSV_HEADERS  = [
    'timestamp', 'experiment_id', 'notes',
    'filters',  'img_size', 'batch_size', 'epochs_run',
    'lr_initial', 'dropout',
    'train_samples', 'test_samples', 'classes',
    'best_val_acc', 'best_val_loss',
    'final_train_acc', 'final_train_loss',
    'tflite_size_kb', 'pi_fps',
    'angry_precision', 'angry_recall', 'angry_f1',
    'happy_precision', 'happy_recall', 'happy_f1',
    'sad_precision',   'sad_recall',   'sad_f1',
    'neutral_precision','neutral_recall','neutral_f1',
    'macro_f1', 'weighted_f1', 'overall_accuracy',
    'conf_matrix_flat',
]


def _ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _experiment_id() -> str:
    return datetime.now().strftime('exp_%Y%m%d_%H%M%S')


# ─────────────────────────────────────────────────────────────────
#  Core logging function — call this after each experiment
# ─────────────────────────────────────────────────────────────────

def log_experiment(
    history:        dict,
    report:         dict,
    conf_matrix:    np.ndarray,
    classes:        list,
    config:         dict,
    dataset_info:   dict,
    tflite_size_kb: float = 0.0,
    pi_fps:         float = 0.0,
    notes:          str   = '',
) -> str:
    """
    Save a complete experiment record.

    Parameters
    ----------
    history       : dict from train() — keys train_loss/acc, val_loss/acc
    report        : dict from sklearn classification_report(output_dict=True)
    conf_matrix   : np.ndarray from sklearn confusion_matrix
    classes       : list of class names
    config        : dict of hyperparameters (from config.py)
    dataset_info  : dict with train_samples, test_samples per class
    tflite_size_kb: size of exported .tflite model in KB
    pi_fps        : measured FPS on Raspberry Pi (0 if not measured yet)
    notes         : free-text description of this experiment

    Returns
    -------
    experiment_id : str  — unique ID for this run
    """
    _ensure_dirs()
    exp_id    = _experiment_id()
    timestamp = _now()

    # ── Best epoch metrics ────────────────────────────────────────
    best_val_acc  = max(history.get('val_acc',  [0]))
    best_val_loss = min(history.get('val_loss', [0]))
    final_t_acc   = history.get('train_acc',  [0])[-1]
    final_t_loss  = history.get('train_loss', [0])[-1]
    epochs_run    = len(history.get('val_acc', []))

    # ── Per-class metrics from report ────────────────────────────
    per_class = {}
    for cls in classes:
        r = report.get(cls, {})
        per_class[cls] = {
            'precision': round(r.get('precision', 0), 4),
            'recall'   : round(r.get('recall',    0), 4),
            'f1'       : round(r.get('f1-score',  0), 4),
            'support'  : int(r.get('support',     0)),
        }

    macro    = report.get('macro avg',    {})
    weighted = report.get('weighted avg', {})
    overall  = round(report.get('accuracy', 0), 4)

    # ── Full record ───────────────────────────────────────────────
    record = {
        'experiment_id'   : exp_id,
        'timestamp'       : timestamp,
        'notes'           : notes,
        'config'          : config,
        'dataset'         : dataset_info,
        'training': {
            'epochs_run'      : epochs_run,
            'best_val_acc'    : round(best_val_acc,  4),
            'best_val_loss'   : round(best_val_loss, 4),
            'final_train_acc' : round(final_t_acc,   4),
            'final_train_loss': round(final_t_loss,  4),
            'history'         : history,
        },
        'evaluation': {
            'overall_accuracy': overall,
            'per_class'       : per_class,
            'macro_avg'       : {k: round(v, 4) for k, v in macro.items()},
            'weighted_avg'    : {k: round(v, 4) for k, v in weighted.items()},
            'confusion_matrix': conf_matrix.tolist(),
        },
        'deployment': {
            'tflite_size_kb': round(tflite_size_kb, 1),
            'pi_fps'        : round(pi_fps, 1),
        },
    }

    # ── Append to JSON log ────────────────────────────────────────
    all_records = []
    if os.path.exists(JSON_LOG):
        with open(JSON_LOG, 'r') as f:
            try:
                all_records = json.load(f)
            except json.JSONDecodeError:
                all_records = []

    all_records.append(record)
    with open(JSON_LOG, 'w') as f:
        json.dump(all_records, f, indent=2)

    # ── Append to CSV summary ─────────────────────────────────────
    write_header = not os.path.exists(CSV_SUMMARY)
    with open(CSV_SUMMARY, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()

        def pc(cls, metric):
            return per_class.get(cls, {}).get(metric, 0)

        writer.writerow({
            'timestamp'         : timestamp,
            'experiment_id'     : exp_id,
            'notes'             : notes,
            'filters'           : config.get('filters', ''),
            'img_size'          : config.get('img_size', 48),
            'batch_size'        : config.get('batch_size', 32),
            'epochs_run'        : epochs_run,
            'lr_initial'        : config.get('lr_initial', ''),
            'dropout'           : config.get('dropout', ''),
            'train_samples'     : dataset_info.get('total_train', ''),
            'test_samples'      : dataset_info.get('total_test', ''),
            'classes'           : '|'.join(classes),
            'best_val_acc'      : round(best_val_acc,  4),
            'best_val_loss'     : round(best_val_loss, 4),
            'final_train_acc'   : round(final_t_acc,   4),
            'final_train_loss'  : round(final_t_loss,  4),
            'tflite_size_kb'    : round(tflite_size_kb, 1),
            'pi_fps'            : round(pi_fps, 1),
            'angry_precision'   : pc('angry',   'precision'),
            'angry_recall'      : pc('angry',   'recall'),
            'angry_f1'          : pc('angry',   'f1'),
            'happy_precision'   : pc('happy',   'precision'),
            'happy_recall'      : pc('happy',   'recall'),
            'happy_f1'          : pc('happy',   'f1'),
            'sad_precision'     : pc('sad',     'precision'),
            'sad_recall'        : pc('sad',     'recall'),
            'sad_f1'            : pc('sad',     'f1'),
            'neutral_precision' : pc('neutral', 'precision'),
            'neutral_recall'    : pc('neutral', 'recall'),
            'neutral_f1'        : pc('neutral', 'f1'),
            'macro_f1'          : round(macro.get('f1-score', 0),    4),
            'weighted_f1'       : round(weighted.get('f1-score', 0), 4),
            'overall_accuracy'  : overall,
            'conf_matrix_flat'  : str(conf_matrix.flatten().tolist()),
        })

    print(f"[Logger] Experiment saved: {exp_id}")
    print(f"         JSON  -> {JSON_LOG}")
    print(f"         CSV   -> {CSV_SUMMARY}")
    return exp_id


# ─────────────────────────────────────────────────────────────────
#  Pi FPS logger — call from inference_pi.py
# ─────────────────────────────────────────────────────────────────

def log_pi_fps(fps_readings: list, exp_id: str = '', notes: str = ''):
    """
    Log FPS measurements from Pi inference session.
    Call with a list of FPS values collected during inference_pi.py run.
    """
    _ensure_dirs()
    fps_log = os.path.join(LOG_DIR, 'pi_fps_log.json')

    entry = {
        'timestamp'  : _now(),
        'exp_id'     : exp_id,
        'notes'      : notes,
        'fps_mean'   : round(float(np.mean(fps_readings)),   2),
        'fps_min'    : round(float(np.min(fps_readings)),    2),
        'fps_max'    : round(float(np.max(fps_readings)),    2),
        'fps_std'    : round(float(np.std(fps_readings)),    2),
        'n_readings' : len(fps_readings),
        'readings'   : [round(f, 2) for f in fps_readings],
    }

    records = []
    if os.path.exists(fps_log):
        with open(fps_log) as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []

    records.append(entry)
    with open(fps_log, 'w') as f:
        json.dump(records, f, indent=2)

    print(f"[Logger] FPS log saved -> {fps_log}")
    print(f"         mean={entry['fps_mean']}  "
          f"min={entry['fps_min']}  max={entry['fps_max']}")
    return entry


# ─────────────────────────────────────────────────────────────────
#  Print comparison table across all experiments
# ─────────────────────────────────────────────────────────────────

def print_comparison_table():
    """Print a formatted comparison of all logged experiments."""
    if not os.path.exists(CSV_SUMMARY):
        print("[Logger] No experiments logged yet.")
        return

    rows = []
    with open(CSV_SUMMARY, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("[Logger] CSV is empty.")
        return

    print("\n" + "=" * 100)
    print("  EXPERIMENT COMPARISON TABLE")
    print("=" * 100)
    print(f"{'ID':<22} {'Notes':<25} {'ValAcc':>7} {'MacroF1':>8} "
          f"{'Size KB':>8} {'FPS':>5} {'Epochs':>7}")
    print("-" * 100)

    for r in rows:
        exp_id   = r.get('experiment_id', '')[-15:]
        notes    = r.get('notes', '')[:24]
        val_acc  = r.get('best_val_acc', '')
        macro_f1 = r.get('macro_f1', '')
        size_kb  = r.get('tflite_size_kb', '')
        fps      = r.get('pi_fps', '')
        epochs   = r.get('epochs_run', '')

        print(f"{exp_id:<22} {notes:<25} {val_acc:>7} {macro_f1:>8} "
              f"{size_kb:>8} {fps:>5} {epochs:>7}")

    print("=" * 100)
    print(f"  Total experiments: {len(rows)}")
    print(f"  Log files: {JSON_LOG}  |  {CSV_SUMMARY}\n")