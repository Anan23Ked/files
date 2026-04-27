# ─────────────────────────────────────────────────────────────────
#  train.py  —  GradientTape training loop
# ─────────────────────────────────────────────────────────────────

import os
import numpy as np
import tensorflow as tf

from config import (EPOCHS, LR_INITIAL, LR_DECAY, LR_PATIENCE,
                    ES_PATIENCE, WEIGHTS_PATH, NUM_CLASSES,
                    TRAIN_DIR, TEST_DIR)
from dataset import build_pipelines
from model  import ExpressionCNN


# ── Loss ──────────────────────────────────────────────────────────

def cross_entropy_loss(logits, labels_oh):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_oh, logits=logits))


def batch_accuracy(logits, labels_oh):
    pred = tf.argmax(logits,    axis=1)
    true = tf.argmax(labels_oh, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))


# ── LR schedule ───────────────────────────────────────────────────

class ManualLRSchedule(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr: float):
        super().__init__()
        self._lr = float(initial_lr)

    def __call__(self, step):
        return self._lr

    def reduce(self, factor: float, min_lr: float = 1e-6):
        self._lr = max(self._lr * factor, min_lr)

    def get_config(self):
        return {'initial_lr': self._lr}

    @property
    def current(self) -> float:
        return self._lr


# ── Step functions ────────────────────────────────────────────────

def make_train_step(model, optimizer):
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model.forward(images, training=True)
            loss   = cross_entropy_loss(logits, labels)
        grads = tape.gradient(loss, model.trainable_variables())
        optimizer.apply_gradients(zip(grads, model.trainable_variables()))
        return loss, batch_accuracy(logits, labels)
    return train_step


def make_val_step(model):
    @tf.function
    def val_step(images, labels):
        logits = model.forward(images, training=False)
        return (cross_entropy_loss(logits, labels),
                batch_accuracy(logits, labels))
    return val_step


# ── Dataset verification ──────────────────────────────────────────

def verify_dataset(train_dir, test_dir, classes):
    """Print image counts and raise early if paths are wrong."""
    print("\n[Train] Verifying dataset paths...")
    all_ok = True
    for split, d in [('train', train_dir), ('test', test_dir)]:
        for cls in classes:
            path = os.path.join(d, cls)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path)
                             if f.lower().endswith(('.png','.jpg','.jpeg'))])
                print(f"  {split}/{cls:<12} : {count} images")
                if count == 0:
                    print(f"  [WARNING] No images in {path}")
                    all_ok = False
            else:
                print(f"  [ERROR] Missing folder: {path}")
                all_ok = False
    if not all_ok:
        raise RuntimeError(
            "Dataset verification failed. "
            "Check your TRAIN_DIR / TEST_DIR in config.py match your folder structure.")
    print("[Train] Dataset OK\n")


# ── Main training function ────────────────────────────────────────

def train(model, train_dataset, test_dataset) -> dict:
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

    lr_schedule = ManualLRSchedule(LR_INITIAL)
    optimizer   = tf.optimizers.Adam(learning_rate=lr_schedule)

    train_step  = make_train_step(model, optimizer)
    val_step    = make_val_step(model)

    history      = {k: [] for k in
                    ['train_loss','train_acc','val_loss','val_acc']}
    best_val_acc = 0.0
    es_counter   = 0
    lr_counter   = 0

    header = (f"{'Epoch':>6} {'T-Loss':>8} {'T-Acc':>7} "
              f"{'V-Loss':>8} {'V-Acc':>7} {'LR':>10}")
    print(header)
    print("-" * len(header))

    for epoch in range(1, EPOCHS + 1):

        t_losses, t_accs = [], []
        for imgs, lbls in train_dataset:
            l, a = train_step(imgs, lbls)
            t_losses.append(float(l))
            t_accs.append(float(a))

        v_losses, v_accs = [], []
        for imgs, lbls in test_dataset:
            l, a = val_step(imgs, lbls)
            v_losses.append(float(l))
            v_accs.append(float(a))

        t_loss = np.mean(t_losses) ; t_acc = np.mean(t_accs)
        v_loss = np.mean(v_losses) ; v_acc = np.mean(v_accs)

        for key, val in zip(history.keys(),
                            [t_loss, t_acc, v_loss, v_acc]):
            history[key].append(val)

        print(f"{epoch:>6} {t_loss:>8.4f} {t_acc:>7.4f} "
              f"{v_loss:>8.4f} {v_acc:>7.4f} "
              f"{lr_schedule.current:>10.6f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            es_counter   = 0
            lr_counter   = 0
            model.save_weights(WEIGHTS_PATH)
            print(f"         * Best saved  (val_acc={v_acc:.4f})")
        else:
            es_counter += 1
            lr_counter += 1

        if lr_counter >= LR_PATIENCE:
            lr_schedule.reduce(LR_DECAY)
            lr_counter = 0
            print(f"         v LR reduced -> {lr_schedule.current:.6f}")

        if es_counter >= ES_PATIENCE:
            print(f"\n[Train] Early stopping at epoch {epoch}.")
            break

    print(f"\n[Train] Best validation accuracy: {best_val_acc:.4f}")
    return history


if __name__ == '__main__':
    from config import CLASSES
    verify_dataset(TRAIN_DIR, TEST_DIR, CLASSES)

    print("[Train] Building pipelines...")
    train_ds, test_ds, counts = build_pipelines(TRAIN_DIR, TEST_DIR, NUM_CLASSES)

    model = ExpressionCNN(num_classes=NUM_CLASSES)
    model.summary()

    history = train(model, train_ds, test_ds)
    print("\n[Train] Done. Run evaluate.py for results.")