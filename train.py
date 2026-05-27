"""
Autoresearch SVM training script. Single-file.
Manual linear SVM (NumPy only) trained under a fixed wall-clock budget.

Usage:
    python prepare.py
    python train.py
"""

import math
import time

import numpy as np

from prepare import TIME_BUDGET, load_train_val, evaluate_accuracy

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly — this is the agent-tunable file)
# ---------------------------------------------------------------------------

C = 1.0
LEARNING_RATE = 0.001
INIT_SCALE = 0.01
WARMUP_STEPS = 10  # exclude early steps from timed budget (mirrors autoresearch)

# ---------------------------------------------------------------------------
# SVM model (manual primal linear SVM, binary labels in {-1, +1})
# ---------------------------------------------------------------------------


def _to_dense(X):
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def _to_pm1(y):
    """Map encoded labels {0, 1} to SVM targets {-1, +1}."""
    y = np.asarray(y, dtype=np.float32)
    return np.where(y > 0, 1.0, -1.0)


class LinearSVM:
    """Linear SVM trained with full-batch subgradient descent on hinge loss."""

    def __init__(self, n_features, C=C, learning_rate=LEARNING_RATE, init_scale=INIT_SCALE, seed=42):
        rng = np.random.default_rng(seed)
        self.C = float(C)
        self.learning_rate = float(learning_rate)
        self.w = (init_scale * rng.standard_normal(n_features)).astype(np.float32)
        self.b = 0.0

    def train_step(self, X, y_pm1):
        # Optimized step
        scores = X @ self.w + self.b
        margins = y_pm1 * scores
        active = margins < 1.0

        grad_w = self.w.copy()
        grad_b = 0.0

        if np.any(active):
            X_active = X[active]
            y_active = y_pm1[active]
            grad_w -= self.C * (y_active @ X_active)
            grad_b = -self.C * y_active.sum()

        self.w -= self.learning_rate * grad_w
        self.b -= self.learning_rate * grad_b

    def predict(self, X):
        X = _to_dense(X)
        scores = X @ self.w + self.b
        return (scores >= 0.0).astype(np.int64)


# ---------------------------------------------------------------------------
# Training loop (fixed time budget)
# ---------------------------------------------------------------------------


def main():
    X_train, X_val, y_train, y_val = load_train_val()
    X_train = _to_dense(X_train)
    y_train_pm1 = _to_pm1(y_train)

    model = LinearSVM(n_features=X_train.shape[1])
    num_samples = X_train.shape[0]

    t_start = time.time()
    t_train_start = time.time()
    total_training_time = 0.0
    step = 0
    smooth_loss = 0.0
    ema_beta = 0.9

    # Use a slightly safer budget to ensure we print results before tool timeout
    effective_budget = min(TIME_BUDGET, 200)

    while True:
        t0 = time.time()

        model.train_step(X_train, y_train_pm1)

        if step % 100 == 0:
            scores = X_train @ model.w + model.b
            margins = y_train_pm1 * scores
            hinge = np.maximum(0.0, 1.0 - margins)
            loss = 0.5 * np.dot(model.w, model.w) + model.C * hinge.sum()
            
            smooth_loss = ema_beta * smooth_loss + (1.0 - ema_beta) * loss
            debiased_loss = smooth_loss / (1.0 - ema_beta ** (step//100 + 1))
            
            print(
                f"\rstep {step:05d} | loss: {debiased_loss:.6f} | "
                f"remaining: {max(0.0, effective_budget - total_training_time):.0f}s ",
                end="",
                flush=True,
            )

        if step > WARMUP_STEPS:
            total_training_time += time.time() - t0

        step += 1
        if step > WARMUP_STEPS and total_training_time >= effective_budget:
            break

    print()

    val_accuracy = evaluate_accuracy(model, X_val=X_val, y_val=y_val)
    val_error = 1.0 - val_accuracy

    t_end = time.time()
    startup_time = t_train_start - t_start
    total_seconds = t_end - t_start

    print("---")
    print(f"val_accuracy:     {val_accuracy:.6f}")
    print(f"val_error:        {val_error:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"startup_seconds:  {startup_time:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_samples:      {num_samples}")
    print(f"C:                {C}")
    print(f"learning_rate:    {LEARNING_RATE}")


if __name__ == "__main__":
    main()
