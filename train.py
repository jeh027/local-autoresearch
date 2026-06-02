"""
Autoresearch pretraining script. Leverages API calls to machine learning models.

Workflow: Run train.py file strictly after prepare.py
Audience: Agent only modifies this file

Usage (e.g. model architecture, optimizer, training loop)
    python train.py
"""

# ---------------------------------------------------------------------------
# Import Dependencies
# ---------------------------------------------------------------------------

import time
import numpy as np
from prepare import TIME_BUDGET, load_train_val, evaluate_accuracy

# ---------------------------------------------------------------------------
# Define Constants
# ---------------------------------------------------------------------------

WARMUP_STEPS = 10

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

C = 1.0
LEARNING_RATE = 0.0008
INIT_SCALE = 0.01

# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

def _scores(X, w, b):
    """Linear scores; supports sparse CSR and dense arrays."""
    return np.asarray(X @ w, dtype=np.float32).ravel() + b


def _to_pm1(y):
    """Map encoded labels {0, 1} to SVM targets {-1, +1}."""
    y = np.asarray(y, dtype=np.float32)
    return np.where(y > 0, 1.0, -1.0)


class LinearSVM:
    """Linear SVM: weights, bias, and inference."""

    def __init__(self, n_features, C=C, init_scale=INIT_SCALE, seed=42):
        rng = np.random.default_rng(seed)
        self.C = float(C)
        self.w = (init_scale * rng.standard_normal(n_features)).astype(np.float32)
        self.b = 0.0

    def predict(self, X):
        scores = _scores(X, self.w, self.b)
        return (scores >= 0.0).astype(np.int64)

    def hinge_subgradient(self, X, y_pm1):
        """Full-batch subgradient of 0.5||w||^2 + C * sum hinge(margin)."""
        margins = y_pm1 * _scores(X, self.w, self.b)
        active = margins < 1.0

        grad_w = self.w.copy()
        grad_b = 0.0

        if np.any(active):
            grad_w -= self.C * np.asarray(X[active].T @ y_pm1[active]).ravel().astype(
                np.float32
            )
            grad_b = -self.C * y_pm1[active].sum()

        return grad_w, grad_b


# ---------------------------------------------------------------------------
# Optimizer (e.g. MuonAdamW, SGD)
# ---------------------------------------------------------------------------

class SGD:
    """Full-batch subgradient descent optimizer."""

    def __init__(self, learning_rate=LEARNING_RATE):
        self.learning_rate = float(learning_rate)

    def step(self, model, X, y_pm1):
        grad_w, grad_b = model.hinge_subgradient(X, y_pm1)
        model.w -= self.learning_rate * grad_w
        model.b -= self.learning_rate * grad_b


# ---------------------------------------------------------------------------
# Configuration (tie all the above pieces together)
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    X_train, X_val, y_train, y_val = load_train_val()
    y_train_pm1 = _to_pm1(y_train)

    model = LinearSVM(n_features=X_train.shape[1])
    optimizer = SGD()
    num_samples = X_train.shape[0]

    step = 0
    smooth_loss = 0.0
    ema_beta = 0.9

    # -----------------------------------------------------------------------
    # Training loop (Fixed Time Budget)
    # -----------------------------------------------------------------------

    t_train_start = time.time()
    total_training_time = 0.0

    while True:
        t0 = time.time()

        optimizer.step(model, X_train, y_train_pm1)

        margins = y_train_pm1 * _scores(X_train, model.w, model.b)
        hinge = np.maximum(0.0, 1.0 - margins)
        loss = 0.5 * np.dot(model.w, model.w) + model.C * hinge.sum()

        if step > WARMUP_STEPS:
            total_training_time += time.time() - t0

        smooth_loss = ema_beta * smooth_loss + (1.0 - ema_beta) * loss
        debiased_loss = smooth_loss / (1.0 - ema_beta ** (step + 1))
        remaining = max(0.0, TIME_BUDGET - total_training_time)

        print(
            f"\rstep {step:05d} | loss: {debiased_loss:.6f} | "
            f"remaining: {remaining:.0f}s ",
            end="",
            flush=True,
        )

        step += 1

        if step > WARMUP_STEPS and total_training_time >= TIME_BUDGET:
            break

    print()

    # -----------------------------------------------------------------------
    # Summary Statistics (e.g. evaluation performance, model details, peak memory)
    # -----------------------------------------------------------------------
    # Necessary for program.md to parse output into log file and results.tsv

    train_accuracy = evaluate_accuracy(model, X_train, y_train)
    train_error = 1.0 - train_accuracy

    val_accuracy = evaluate_accuracy(model, X_val, y_val)
    val_error = 1.0 - val_accuracy

    t_end = time.time()
    startup_time = t_train_start - t_start
    total_seconds = t_end - t_start

    print("---")
    print(f"train_accuracy:   {train_accuracy:.6f}")
    print(f"train_error:      {train_error:.6f}")
    print(f"val_accuracy:     {val_accuracy:.6f}")
    print(f"val_error:        {val_error:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"startup_seconds:  {startup_time:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_samples:      {num_samples}")


if __name__ == "__main__":
    main()
