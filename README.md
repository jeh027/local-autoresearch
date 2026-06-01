# autoresearch

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

This repo is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for a **small classical ML** setup: an AI agent autonomously improves a **linear SVM** on a Spanish vs French word-classification task. The agent edits `train.py` only, trains for a **fixed 5-minute wall-clock budget**, and is scored on **validation accuracy** on a hold-out set prepared in `prepare.py`.

The core autoresearch idea is unchanged: you are not hand-tuning the training code — you program **`program.md`** so agents can run experiments overnight, keep what improves `val_accuracy`, and discard what does not.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (download CSV, parquet train/val split, TF-IDF + extra features), loaders, and `evaluate_accuracy`. **Do not modify** (agent read-only).
- **`train.py`** — manual NumPy linear SVM (`LinearSVM`), hinge-loss gradients, hyperparameters, and training loop. **This is the file the agent edits.**
- **`program.md`** — instructions for the autonomous experiment loop: the two-phase search (architecture → hyperparameters), gates, branching, logging, and keep/discard. **Edited by the human** to steer the research org.

### Task and features

Each row is one **word** labeled `spanish` or `french`. `prepare.py`:

1. Downloads `train.csv` from the configured GitHub URL
2. Writes stratified **train/val parquet** splits (~80/20)
3. Builds a fixed feature vector per word:
   - **Character TF-IDF** (n-grams of length 1–3)
   - **Word length**, **vowel ratio**, **consonant ratio**
4. Caches sparse matrices under `~/.cache/autoresearch/preprocessor/`

`train.py` loads those matrices and optimizes `w`, `b` with subgradient descent until `TIME_BUDGET` (300 seconds) elapses, then reports **`val_accuracy`** via the fixed eval harness in `prepare.py`.

There is **no GPU**, **no tokenizer**, and **no neural network** in this fork.

### Metric and time budget

- **Metric:** `val_accuracy` on the hold-out validation split — **higher is better**
- **Time:** training stops after **5 minutes** of timed steps (after a short warmup), matching the [original autoresearch design](https://github.com/karpathy/autoresearch)

### Experiment workflow (two phases)

`program.md` runs the search in two ordered phases so that architecture and hyperparameters aren't tuned at the same time:

- **Phase A — architecture.** Change *what the model computes* (one architectural idea per commit) while keeping hyperparameters at screening defaults. Each candidate must clear two gates:
  - **Gate 1 (can it learn?):** `train_accuracy >= 0.80`, otherwise it's logged as `gate1_fail`, discarded, and reset — never ranked.
  - **Gate 2 (can it generalize?):** among Gate 1 passers, keep the architecture that minimizes `val_error`.
- **Phase B — hyperparameters (architecture frozen).** Once the winning architecture is chosen, loop forever tuning hyperparameters and training-loop details (architecture is frozen except bugfixes), keeping whatever lowers `val_error` and preferring simpler code on ties.

Rule of thumb: does the change alter *what the network computes / how many params it has*? → Phase A. Does it change *how a fixed model is trained*? → Phase B.

### Logging results

Every experiment is one edit → `git commit` → 5-minute run → keep/reset decision, appended to `results.tsv` (tab-separated, kept **untracked**). Columns:

```
commit	train_accuracy	val_accuracy	status	description
```

`status` is `keep`, `discard`, or `crash`; crashes and `gate1_fail` rows use `0.000000` for accuracy. The description carries the `phase=A` / `phase=B` tag and a short note of what was tried.

## Quick start

**Requirements:** Python 3.10+, NumPy, and dependencies used by `prepare.py` (pandas, pyarrow, requests, scipy, scikit-learn). **CPU only.**

```bash
# 1. Install dependencies (example)
pip install numpy pandas pyarrow requests scipy scikit-learn

# 2. One-time data + feature prep (~seconds)
python prepare.py

# 3. Run a single training experiment (~5 minutes)
python train.py
```

If both commands finish without errors, you are ready for autonomous research mode.

Artifacts are stored in:

```
~/.cache/autoresearch/
  data/train.csv, train.parquet, val.parquet
  preprocessor/vectorizer.pkl, label_encoder.pkl, extra_scaler.pkl
  preprocessor/train_features.npz, val_features.npz
```

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
README.md       — overarching view of autoresearch
```

## Example output

After `python train.py`:

```
---
train_accuracy:   0.910000
train_error:      0.090000
val_accuracy:     0.837500
val_error:        0.162500
training_seconds: 300.0
startup_seconds:  1.2
total_seconds:    301.4
num_steps:        12345
num_samples:      960
```

Parse the key metric from a log file:

```bash
grep "^val_accuracy:" run.log
```

## Design choices

- **Single file to modify.** The agent only touches `train.py` — small diffs, easy review.
- **Fixed time budget.** Every run gets the same training wall clock (~5 minutes), so experiments are comparable even when hyperparameters change.
- **Fixed data and eval.** Splits and features live in `prepare.py`; all runs share the same `val_accuracy` definition.
- **NumPy-only model.** The SVM in `train.py` uses no sklearn/torch for training — only NumPy for `w`, `b`, and gradients. Preprocessing in `prepare.py` may use sklearn (TF-IDF, scaling, metrics).

## Tuning ideas (in `train.py` only)

- `C` — tradeoff between margin width and hinge penalty
- `LEARNING_RATE` — step size for subgradient updates
- `INIT_SCALE` — weight initialization
- `WARMUP_STEPS` — steps excluded from the timed budget
- Full-batch vs mini-batch updates, learning-rate decay, gradient clipping

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Upstream

This fork follows the experiment loop from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). For the original LLM / nanochat / GPU setup and community forks (macOS, MLX, Windows, AMD), see that repository.

## License

MIT
