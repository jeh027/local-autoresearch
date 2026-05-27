# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, CSV download, parquet train/val split, TF-IDF + extra features, loaders, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains:
   - `data/train.csv`, `data/train.parquet`, `data/val.parquet`
   - `preprocessor/vectorizer.pkl`, `label_encoder.pkl`, `extra_scaler.pkl`
   - `preprocessor/train_features.npz`, `preprocessor/val_features.npz`  
   If missing, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## What the repo does

**Task:** classify single words as `spanish` or `french` from the CSV at the URL configured in `prepare.py`.

**`prepare.py` (fixed):**
- Downloads `train.csv` → stratified 80/20 train/val parquet splits
- Builds features per word: **char TF-IDF (1–3 grams)** + **word length**, **vowel ratio**, **consonant ratio**
- Saves sparse feature matrices and fitted preprocessors under `~/.cache/autoresearch/preprocessor/`
- Exposes `load_train_val()`, `evaluate_accuracy(model, ...)` (sklearn `accuracy_score` on the fixed val split)

**`train.py` (agent-editable):**
- Loads precomputed `(X_train, X_val, y_train, y_val)` from `prepare.py`
- Trains a **manual linear SVM** with NumPy only (`LinearSVM`: hinge loss, subgradient updates on `w`, `b`)
- Runs until `TIME_BUDGET` seconds (300s = 5 minutes) of training time after warmup steps
- Evaluates once at the end via `evaluate_accuracy`

There is **no GPU**, **no tokenizer**, and **no neural network** in this fork.

## Experimentation

Each experiment runs on CPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate_accuracy` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest `val_accuracy`.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 1 point accuracy improvement that adds 20 lines of hacky code? Probably not worth it. A 1 point accuracy improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

When the script finishes it prints a summary like this:

```
---
val_accuracy:     0.837500
val_error:        0.162500
training_seconds: 300.0
startup_seconds:  0.0
total_seconds:    300.2
num_steps:        12345
num_samples:      960
C:                1.0
learning_rate:    0.05
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```bash
grep "^val_accuracy:" run.log
```

`val_error` is `1 - val_accuracy` (for convenience only; **rank experiments by `val_accuracy`**).

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

Header and columns:

```
commit	val_accuracy	status	description
```

1. git commit hash (short, 7 chars)
2. `val_accuracy` achieved (e.g. `0.837500`) — use `0.000000` for crashes
3. status: `keep`, `discard`, or `crash`
4. short description of what this experiment tried

Example:

```
commit	val_accuracy	status	description
a1b2c3d	0.837500	keep	baseline
b2c3d4e	0.850000	keep	lower learning_rate to 0.02
c3d4e5f	0.820000	discard	increase C to 10.0
d4e5f6g	0.000000	crash	typo in gradient (IndexError)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Tune `train.py` with an experimental idea by editing the code directly
3. `git commit`
4. Run: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or flood context with live output)
5. Read results: `grep "^val_accuracy:" run.log`
6. If grep is empty, the run crashed. `tail -n 50 run.log` for the stack trace; fix easy bugs or log `crash` and move on
7. Record the row in `results.tsv` (do **not** commit `results.tsv`)
8. If `val_accuracy` **improved** (higher than best so far), advance the branch (keep the commit)
9. If `val_accuracy` is equal or worse, `git reset` to the previous best commit

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!