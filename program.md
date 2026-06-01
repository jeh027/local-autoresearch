# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from main branch.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, CSV download, parquet train/val split, TF-IDF + additional features, loaders, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains:
   - `data/train.csv`, `data/train.parquet`, `data/val.parquet`
   - `preprocessor/vectorizer.pkl`, `label_encoder.pkl`, `extra_scaler.pkl`
   - `preprocessor/train_features.npz`, `preprocessor/val_features.npz`  
   If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Repository Objective and File Roles

**Task:** Classify single words as `spanish` or `french` from the CSV at the URL configured in `prepare.py`.

**`prepare.py` (noneditable):**
- Downloads `train.csv` → stratified 80/20 train/val parquet splits
- Builds features per word: **char TF-IDF (1–3 grams)** + **word length**, **vowel ratio**, **consonant ratio**
- Saves sparse feature matrices and fitted preprocessors under `~/.cache/autoresearch/preprocessor/`
- Exposes `load_train_val()`, `evaluate_accuracy(model, ...)`

**`train.py` (agent-editable):**
- Loads precomputed `(X_train, X_val, y_train, y_val)` from `prepare.py`
- Trains a **manual linear SVM** with NumPy only (`LinearSVM`: hinge loss, subgradient updates on `w`, `b`)
- Runs until `TIME_BUDGET` seconds (300s = 5 minutes) of training time after warmup steps
- Evaluates once at the end via `evaluate_accuracy`

There is **no GPU**, **no tokenizer**, and **no neural network** in this fork.

## Experimentation

Each experiment runs on CPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and training constants (time budget, feature engineering, etc).
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate_accuracy` function in `prepare.py` is the ground truth metric.
- Merge or delete branches in the GitHub repository

**The goal is simple: get the highest `val_accuracy`.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. The only constraint is that the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 1 point accuracy improvement that adds 20 lines of hacky code? Probably not worth it. A 1 point accuracy improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is with preexisting hyperparameters.

## Phase Scope (what to change when)

Follow this two-phase framework for editing `train.py`. This section defines *what* each phase changes and *why*; the concrete step-by-step loop lives in **The experiment loop** below.

**Phase A — architecture.** Change model architecture only. While searching architecture, keep hyperparameters fixed at sensible values that you infer through reasoning about the model and data, held constant across all architecture candidates for a fair comparison (5-min train). Exit when all meaningful and effective model architectures have been exhausted; then select the architecture that passes Gate 1 AND minimizes validation error to move on to Phase B.
   - Gate 1:  Can this architecture learn at all?  → pass / fail           (>= 80% training accuracy)
   - Gate 2:  Can this architecture generalize at all?  → keep / discard   (min(val_error))
   - **Objective:** Find an architecture powerful enough to fully learn the data and simple enough to generalize, using default screening hyperparameter profiles.
   - **Purpose:** Filter broken or under-capacity architectures.

**Phase A exit criteria (then switch to Phase B):**
- **Plateau:** no Gate 2 val_error improvement for **15 consecutive** discard runs, OR
- **Queue exhausted:** sensible architectural candidates from the list above are tried, OR
- **Clear winner:** one config beats recent variants by a margin you judge meaningful.

**Phase B — hyperparameters (architecture frozen).** Change hyperparameters and training-loop details only, once the model architecture is chosen. Architecture is frozen except bugfixes.
   - **Objective:** With architecture **frozen**, find the lowest val_error by tuning training hyperparameters and loop details.
   - **Purpose:** Search for optimal hyperparameters to achieve best performance on the validation set.

When in doubt, ask: does this change *what the network computes* or *how many params/FLOPs it has*? → Phase A. Does it change *how you train a fixed graph*? → Phase B.
Document everything into `results.tsv` description section and refer back to TSV so identical edits are not repeated.


## Output format

When the script finishes it prints a summary like this:

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

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```bash
grep "^val_accuracy:" run.log
```

`val_error` is `1 - val_accuracy` (for convenience only; **rank experiments by `val_accuracy`**).

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

Header and columns:

```
commit	train_accuracy	 val_accuracy	status	description
```

1. git commit hash (short, 7 chars)
2. `train_accuracy` achieved (e.g. `0.910000`) — use `0.000000` for crashes
3. `val_accuracy` achieved (e.g. `0.837500`) — use `0.000000` for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what this experiment tried (document everything)

Example:

```
commit	train_accuracy	val_accuracy	status	description
a1b2c3d	0.912000	0.838000	keep	phase=A baseline linear SVM (char tfidf 1-3grams + extras)
b2c3d4e	0.731000	0.000000	discard	phase=A extras-only (drop tfidf), gate1_fail train_acc=0.73
c3d4e5f	0.946000	0.851000	keep	phase=A add quadratic feature interactions
d4e5f6a	0.000000	0.000000	crash	phase=A degree-3 poly expand, MemoryError on dense map
e5f6a7b	0.961000	0.847000	discard	phase=A wider poly map, val worse than best (overfit)
f6a7b8c	0.949000	0.857000	keep	phase=B lower learning_rate 0.05 -> 0.02
a7b8c9d	0.947000	0.862000	keep	phase=B C=0.5 (stronger regularization)
b8c9d0e	0.952000	0.861000	discard	phase=B cosine lr decay, ~equal val + more complexity
c9d0e1f	0.945000	0.864000	keep	phase=B mini-batch SGD, batch=256
d0e1f2a	0.000000	0.000000	crash	phase=B unnormalized grad, overflow -> NaN loss
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

This is the authoritative step-by-step loop (see **Phase Scope** above for what each phase changes and why): you run **Phase A (architecture)** first, then **Phase B (hyperparameters)** on the frozen winner. Every experiment is one edit → commit → 5-minute run → decision, recorded in `results.tsv`.

**Phase A — architecture search.** Repeat until architectures are exhausted (no `keep` for many consecutive tries) or a clear winner emerges:

1. Look at git state: current branch/commit, best Phase A val_error so far among Gate 1 passers.
2. Change **one architectural idea** in `train.py`. Do not tune hyperparameters.
3. `git commit`
4. Run: `python train.py > run.log 2>&1`
5. Read results: `grep "^train_accuracy:\|^train_error:\|^val_accuracy:\|^val_error:" run.log`
6. If grep is empty, the run crashed. Run `tail -n 50 run.log`, attempt a trivial fix if obvious; otherwise log `crash`, reset, continue.
7. **Gate 1 (can it learn?):** if `train_accuracy < 0.80`, record the row with status `discard` and `gate1_fail` noted in the description (use `0.000000` for `val_accuracy`), `git reset`, and continue — do not rank it.
8. **Gate 2 (can it generalize?):** for Gate 1 passers only, compare `val_error` (= `1 - val_accuracy`) against the best passer so far.
9. Record results in TSV with `phase=A` in the description (do **not** commit `results.tsv` — keep it untracked).
10. If val_error improved (lower than the best passer so far), advance the branch (keep the commit). If equal or worse, `git reset` back to where you started.
11. Once architecture has been finalized, advance branch with chosen architecture implemented

**Phase B — hyperparameter search (architecture frozen).** LOOP FOREVER (until human stops you):

1. Look at the git state: current branch/commit, best val_error so far.
2. Change **hyperparameters or training loop** only in `train.py`. One main idea per commit.
3. `git commit`
4. Run: `python train.py > run.log 2>&1`
5. Read results: `grep "^train_accuracy:\|^train_error:\|^val_accuracy:\|^val_error:" run.log`
6. If grep is empty, crashed. Run `tail -n 50 run.log`, attempt fix if trivial; otherwise log `crash`, reset, continue.
7. Record results in TSV with `phase=B` in the description (do **not** commit `results.tsv` — keep it untracked).
8. If val_error improved (lower), advance the branch. When val_error is nearly equal: prefer fewer params, lower complexity, simpler code.
9. If val_error is equal or worse, `git reset` back to where you started.

No further architecture changes in Phase B — the architecture was already validated in Phase A.


The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
