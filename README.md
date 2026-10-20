# CROMA experiments on Sen1Floods11

Two variants of the same experiment are organized by preprocessing. The existing training, model, preprocessing, and evaluation calculations are unchanged.

```text
experiments/croma_sen1floods11/
  original_preprocessing/   # Original training and test notebooks
  own_preprocessing/        # Training-statistics preprocessing notebooks (01B / 02B)
  exploration/              # Preliminary notebook
src/                        # Existing Python helpers, model, and visualization code
runs/croma_sen1floods11/
  original_preprocessing/
    train_<date>_<time>_<unique-id>/
    test_<date>_<time>_<unique-id>/
  own_preprocessing/
    train_<date>_<time>_<unique-id>/
    test_<date>_<time>_<unique-id>/
reports/legacy/             # Previously saved plots from the repository root
checkpoints/                # Existing checkpoints, kept in place
results/                    # Existing test outputs, kept in place
dataset/                    # Existing dataset, kept in place
pretrained/                 # Existing CROMA weights, kept in place
```

## Running notebooks

1. Open the training or test notebook in the appropriate preprocessing folder using your existing training environment (for example, `THESIS D RTX3070`).
2. Run the first cell. It locates the repository using relative paths, sets the working directory to the repository root, and makes `src/` importable. This works when Jupyter starts in the repository root or the notebook directory. No machine-specific absolute paths are stored.
3. Execute the notebook in order. A training run starts at its training-loop cell; a test run starts at its non-Bolivia evaluation cell. Re-executing that run-start cell creates a new output directory. Run the Bolivia and reporting cells afterward to keep both splits in the same test run.
4. For evaluation of a new training run, set `checkpoint_path` in the checkpoint-loading cell to the saved `.pt` path relative to the repository root. Existing checkpoint selections have been preserved.

Changing folders does not reset model or optimizer state. Restart the kernel and run from the beginning when starting an independent training experiment.

**Existing notebook state:** the original-preprocessing training notebook's training loop is enclosed in a triple-quoted string and therefore disabled. This was preserved; remove those quotes when you intend to train with it. Its plotting cells depend on the training variables. The own-preprocessing test notebook currently selects `croma_best_head_weighted_loss_with_scheduler.pt`; this selection was also preserved. Choose the checkpoint from the matching preprocessing experiment before evaluating.

## Run outputs

Every new training/test run folder has an experiment name in its parent path and a timestamp plus unique suffix in its name. `run.json` records the experiment, phase, time (including timezone), source notebook, pretrained weights, and checkpoint path. Test metadata identifies exactly which checkpoint was selected; it does not infer or change its preprocessing.

- Training: best checkpoint, training history, and loss/IoU figures when their plotting cells are executed.
- Testing: `non_bol/` and `bol/` predictions and metrics, category summary CSVs, and separate visualization files.

Old outputs were not relabeled because their exact provenance cannot safely be inferred. They remain in `checkpoints/` and `results/`. New outputs are ignored by Git.

## Adding a variant

Copy the matching pair of notebooks into a new sibling preprocessing folder. In each first cell, update `EXPERIMENT_NAME` and `NOTEBOOK_PATH` to match that variant. Use the same experiment name in its training and testing notebooks. Keep all file paths relative to the repository root. The experiment identifier separates its future runs from the other variants.
