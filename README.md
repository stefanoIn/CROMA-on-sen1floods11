# CROMA experiments on Sen1Floods11

Two variants of the same experiment are organized by preprocessing. The folder reorganization preserved the existing calculations. The own-preprocessing variant now explicitly uses floating-point normalization without uint8 quantization, as described below.

```text
experiments/croma_sen1floods11/
  original_preprocessing/   # Original training and test notebooks
  own_preprocessing/        # Training-statistics preprocessing notebooks (01B / 02B)
  exploration/              # Preliminary notebook
src/                        # Existing Python helpers, model, and visualization code
runs/croma_sen1floods11/
  original_preprocessing/
    train_<model-name>_<date>_<time>_<unique-id>/
    test_<model-name>_<date>_<time>_<unique-id>/
  own_preprocessing/
    train_<model-name>_<date>_<time>_<unique-id>/
    test_<model-name>_<date>_<time>_<unique-id>/
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

**Existing notebook state:** the original-preprocessing training notebook's training loop is enclosed in a triple-quoted string and therefore disabled. This was preserved; remove those quotes when you intend to train with it. Its plotting cells depend on the training variables. The own-preprocessing test notebook requires an explicit `checkpoint_path` from a new float-preprocessing training run; it rejects checkpoints without matching preprocessing metadata.

## Run outputs

Every new training/test run folder has an experiment name in its parent path and the model name, timestamp, and unique suffix in its name. `run.json` records the experiment, model name, phase, time (including timezone), source notebook, pretrained weights, and checkpoint path. Test metadata identifies exactly which checkpoint was selected; it does not infer or change its preprocessing.

- Training: best checkpoint, training history, and loss/IoU figures when their plotting cells are executed.
- Testing: `non_bol/` and `bol/` predictions and metrics, category summary CSVs, and separate visualization files.

Old outputs were not relabeled because their exact provenance cannot safely be inferred. They remain in `checkpoints/` and `results/`. New outputs are ignored by Git.

## Adding a variant

Copy the matching pair of notebooks into a new sibling preprocessing folder. In each first cell, update `EXPERIMENT_NAME` and `NOTEBOOK_PATH` to match that variant. Use the same experiment name in its training and testing notebooks. Keep all file paths relative to the repository root. The experiment identifier separates its future runs from the other variants.

## Own-preprocessing float experiment

Both 01B and 02B normalize raw SAR with fixed training-channel mean +/- two standard deviations, clamp to [0, 1], and retain float32 values without uint8 conversion. Training computes statistics from finite training pixels; new checkpoints and run metadata store those statistics, the quantization setting, channel order, and nonfinite fill rule. Testing loads the statistics from the selected checkpoint, rather than hard-coding them.

Restart the training kernel and run 01B from the beginning. Copy the printed checkpoint path into 02B's `checkpoint_path` setting (as a relative `Path` or string), then run 02B from the beginning. Older baseline/quantized checkpoints cannot be used as evidence for this training protocol. The pretrained CROMA backbone is reused; the segmentation head must be trained under the new protocol. Model names include `float` to distinguish new runs.

The upstream [CROMA README](https://github.com/antofuller/CROMA/blob/main/README.md) provides both quantized and continuous normalization branches. Its example calculates channel statistics over the supplied tensor, whereas this experiment fixes them from the training split. This is a controlled adaptation, not a claim of exact pretraining reproduction or improved performance.

Nonfinite replacement remains NaN -> -50 and +/-Inf -> 0, consistently across training and testing. NaNs exist in all four local splits; declared SAR NoData is NaN. With these statistics, the current replacements map to extreme normalized values. Testing channel-mean replacement would be a separate experiment. Label handling, split membership, model architecture, and metric calculations remain unchanged by this normalization update.

## Authors' Sen1Floods11 reproduction

The four authors' distributed FCN models and the original Test_Modelsv0v0 evaluation workflow are available under [experiments/sen1floods11_reproduction/authors](experiments/sen1floods11_reproduction/authors/README.md). The completed [native-resolution visual comparison](runs/sen1floods11_reproduction/comparison_2026-10-20_18-31-06_611597_d03e130b/index.html) includes v1.0 reproduction, v1.1 comparison with CROMA, saved prediction GeoTIFFs, and separate original-notebook and integer-label metrics.
