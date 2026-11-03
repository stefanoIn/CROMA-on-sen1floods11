# GeoFM experiments

Experiments comparing geospatial foundation models on flood segmentation. Start with a notebook in `experiments/`; its outputs go to the matching location in `runs/`.

For the detailed notebook/source guide, saved results and metric definitions, read the [project report](reports/PROJECT_REPORT.md).

## Folder guide

```text
datasets/
  sen1floods11/
    sen1floods11_hand_labeled_only/  # Images, hand labels and official splits
    WeaklyLabeled/                  # Weak labels and their acquisitions
pretrained/
  croma/                           # Downloaded CROMA backbone
  terramind/                       # Downloaded TerraMind backbone
  sen1floods11_authors/             # Models released by the dataset authors
experiments/
  sen1floods11/
    croma/
      fcn_s1/
      fcn_s2/
      linear_s1/
      fcn_s1_own_preprocessing/
      upernet_s1/
      upernet_s2/
      upernet_s1s2/
      exploration/
    terramind/fcn_s1/              # TerraMind S1 FCN training and evaluation
runs/
  sen1floods11/
    croma/<experiment>/<run>/       # Checkpoints, metrics, predictions and plots
    terramind/                     # Future TerraMind runs
    authors_baseline/              # Existing author-model evaluation outputs
reports/                           # Analyses and exports
src/                               # Python code imported by notebooks
.geofm-root                        # Marker used to find the project root
```

S1 means Sentinel-1 radar (VV and VH). S2 means Sentinel-2 optical imagery. S1S2 means both modalities. FCN, linear and UPerNet describe the segmentation head attached to CROMA.

The physical project directory is currently `D:\UNI\CROMA_experiment`. Active notebooks use paths relative to its root.

## Choose an experiment

These folders live under `experiments/sen1floods11/croma/`. Their results use the identically named folder under `runs/sen1floods11/croma/`. Notebook names beginning with `01` are for training; `02` is evaluation; `00` is exploration.

| Experiment | Input and purpose | Previous notebook/run folder name |
| --- | --- | --- |
| [fcn_s1](experiments/sen1floods11/croma/fcn_s1/) | S1; original preprocessing experiment, FCN head | `original_S1` |
| [fcn_s2](experiments/sen1floods11/croma/fcn_s2/) | S2; FCN head | `original_S2` |
| [linear_s1](experiments/sen1floods11/croma/linear_s1/) | S1; linear segmentation head | `linear_probing_S1` / `original_S1_linear_probing` |
| [fcn_s1_own_preprocessing](experiments/sen1floods11/croma/fcn_s1_own_preprocessing/) | S1; FCN with normalization based on training statistics | `own_preprocessing_S1` / `original_S1_own_preprocessing` |
| [upernet_s1](experiments/sen1floods11/croma/upernet_s1/) | S1; UPerNet | `UperNet_head_S1` |
| [upernet_s2](experiments/sen1floods11/croma/upernet_s2/) | S2; UPerNet | `UperNet_Head_S2` |
| [upernet_s1s2](experiments/sen1floods11/croma/upernet_s1s2/) | S1 and S2; UPerNet | `UperNet_head_S1S2` |
| [exploration](experiments/sen1floods11/croma/exploration/) | Preliminary inspection | `exploration` |

TerraMind Large weights are in `pretrained/terramind/`. Its S1 FCN experiment is now prepared:

- [TerraMind S1 training](experiments/sen1floods11/terramind/fcn_s1/01_terramind_on_sen1floods11_training.ipynb)
- [TerraMind S1 evaluation](experiments/sen1floods11/terramind/fcn_s1/02_terramind_on_sen1floods11_test.ipynb)

Install `terratorch==1.2.5` and `torchgeo==0.8.1` in the notebook kernel once (`%pip install terratorch==1.2.5 torchgeo==0.8.1`), restart the kernel, then run training and evaluation in order. TerraTorch supplies the architecture; the loops remain ordinary PyTorch. The local checkpoint is loaded with complete encoder-key/shape checks; no weights are downloaded by the notebooks.

Outputs go to `runs/sen1floods11/terramind/fcn_s1/`. The experiment follows the existing CROMA S1 FCN recipe: frozen backbone, the same 1024-to-256-to-2 FCN with GroupNorm, 100 epochs, batch 16, weighted CE [1,8], AdamW/cosine warm restarts, validation each epoch and selection by validation water IoU. Evaluation keeps the same 120-pixel windows, 60-pixel stride, full split membership, pooled metrics, AW/PW/FW conventions and prediction/error plots. The test notebook discovers a completed compatible training run or accepts an explicit checkpoint.

Model-specific differences are recorded rather than hidden: TerraMind uses its native S1GRD normalization and 16-pixel patches. The wrapper pads each 120-pixel crop to 128, interpolates logits to the padded extent, then crops back without resizing the labels. Its final token grid is 8x8 versus CROMA's 15x15. The frozen encoder is processed in chunks of four; the FCN still receives the full training batch. Seed, split membership/hashes, package versions and pretrained checkpoint hash are saved. Existing author/CROMA runs and notebooks are unchanged; this is a comparison of those concrete pipelines, not a claim that every input representation or historical training condition is identical.

The implementation has passed a real-checkpoint CPU forward/gradient smoke check and evaluation-artifact checks. Full training/testing has not been run. The earlier project report remains a snapshot of the results available before this TerraMind addition.

Experiment folders retain older protocol variants. For example, `fcn_s1` contains an older DeepLab-head training run. Original run names and metadata distinguish those runs; a folder name alone does not establish which protocol produced a result.

## Running notebooks

1. Open the chosen training notebook with the existing Python environment containing PyTorch, TorchVision, Rasterio and the notebook dependencies.
2. Restart the kernel and run cells in order. The first cell finds `.geofm-root` by searching parent directories and makes `src/` importable.
3. The training-loop cell creates a fresh `train_...` folder. Its printed path identifies the checkpoint and history for that execution.
4. Open the matching evaluation notebook. Inspect its checkpoint-selection cell and printed checkpoint path: notebooks may discover a recent checkpoint automatically or use an explicit selection.
5. Run both non-Bolivia and Bolivia evaluation, followed by metrics and plotting cells. These save a separate `test_...` run.

Each notebook contains its own scientific settings. The protocols differ between experiments; this reorganization does not standardize losses, normalization, schedules or validation frequency.

## Data and saved results

Official hand-labelled splits live in `datasets/sen1floods11/sen1floods11_hand_labeled_only/flood_handlabeled/`: 252 training, 89 validation, 90 non-Bolivia test and 15 Bolivia test observations. Some experiments explicitly exclude unusable training samples; consult their dataset cells. Bolivia is evaluated separately.

Within `HandLabeled/`, `S1Hand/` and `S2Hand/` hold inputs, `LabelHand/` holds reference masks, and `JRCWaterHand/` supplies the permanent-water layer for AW/PW/FW analysis. Dataset contents and split filenames are preserved.

Downloaded weights belong in `pretrained/<model>/`. Trained checkpoints stay inside their associated `train_...` run. Training and evaluation are separate runs, linked through the checkpoint path in `run.json`.

Run contents vary by experiment: histories and checkpoints, evaluation JSON/CSV metrics, prediction rasters, and plots. Existing run names and internal file layouts are retained. Saved notebook outputs and notebook copies inside old runs are historical snapshots and may display former paths.

Distinguish pooled water IoU and binary mIoU from mean per-chip water IoU. AW, PW and FW mean all water, permanent water and flood water. Check the saved metric definitions before comparing protocols.

The original `runs/sen1floods11_reproduction/` outputs are now in `runs/sen1floods11/authors_baseline/`. The corresponding experiment source notebooks were already absent before this reorganization.

## Shared Python code

| File | Responsibility |
| --- | --- |
| [use_croma.py](src/use_croma.py) | CROMA architecture and pretrained-weight loading |
| [helper.py](src/helper.py) | Original S1 data utilities and CROMA/FCN wrapper |
| [croma_upernet_pangaea.py](src/croma_upernet_pangaea.py) | CROMA/UPerNet components and associated data utilities |
| [segmentation_metrics.py](src/segmentation_metrics.py) | Segmentation metrics and reporting helpers |
| [sen1floods11_visualization.py](src/sen1floods11_visualization.py) | Prediction and error-map visualization |
| [sar_input_quality.py](src/sar_input_quality.py) | SAR input-quality checks |

## Adding models and datasets

Use `experiments/<dataset>/<model>/<experiment>/`, with the same hierarchy under `runs/`. Add data under `datasets/<dataset>/` and downloaded weights under `pretrained/<model>/`. Keep the notebook's `EXPERIMENT_NAME` and `NOTEBOOK_PATH` consistent with its location.

Large local data, weights, runs and generated reports are excluded from Git. The project report linked above is tracked. The local reorganization record (`reports/folder_reorganization.json`) contains old-to-new paths, integrity checks and references already missing before the move. No training or metric recomputation is part of the folder reorganization.
