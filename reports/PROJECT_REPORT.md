# GeoFM experiments: project guide and results

Snapshot: 2 November 2026. This report describes the current files and locally saved experiments. It is readable on GitHub without downloading the datasets, checkpoints or prediction images.

The project evaluates **CROMA Large on Sen1Floods11 flood segmentation** with Sentinel-1 radar (S1), Sentinel-2 optical imagery (S2), and joint S1+S2 input. There are **17 active notebooks (15 CROMA and 2 TerraMind), 13 CROMA training records and 12 CROMA evaluation records**. These include interrupted training and repeated evaluations, not 25 independent completed experiments. TerraMind Large weights are downloaded and its S1 FCN training/evaluation notebooks are now prepared. No TerraMind training or full-test results are available yet.

The main UPerNet comparison reaches **79.78% mIoU / 64.94% water IoU with S1**, **89.88% / 82.40% with S2**, and **88.46% / 79.96% with S1+S2**, on the non-Bolivia test set. The joint model slightly exceeds S2 on the separate Bolivia test. These are individual runs, not multi-seed averages.

## Contents

- [Project layout](#project-layout)
- [Work completed](#work-completed)
- [Notebooks and protocols](#notebooks-and-protocols)
- [How CROMA and UPerNet connect](#how-croma-and-upernet-connect)
- [TerraMind S1 FCN addition](#terramind-s1-fcn-addition)
- [Every module in src](#every-module-in-src)
- [Saved files and metric definitions](#saved-files-and-metric-definitions)
- [Selected UPerNet results](#selected-upernet-results)
- [Complete run inventory](#complete-run-inventory)
- [Interpretation and literature](#interpretation-and-literature)
- [Reproducibility and GitHub](#reproducibility-and-github)

## Project layout

The physical checkout remains `CROMA_experiment`; the internal organization is dataset-first and supports other models.

```text
datasets/sen1floods11/
    sen1floods11_hand_labeled_only/   Inputs, hand labels, reference layers, splits
    WeaklyLabeled/                   Weak labels and corresponding acquisitions
pretrained/
    croma/                          Downloaded CROMA backbone
    terramind/                      Downloaded TerraMind backbone
    sen1floods11_authors/            Dataset authors' released weights
experiments/sen1floods11/
    croma/<experiment>/             Active training/evaluation notebooks
    terramind/fcn_s1/               TerraMind S1 FCN training/evaluation
runs/sen1floods11/
    croma/<experiment>/<run>/       Saved training or evaluation execution
    authors_baseline/               Existing author-model evaluations
    terramind/                      Local placeholder
reports/                            This report; local analyses/exports/audit
src/                                Six shared Python modules
.geofm-root                         Root-discovery marker
README.md                           Short guide and running instructions
.gitignore                          Exclusions for large/generated local files
```

`pretrained/` contains downloaded starting weights. Trained heads/checkpoints belong to their training runs in `runs/`. Existing runs retain their original internal layouts; not every old run has `metrics/` or `checkpoints/` subdirectories.

Inside `HandLabeled/`, `S1Hand/` contains VV/VH radar, `S2Hand/` optical bands, `LabelHand/` reference water masks, `JRCWaterHand/` the permanent-water reference, and `S1OtsuLabelHand/` an additional radar-derived label product. The comparisons here use `LabelHand/` targets. A label folder's presence does not imply its use in training.

CSV splits under `flood_handlabeled/` contain **252 training, 89 validation, 90 non-Bolivia test and 15 Bolivia test observations**. S1/S2 experiments use corresponding observations and reference masks. The selected UPerNet runs exclude fully ignored training observations and audit unusable inputs, leaving 251 effective training samples. CSV membership is preserved; the main test tables retain the official test lists.

Weakly labelled data are present locally. Results summarized here are hand-labelled experiments and do not establish a weak-supervision or pseudo-labelling result.

## Work completed

1. Established S1 experiments with a frozen CROMA encoder and FCN head, a training-statistics preprocessing variant, and a linear head.
2. Added S2 FCN training/evaluation with optical band selection and normalization.
3. Added UPerNet using intermediate features. An initial S1 cosine-schedule trial was followed by a PANGAEA-inspired protocol and S2/joint variants.
4. Added precision/recall/F1, omission/commission, confusion counts, per-chip/regional analysis, AW/PW/FW, prediction panels and input-quality diagnostics.
5. Investigated modality differences, fine versus coarse predictions, ignored labels and missing acquisitions. The metrics establish performance differences; their causes remain partly unresolved.
6. Reorganized paths into dataset/model/experiment folders. The local `reports/folder_reorganization.json` records mappings and integrity verification. Data, checkpoints, metric values and notebook cell structure were preserved; active path references/root discovery were updated.
7. Added TerraMind v1 Large S1 FCN training/evaluation notebooks based on the existing CROMA S1 FCN notebooks. Verified local checkpoint compatibility, a real-input CPU forward/gradient check, and evaluation artifact generation. Full training and testing remain to be run; no TerraMind performance results are claimed.

Creating this report did not modify notebook code, recompute metrics, train models or run inference.

## Notebooks and protocols

Each experiment is under `experiments/sen1floods11/croma/`, with the identically named run root under `runs/sen1floods11/croma/`. Prefixes `01`, `02` and `00` mean training, evaluation and exploration. Historical run metadata takes precedence over today's notebook defaults when interpreting old results.

| Experiment / matching run folder | Purpose | Notebooks |
| --- | --- | --- |
| `fcn_s1` | S1 FCN; earlier head variants also retained | [01_croma_on_sen1floods11_training_og.ipynb](../experiments/sen1floods11/croma/fcn_s1/01_croma_on_sen1floods11_training_og.ipynb)<br>[02_croma_on_sen1floods11_test_og.ipynb](../experiments/sen1floods11/croma/fcn_s1/02_croma_on_sen1floods11_test_og.ipynb) |
| `fcn_s2` | S2 FCN | [01_croma_on_sen1floods11_training_s2.ipynb](../experiments/sen1floods11/croma/fcn_s2/01_croma_on_sen1floods11_training_s2.ipynb)<br>[02_croma_on_sen1floods11_test_s2.ipynb](../experiments/sen1floods11/croma/fcn_s2/02_croma_on_sen1floods11_test_s2.ipynb) |
| `linear_s1` | S1 linear head | [01C_croma_on_sen1floods11_training_probing.ipynb](../experiments/sen1floods11/croma/linear_s1/01C_croma_on_sen1floods11_training_probing.ipynb)<br>[02C_croma_on_sen1floods11_test_probing.ipynb](../experiments/sen1floods11/croma/linear_s1/02C_croma_on_sen1floods11_test_probing.ipynb) |
| `fcn_s1_own_preprocessing` | S1 FCN with training-statistics normalization | [01B_croma_on_sen1floods11_training_own_pre.ipynb](../experiments/sen1floods11/croma/fcn_s1_own_preprocessing/01B_croma_on_sen1floods11_training_own_pre.ipynb)<br>[02B_croma_on_sen1floods11_test_own_pre.ipynb](../experiments/sen1floods11/croma/fcn_s1_own_preprocessing/02B_croma_on_sen1floods11_test_own_pre.ipynb) |
| `upernet_s1` | S1 UPerNet | [01C_croma_on_sen1floods11_training_UperNet_S1.ipynb](../experiments/sen1floods11/croma/upernet_s1/01C_croma_on_sen1floods11_training_UperNet_S1.ipynb)<br>[02C_croma_on_sen1floods11_test_UperNet_S1.ipynb](../experiments/sen1floods11/croma/upernet_s1/02C_croma_on_sen1floods11_test_UperNet_S1.ipynb) |
| `upernet_s2` | S2 UPerNet | [01C_croma_on_sen1floods11_training_UperNet_S2.ipynb](../experiments/sen1floods11/croma/upernet_s2/01C_croma_on_sen1floods11_training_UperNet_S2.ipynb)<br>[02C_croma_on_sen1floods11_test_UperNet_S2.ipynb](../experiments/sen1floods11/croma/upernet_s2/02C_croma_on_sen1floods11_test_UperNet_S2.ipynb) |
| `upernet_s1s2` | S1+S2 UPerNet | [01C_croma_on_sen1floods11_training_UperNet_S1S2.ipynb](../experiments/sen1floods11/croma/upernet_s1s2/01C_croma_on_sen1floods11_training_UperNet_S1S2.ipynb)<br>[02C_croma_on_sen1floods11_test_UperNet_S1S2.ipynb](../experiments/sen1floods11/croma/upernet_s1s2/02C_croma_on_sen1floods11_test_UperNet_S1S2.ipynb) |
| `exploration` | Preliminary data/model inspection | [00_croma_on_sen1floods11_preliminary.ipynb](../experiments/sen1floods11/croma/exploration/00_croma_on_sen1floods11_preliminary.ipynb) |

### FCN and linear experiments

- **FCN S1:** `helper.py` replaces nonfinite inputs, clips SAR to `[-50, 1]`, scales to `[0, 1]`, then applies per-image/channel mean +/- two-standard-deviation normalization. The current notebook uses weighted cross entropy `[1, 8]`, 100 epochs, AdamW and cosine warm restarts. The frozen backbone supplies final SAR tokens.
- **Own preprocessing S1:** computes normalization statistics from training data, uses continuous floating-point normalization, and stores statistics with the checkpoint for evaluation. Data/preprocessing definitions are in its notebooks; the SAR segmentation wrapper is shared.
- **Linear S1:** a linear classifier on frozen spatial features, unweighted cross entropy and a cosine schedule.
- **FCN S2:** 12 optical bands excluding B10, fixed PANGAEA-style mean/std normalization, final optical features, weighted cross entropy and cosine warm restarts. Two completed training runs have validation intervals of 5 and 2 epochs; the current notebook uses 2.

The former names `original_S1` and `original_S2` do **not** establish exact reproduction of the Sen1Floods11 paper's preprocessing. S1 includes additional CROMA normalization; S2 uses optical mean/std normalization. Also, `fcn_s1` contains an older run explicitly named `DEEPLABHEAD`, which should not be silently classified as an FCN result.

### Selected UPerNet protocol

| Setting | Value in the three selected runs |
| --- | --- |
| Backbone | CROMA Large, frozen; decoder trained |
| Epochs / validation | 80 / every 2 epochs |
| Seed | 234 |
| Crop / batch | 120 x 120 / 8; incomplete final batch dropped |
| Precision | FP32 |
| Loss | Unweighted cross entropy; ignore index 255 |
| Optimizer | AdamW, learning rate 0.0001, weight decay 0.05 |
| Schedule | MultiStepLR; multiply LR by 0.1 at 60% and 90% of optimizer updates |
| Checkpoint selection | Best validation mIoU |
| Features / decoder | Four 1024-channel maps; UPerNet with 512 internal channels and two classes |
| Pyramid pooling | Scales 1, 2, 3, 6 |
| Sampling | Exclude fully ignored training images; reject fully ignored crops |
| Water oversampling | Disabled |
| Full-image inference | Tiled; grid recorded as `linspace_ceil_image_div_tile` |

A valid training crop needs a labelled pixel, **not a water pixel**. After at most 32 attempts, the fallback picks a crop containing a valid pixel. Land-only crops remain possible.

S1 normalization uses VV/VH means `[-10.184408, -16.895273]` and standard deviations `[4.255339, 5.290568]`, without FCN clipping/scaling. S2 uses the arrays in `run.json`, ordered B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12. Joint input applies the appropriate normalization separately to each sensor. Display stretching is not model preprocessing.

Earlier UPerNet runs used other validation intervals and, initially, a cosine schedule. They remain distinct records in the inventory.

## How CROMA and UPerNet connect

The local CROMA Large architecture has 1024-dimensional tokens, 8 x 8 patches, a 12-block radar encoder, a 24-block optical encoder and a 12-block joint encoder. A 120 x 120 crop yields a 15 x 15 spatial-token grid. The segmentation heads use these spatial representations, not the pooled contrastive embeddings.

| Experiment | Features supplied to UPerNet |
| --- | --- |
| S1 | Radar blocks 4, 6, 8, 12 |
| S2 | Optical blocks 4, 6, 8, 12; extraction stops after block 12 of 24 |
| S1+S2 | Joint blocks 4, 6, 8, 12, after the full radar and optical encoders |

Indices are stored as `[3, 5, 7, 11]` in zero-based code. Only the last selected representation receives final encoder normalization in these wrappers. Joint attention uses radar queries and optical context. The decoder receives four **joint** maps, not four optical maps plus an additional joint map, nor concatenated prediction masks.

UPerNet first turns equal-resolution transformer maps into a pyramid using scale factors 4, 2, 1 and 0.5. Pyramid pooling adds context to the deepest map. Lateral projections and top-down fusion combine the four levels; convolutions and a classifier produce logits, interpolated back to crop size. This implementation requires four inputs.

This gives more contextual/spatial information to the decoder than a final-grid classifier, but cannot restore absent input information. S2 versus joint UPerNet also changes the representation stream and feature depths: it is not a controlled comparison changing only the presence of SAR.

## TerraMind S1 FCN addition

Added on 2 November 2026. This adds TerraMind to the existing S1 comparison without retraining or changing the CROMA or authors' experiments.

| New notebook | Role |
| --- | --- |
| [01_terramind_on_sen1floods11_training.ipynb](../experiments/sen1floods11/terramind/fcn_s1/01_terramind_on_sen1floods11_training.ipynb) | Train an FCN head on frozen TerraMind v1 Large S1 features |
| [02_terramind_on_sen1floods11_test.ipynb](../experiments/sen1floods11/terramind/fcn_s1/02_terramind_on_sen1floods11_test.ipynb) | Load a compatible checkpoint and evaluate non-Bolivia and Bolivia separately |

The notebooks retain the CROMA S1 FCN templates' cell counts (30 training, 31 testing), IDs, ordering and general workflow. Saved template outputs were cleared. Existing headings were adapted to TerraMind; no extra Markdown cells or external Python helper files were added. Model-specific definitions are in the notebooks, which reuse the existing metrics and visualization modules. The README now links the notebooks and documents setup.

### Settings retained for comparison

- Same S1 VV/VH acquisitions, hand-labelled masks and official CSV splits: 252 train, 89 validation, 90 non-Bolivia test and 15 Bolivia test observations. Strict checks reject missing pairs, duplicate IDs and split overlap rather than silently dropping observations.
- Same 120 x 120 random training crops and horizontal/vertical flips, preserving split membership. All-ignore batches are skipped; there is no added water oversampling.
- Frozen pretrained encoder and the same FCN head: 1024 input channels, 256 hidden channels, GroupNorm with 16 groups and two output classes.
- 100 epochs, batch size 16, weighted cross entropy `[1, 8]`, AdamW with learning rate `5e-4` and weight decay `0.01`, and cosine warm restarts stepped after optimizer updates.
- Validation every epoch and checkpoint selection by validation water IoU, following the original S1 FCN notebook rather than the UPerNet protocol.
- Full-image evaluation using 120-pixel windows, stride 60 and mean overlapping logits. Labels are not resized.
- Same pooled segmentation metrics, separate Bolivia/non-Bolivia reporting, historical AW/PW/FW rules, RGB/reference/prediction/error panels and training plots.

### Necessary model-specific differences

TerraTorch supplies the TerraMind architecture; training and evaluation remain ordinary PyTorch loops. The notebooks pin `terratorch==1.2.5` and load `pretrained/terramind/TerraMind_v1_large.pt` locally. Every tensor required by the selected S1 encoder must be present with the expected shape; unused modalities and the generative decoder are excluded. No model-weight download is performed by the notebooks.

TerraMind uses its native S1GRD normalization: VV/VH means `[-12.599, -20.293]` and standard deviations `[5.195, 5.890]`. Raw data are already in dB; the CROMA clipping/per-image normalization is not applied. Nonfinite raw values are replaced by zero. These statistics follow the [official TerraMind Sen1Floods11 configuration](https://github.com/IBM/terramind/blob/main/configs/terramind_v1_base_sen1floods11.yaml).

TerraMind has 16-pixel patches. Each 120-pixel crop is padded on its right/bottom edges to 128 using replicated boundary values. The decoder logits are interpolated to the padded extent and then cropped back to 120. The model uses the final, normalized block-24 features, giving an 8 x 8 token grid versus CROMA's 15 x 15 grid. The checkpoint's positional embeddings are retained and interpolated by the architecture for the input size.

To limit memory use, only the frozen encoder is processed in chunks of four images; the FCN still receives the full training batch. These differences are recorded explicitly. The comparison evaluates the concrete pretrained-model pipelines; it does not isolate architecture from normalization, pretraining or feature resolution. Existing author/CROMA results remain historical observations under their recorded conditions.

### Outputs and provenance

Outputs are created under `runs/sen1floods11/terramind/fcn_s1/`, with separate training and evaluation runs. Training records seed 42, split membership and CSV hashes, package versions, pretrained-checkpoint SHA-256, preprocessing, optimizer/scheduler settings and the selected checkpoint. Independent crop and loader generators are reset before training.

Evaluation automatically selects the latest completed compatible training run, or accepts an explicit `CHECKPOINT_PATH`. It verifies the checkpoint schema, preprocessing, split manifest and pretrained-weight hash before loading the FCN state. It saves prediction rasters, split metrics, combined JSON/CSV summaries, per-chip/regional scores, AW/PW/FW CSVs and plots. It also saves a comparison table against the latest existing CROMA S1 FCN evaluation, retaining source paths and checking valid-pixel counts. Matching pixel counts alone do not establish identical training provenance; this table does not rerun CROMA or the authors' model.

### Verification and remaining execution

Verified notebook syntax and preserved cell structure; confirmed the four split counts and non-overlap. Using the actual TerraTorch 1.2.5 wheel's core model modules and local checkpoint, a real-input CPU smoke check produced finite 120 x 120 logits and gradients only for the FCN head. A separate synthetic-prediction check verified tiled coverage, ignored-label handling, raster output, metric JSON/CSVs, regional reporting and AW/PW/FW masks. These checks are not benchmark results.

TerraTorch is not installed in the existing notebook environment. For model verification, its core modules were loaded in isolation from a temporary wheel, bypassing unrelated package imports; a full installed-package notebook run has not been verified. Before running, execute `%pip install terratorch==1.2.5` in the intended notebook kernel and restart it. Then run the training notebook followed by testing. No full training, full-test inference or TerraMind performance measurement has been performed.

## Every module in src

`src/` contains six shared modules, not a separate training framework. Training loops, optimizer settings, modality-specific adaptations, inference orchestration and much AW/PW/FW analysis remain inside notebooks. Notebook-defined classes can override imported names: follow the active definition for that experiment.

### use_croma.py

[Source](../src/use_croma.py). Defines the pretrained architecture and weight loading, based on the [CROMA implementation](https://github.com/antofuller/CROMA).

| Symbol | Responsibility |
| --- | --- |
| `PretrainedCROMA` | Builds model size/modality, loads pretrained components, returns spatial and pooled encodings |
| `get_2dalibi` | Distance-based 2D attention bias |
| `ViT` | Image patches to tokens, then transformer processing |
| `Attention` | Self-attention within a stream |
| `CrossAttention` | Attention between streams, used for joint representations |
| `FFN` | Per-token feed-forward transformation |
| `BaseTransformer` | Residual self-attention/feed-forward blocks |
| `BaseTransformerCrossAttn` | Residual self-attention, cross-attention and feed-forward blocks |

This file does not select data splits, train a flood decoder or calculate evaluation metrics.

### helper.py

[Source](../src/helper.py). Original S1-style data and segmentation utilities.

| Symbol | Responsibility / behavior |
| --- | --- |
| `CROP_SIZE`, `IGNORE_INDEX` | Defaults 120 and 255 |
| `load_split_paths` | Reads CSV image/label pairs; prints and skips missing pairs, so resulting counts must be checked |
| `load_image_pair` | Reads SAR/labels, replaces nonfinite values, clips/scales SAR, maps label -1 to 255 |
| `normalize_croma` | Per-image/channel mean +/- 2 standard deviation normalization; clamps to [0,1] |
| `normalize_croma_reference` | Alternative batched normalization with optional 8-bit quantization; not the active default merely because it exists |
| `prepare_sample` | Tensor conversion, shape/class checks, per-image normalization |
| `Sen1Floods11Dataset` | Loads each pair and applies a supplied preprocessing function |
| `CROMAFloodSegmentation` | Frozen SAR tokens to spatial grid, supplied segmentation head, then upsampled logits |

The wrapper keeps CROMA in evaluation mode when training the head, and extracts features without gradients. S2 loading and wrappers are defined in the optical notebooks, not in this SAR wrapper.

### croma_upernet_pangaea.py

[Source](../src/croma_upernet_pangaea.py). Local UPerNet implementation and S1 utilities. The source credits PANGAEA and its GPL-3.0 license, and records revision `ca19b837a16797cd809225225b671473c7c93768`.

| Symbol | Responsibility / behavior |
| --- | --- |
| `PANGAEA_REVISION`, `SCHEMA`, `PREPROCESSING` | Source provenance, compatible run identifier and fixed SAR normalization |
| `load_split_paths` | Strict pair loading: missing files raise an error, unlike `helper.py` |
| `filter_labeled_training_pairs` | Excludes all-ignore masks and rejects unexpected labels |
| `random_labeled_crop` | Bounded crop sampling with a valid-pixel fallback; no water oversampling |
| `Sen1Floods11Dataset` | Loads raw SAR and labels, without FCN clipping/per-image normalization |
| `prepare_sample` | Validates aligned VV/VH and labels, converts ignore values, applies fixed mean/std |
| `Feature2Pyramid` | Converts four token maps into multiple spatial scales |
| `PPM` | Pyramid pooling for multi-scale context |
| `UPerNetHead` | Pooling, feature-pyramid fusion, convolutional refinement and classification |
| `CROMAUPerNet` | Frozen SAR wrapper extracting selected radar blocks |
| `latest_completed_checkpoint` | Selects the most recent compatible completed training run with an existing checkpoint |

S2/joint notebooks reuse `UPerNetHead` but define their own preprocessing, loaders, feature extraction and checkpoint discovery. The discovery function selects by compatible completion/recency, not highest test score. Evaluation metadata records what was actually loaded.

A separate `pangaea_reference` checkout is not needed to execute these local definitions. The adaptations are in this module and the notebooks.

### segmentation_metrics.py

[Source](../src/segmentation_metrics.py). Binary metrics and reporting.

| Symbol | Responsibility |
| --- | --- |
| `METRIC_KEYS` | Ordered names used in training/evaluation reporting |
| `confusion_counts` | TP/FP/FN/TN after ignoring invalid targets and checking shapes/classes |
| `binary_metrics` | IoUs, accuracy, precision/recall/F1, error rates, specificity, balanced accuracy, MCC and water fractions |
| `clean_json` | Converts NumPy values; nonfinite values become JSON null |
| `save_json` | Writes JSON without nonstandard NaN literals |
| `metric_table` | Split-level table and optional CSV/JSON summaries |
| `group_metrics` | Pools confusion counts within groups before calculating metrics |
| `save_per_chip_metrics` | Chip-level and pooled regional CSVs |
| `plot_metric_history` | Validation plots respecting epochs without validation |

Undefined ratios become NaN internally and null in JSON. Regional scores pool counts rather than average chip IoUs. Historical AW/PW/FW code uses a different empty-union convention, explained below.

### sar_input_quality.py

[Source](../src/sar_input_quality.py). Checks input availability separately from correctness.

- `audit_sar_pairs` examines raw SAR values/raster masks, counts usable pixels and overlap with valid labels, and flags images with no usable SAR.
- `report_input_quality` saves audit/exclusion tables and a supplementary usable-image score calculated from saved predictions.

A usable SAR pixel must be finite and raster-valid in every band. The supplementary score excludes entire images without usable SAR; partially missing images retain all labelled pixels. This is not per-pixel input-quality masking. It does not rerun inference or alter labels. S2/joint input audits live in their notebooks.

### sen1floods11_visualization.py

[Source](../src/sen1floods11_visualization.py). RGB, reference, prediction and optional error panels.

| Symbol | Responsibility |
| --- | --- |
| `PREDICTION_SUFFIX`, `IGNORE_VALUES` | Filename convention and ignored labels |
| `_chip_id` | Observation ID from prediction filename |
| `_same_grid` | Raster shape, CRS and transform agreement |
| `_load_case` | Loads rasters/display, checks alignment/classes and computes displayed chip water IoU |
| `plot_predictions` | Builds sampled or explicitly selected comparison grids |

Default prediction suffix: `_S1Hand_prediction.tif`. Other-modality notebooks adapt visualization configuration. RGB displays B4/B3/B2 divided by 3000 and clipped; this is a display choice, not model normalization or a documented reconstruction of the annotation GUI stretch. Optical display data can be reprojected onto the label grid; prediction/label mismatches raise an error.

Ignored reference pixels are grey in both reference/prediction panels, not a predicted third class. Water is cyan, and false positives/negatives have separate error colors. Empty-union water IoU is undefined. A black optical display alone does not establish missing radar input.

## Saved files and metric definitions

### Run artifacts

| File/directory | Meaning |
| --- | --- |
| `run.json` | Identity, modality, protocol, notebook and checkpoint references; newer runs record more detail |
| `*training_history.json` | Epoch losses/metrics, completion and best-checkpoint summary |
| `validation_metrics.csv` | Scores at validation epochs |
| `all_metrics_history.csv` | Broader training/validation history, where generated |
| Training-run `*.pt` | Trained state, sometimes including configuration/normalization |
| `non_bol/metrics.json`, `bol/metrics.json` | Split test results |
| `metrics_summary.json`, `.csv` | Combined split tables in newer evaluations |
| `*_prediction.tif` | Georeferenced predicted masks |
| `metrics_per_chip.csv`, `metrics_per_region.csv` | Chip metrics and pooled region metrics, where generated |
| `croma_aw_pw_fw_summary.csv` | Mean-chip and pooled category IoUs, with counts |
| `croma_aw_pw_fw_per_chip.csv` | Underlying chip/category values |
| `*input_quality*.csv`, `excluded_missing_*.csv` | Availability checks and exclusions |
| `usable_s2_summary.csv`, `usable_joint_summary.csv`, `usable_sar_images/` | Supplementary usable-input subset scores |
| `s1_s2_joint_comparison.csv`, `joint_minus_s2_percentage_points.csv`, `comparison_sources.json` | Modality comparisons with source references |
| Plot images / saved notebook snapshots | Visual inspection / historical execution context |

Files differ with experiment age. Best-validation scores are not test results; filtered usable-input scores are not full-test scores. Old notebook snapshots/output text may display former paths even after active references were updated.

`runs/sen1floods11/authors_baseline/` retains hand-labelled author-model v1.0/v1.1 evaluations and a comparison run. Their original source notebooks were already absent before reorganization. This report does not infer a reproducible baseline pairing from folder names alone.

### Metrics

Water is positive. Counts exclude ignored targets. Unless specified as per-chip, they are pooled over the named split.

| Metric | Formula / meaning |
| --- | --- |
| Water IoU | `TP / (TP + FP + FN)` |
| Non-water IoU | `TN / (TN + FP + FN)` |
| Binary mIoU | `(water IoU + non-water IoU) / 2` |
| Pixel accuracy | `(TP + TN) / valid pixels`; land dominance can make it high |
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| F1 / Dice | `2 TP / (2 TP + FP + FN)` |
| Omission | `1 - recall` |
| Commission | `1 - precision` |
| Specificity | `TN / (TN + FP)` |
| False-positive rate | `FP / (TN + FP)` |
| Balanced accuracy | `(recall + specificity) / 2` |
| MCC | `(TP*TN-FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))` |
| Ground-truth water fraction | `(TP + FN) / valid pixels` |
| Predicted water fraction | `(TP + FP) / valid pixels` |
| Mean per-chip water IoU | Equal-weight mean of included chips' water IoUs |

Pooled water IoU effectively weights chips by water-union size. It can exceed mean per-chip IoU when small-water/difficult chips perform poorly; it is not mathematically required to do so.

**AW/PW/FW:** all water, permanent water and flood water. The JRC layer separates water categories: water pixels belonging to the other category are ignored while relevant background remains for false positives. The resulting confusion matrices and chip sets are not disjoint partitions, so **AW is not the average of PW and FW**. Non-Bolivia PW includes 54 chips versus 90 for AW/FW; Bolivia PW includes 7 versus 15.

**Historical empty-union convention:** saved category code uses `(TP+1e-7)/(TP+FP+FN+1e-7)`, returning 1 for empty unions, including an all-ignore chip if retained. It skips category-specific chips when water originally existed but none of that category remains; originally dry chips may remain. This differs from the shared metric/visualization helpers' NaN convention. This report preserves the saved values. State these conventions when comparing per-chip results to literature.

## Selected UPerNet results

These are the latest completed training runs for each modality and their linked evaluations. The inventory identifies them explicitly. Percentage metrics are rounded to two decimals; MCC/loss retain their natural scales. No metrics were recomputed. Non-Bolivia and Bolivia remain separate.

### Best validation checkpoints

| Input | Completed epochs | Selected epoch | Validation mIoU (%) | Validation water IoU (%) |
| --- | --- | --- | --- | --- |
| S1 | 80 | 68 | 79.85 | 64.51 |
| S2 | 80 | 56 | 90.54 | 83.30 |
| S1+S2 | 80 | 68 | 89.28 | 81.09 |

### Non-Bolivia: 90 test observations

| Input | mIoU (%) | Water IoU (%) | Precision (%) | Recall (%) | F1 (%) | Accuracy (%) | Omission (%) | Commission (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 79.78 | 64.94 | 86.23 | 72.45 | 78.74 | 95.11 | 27.55 | 13.77 |
| S2 | 89.88 | 82.40 | 92.80 | 88.03 | 90.35 | 97.65 | 11.97 | 7.20 |
| S1+S2 | 88.46 | 79.96 | 91.55 | 86.33 | 88.86 | 97.29 | 13.67 | 8.45 |

### Bolivia: 15 test observations

| Input | mIoU (%) | Water IoU (%) | Precision (%) | Recall (%) | F1 (%) | Accuracy (%) | Omission (%) | Commission (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 72.91 | 54.51 | 86.77 | 59.44 | 70.56 | 92.13 | 40.56 | 13.23 |
| S2 | 84.90 | 74.57 | 95.39 | 77.36 | 85.43 | 95.82 | 22.64 | 4.61 |
| S1+S2 | 85.79 | 76.24 | 91.62 | 81.95 | 86.52 | 95.95 | 18.05 | 8.38 |

### Additional saved metrics

| Split | Input | Non-water IoU (%) | Specificity (%) | FPR (%) | Balanced accuracy (%) | MCC | GT water (%) | Predicted water (%) | Test loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| non_bol | S1 | 94.62 | 98.35 | 1.65 | 85.40 | 0.7637 | 12.51 | 10.51 | 0.1442 |
| non_bol | S2 | 97.36 | 99.02 | 0.98 | 93.53 | 0.8905 | 12.51 | 11.87 | 0.0683 |
| non_bol | S1+S2 | 96.97 | 98.86 | 1.14 | 92.60 | 0.8737 | 12.51 | 11.79 | 0.0817 |
| bol | S1 | 91.31 | 98.29 | 1.71 | 78.87 | 0.6777 | 15.86 | 10.87 | 0.1800 |
| bol | S2 | 95.23 | 99.29 | 0.71 | 88.33 | 0.8364 | 15.86 | 12.86 | 0.1025 |
| bol | S1+S2 | 95.34 | 98.59 | 1.41 | 90.27 | 0.8432 | 15.86 | 14.19 | 0.0989 |

### Confusion counts

| Split | Input | TP | FP | FN | TN | Valid pixels |
| --- | --- | --- | --- | --- | --- | --- |
| non_bol | S1 | 1859165 | 296929 | 706936 | 17654337 | 20517367 |
| non_bol | S2 | 2259052 | 175369 | 307049 | 17775897 | 20517367 |
| non_bol | S1+S2 | 2215368 | 204602 | 350733 | 17746664 | 20517367 |
| bol | S1 | 270393 | 41214 | 184471 | 2371737 | 2867815 |
| bol | S2 | 351881 | 17020 | 102983 | 2395931 | 2867815 |
| bol | S1+S2 | 372765 | 34073 | 82099 | 2378878 | 2867815 |

### AW / PW / FW: mean per-chip water IoU (%)

These are not binary mIoU or pooled water IoU. The historical inclusion/smoothing policy above applies.

| Split | Input | AW (%) | PW (%) | FW (%) | Chips AW / PW / FW |
| --- | --- | --- | --- | --- | --- |
| non_bol | S1 | 40.27 | 48.54 | 32.78 | 90 / 54 / 90 |
| bol | S1 | 36.71 | 47.02 | 29.40 | 15 / 7 / 15 |
| non_bol | S2 | 62.66 | 56.42 | 57.25 | 90 / 54 / 90 |
| bol | S2 | 53.84 | 61.14 | 49.43 | 15 / 7 / 15 |
| non_bol | S1+S2 | 57.89 | 56.30 | 51.73 | 90 / 54 / 90 |
| bol | S1+S2 | 54.72 | 58.85 | 49.08 | 15 / 7 / 15 |

## Complete run inventory

IDs are unique trailing identifiers of stored directories, shortened only for readability. Full paths follow. A dash indicates a missing summary field, not zero. Recorded epochs do not establish completion of the intended duration when that duration is absent.

### Training records

| Experiment | ID | Completed/planned epochs | Best epoch | Best val mIoU (%) | Best val water IoU (%) |
| --- | --- | --- | --- | --- | --- |
| fcn_s1 | 6c07bbb6 | 100 recorded | 71 | - | 63.13 |
| fcn_s1 | 1d930b22 | 72 recorded | 72 | - | 59.19 |
| fcn_s1_own_preprocessing | 48942d00 | 100 recorded | 98 | - | 61.77 |
| fcn_s2 | 800eae90 | 100/100 | 60 | - | 67.18 |
| fcn_s2 | d48b9c97 | 100/100 | 38 | - | 67.81 |
| linear_s1 | d6eabe85 | 100/100 | 65 | - | 60.44 |
| upernet_s1 | 37993207 | 80/80 | 75 | 79.10 | 63.06 |
| upernet_s1 | 0d253314 | 80/80 | 45 | 79.54 | 64.26 |
| upernet_s1 | 25793d6d | 29/80 | 24 | 78.63 | 62.56 |
| upernet_s1 | 95299899 | 80/80 | 68 | 79.85 | 64.51 |
| upernet_s1 | 7c0f850d | 14/100 | 8 | - | 59.73 |
| upernet_s1s2 | d8872ab6 | 80/80 | 68 | 89.28 | 81.09 |
| upernet_s2 | 2f9eec0f | 80/80 | 56 | 90.54 | 83.30 |

### Evaluation records

Scores are read from split `metrics.json` or combined `metrics_summary.json`. Repeated evaluations are executions, not independent model improvements. A training ID is shown only when the recorded checkpoint exists.

| Experiment | Test ID | Non-Bol mIoU (%) | Non-Bol water IoU (%) | Bol mIoU (%) | Bol water IoU (%) | Training ID / provenance |
| --- | --- | --- | --- | --- | --- | --- |
| fcn_s1 | a2dd4457 | 79.35 | 64.78 | 74.50 | 60.03 | Missing legacy checkpoint reference |
| fcn_s1 | 9a25367f | 79.35 | 64.78 | 74.50 | 60.03 | Missing legacy checkpoint reference |
| fcn_s1_own_preprocessing | 8956ee0b | 78.91 | 64.38 | 75.11 | 61.02 | Missing legacy checkpoint reference |
| fcn_s2 | 867de1f5 | 82.61 | 70.51 | 71.12 | 53.45 | 800eae90 |
| fcn_s2 | a6659442 | 81.76 | 68.62 | 71.07 | 52.90 | d48b9c97 |
| linear_s1 | c98e7df9 | 79.14 | 63.88 | 77.38 | 62.93 | d6eabe85 |
| upernet_s1 | eb1dbd8d | 78.91 | 63.33 | 66.30 | 42.91 | 37993207 |
| upernet_s1 | 31978d61 | 78.91 | 63.33 | 66.30 | 42.91 | 37993207 |
| upernet_s1 | 931e821b | 78.71 | 63.39 | 81.80 | 69.80 | 0d253314 |
| upernet_s1 | 75b6c9f3 | 79.78 | 64.94 | 72.91 | 54.51 | 95299899 |
| upernet_s1s2 | 735e86a8 | 88.46 | 79.96 | 85.79 | 76.24 | d8872ab6 |
| upernet_s2 | 3f3c3244 | 89.88 | 82.40 | 84.90 | 74.57 | 2f9eec0f |

Both FCN S1 evaluations have identical scores and unresolved legacy checkpoint paths; they must not automatically be paired with the newer FCN training. The own-preprocessing evaluation also has an unresolved legacy path; a matching-looking filename is not verified provenance. Two early UPerNet S1 evaluations repeat the same scores and linked model.

### Full run-directory provenance

Paths are relative to `runs/sen1floods11/croma/`. These are local artifacts excluded from Git. Selected modality-comparison evaluations are marked **selected**.

- `fcn_s1/test_croma_best_head_weighted_loss_with_scheduler_2026-10-26_18-55-54_701666_a2dd4457`
- `fcn_s1/test_croma_best_head_weighted_loss_with_scheduler_2026-10-27_14-15-58_731089_9a25367f`
- `fcn_s1/train_croma_best_head_weighted_loss_with_scheduler_2_2026-10-27_12-13-18_603141_6c07bbb6`
- `fcn_s1/train_croma_best_head_weighted_loss_with_scheduler_DEEPLABHEAD_2026-10-21_10-56-19_933029_1d930b22`
- `fcn_s1_own_preprocessing/test_croma_own_preprocessing_float_2026-10-20_16-05-26_695085_48942d00_2026-10-20_18-17-34_175070_8956ee0b`
- `fcn_s1_own_preprocessing/train_croma_own_preprocessing_float_2026-10-20_16-05-26_695085_48942d00`
- `fcn_s2/test_croma_large_s2_fcn_weighted_ce_2026-10-23_00-43-04_428425_867de1f5`
- `fcn_s2/test_croma_large_s2_fcn_weighted_ce_2026-10-23_12-23-49_249697_a6659442`
- `fcn_s2/train_croma_large_s2_fcn_weighted_ce_2026-10-22_23-32-07_277161_800eae90`
- `fcn_s2/train_croma_large_s2_fcn_weighted_ce_2026-10-23_08-59-24_626620_d48b9c97`
- `linear_s1/test_croma_large_linear_unweighted_ce_cosine_2026-10-28_12-36-58_119775_c98e7df9`
- `linear_s1/train_croma_large_linear_unweighted_ce_cosine_2026-10-21_12-17-56_597659_d6eabe85`
- `upernet_s1/test_croma_large_upernet_pangaea_sar_2026-10-22_00-24-54_786448_eb1dbd8d`
- `upernet_s1/test_croma_large_upernet_pangaea_sar_2026-10-22_12-55-26_948345_31978d61`
- `upernet_s1/test_croma_large_upernet_pangaea_sar_2026-10-22_13-44-01_827886_931e821b`
- `upernet_s1/test_croma_large_upernet_pangaea_sar_2026-10-22_16-53-42_953262_75b6c9f3` **selected**
- `upernet_s1/train_croma_large_upernet_pangaea_sar_2026-10-21_23-50-44_218671_37993207`
- `upernet_s1/train_croma_large_upernet_pangaea_sar_2026-10-22_13-12-32_280077_0d253314`
- `upernet_s1/train_croma_large_upernet_pangaea_sar_2026-10-22_14-14-06_626954_25793d6d`
- `upernet_s1/train_croma_large_upernet_pangaea_sar_2026-10-22_16-00-47_446450_95299899`
- `upernet_s1/train_croma_large_upernet_unweighted_ce_cosine_2026-10-21_23-20-21_205214_7c0f850d`
- `upernet_s1s2/test_croma_large_upernet_pangaea_s1s2_2026-10-26_14-29-18_580969_735e86a8` **selected**
- `upernet_s1s2/train_croma_large_upernet_pangaea_s1s2_2026-10-26_10-57-52_909543_d8872ab6`
- `upernet_s2/test_croma_large_upernet_pangaea_s2_2026-10-24_19-19-25_195280_3f3c3244` **selected**
- `upernet_s2/train_croma_large_upernet_pangaea_s2_2026-10-24_17-53-29_579252_2f9eec0f`

## Interpretation and literature

### Findings supported by the saved results

- **S2 UPerNet leads on non-Bolivia.** Its water IoU exceeds S1 UPerNet by about 17.47 percentage points, with substantially higher recall. This is not just a land-dominated accuracy effect.
- **Joint input is not uniformly worse.** It trails S2 by about 1.42 mIoU points on non-Bolivia but exceeds it by about 0.89 points on Bolivia. Both splits matter.
- **The S2 UPerNet configuration improves over the stored FCN configurations.** This is not a decoder-only ablation: feature selection, loss, schedule and other settings differ too.
- **Validation is not test performance.** Of the two FCN S2 runs, the higher best-validation water IoU accompanies the lower non-Bolivia test water IoU. Retrospective selection by test score would bias the comparison.
- **Performance is uneven across chips.** Mean per-chip AW scores are much lower than pooled water IoU. Large water unions performing well does not imply every observation is segmented well.

### Explanations still requiring evidence

Spectral information, SAR ambiguity, input quality, label uncertainty, acquisition timing and pretrained feature suitability may all contribute. Saved aggregate metrics do not establish their separate effects. Joint UPerNet also changes the representation stream: the frozen joint encoder cannot learn a new fusion strategy from these labels, while S2 UPerNet accesses optical intermediate features directly.

A grid artifact or black display warrants checking raw values, raster masks, alignment and tile boundaries. It is not sufficient evidence to discard an official test observation. Full-test and quality-filtered scores should remain separately labelled.

These results do not prove that acquisition-date offsets caused the performance gap, that reference labels are universally wrong for SAR, or that weak supervision improves this implementation. Each requires observation-level evidence or controlled experiments. The existing data are useful for investigating these questions; their presence does not establish a completed experiment.

### PANGAEA comparison

[PANGAEA Table A6](https://arxiv.org/html/2412.04204v2#A4.T6) reports CROMA with 100% training labels at **90.89% mIoU and 84.15% water IoU**. The table has one CROMA row, without separate S1/S2/joint columns. Do not relabel it as three modality-specific reference results without checking the released experiment configuration.

Numerically, local non-Bolivia S2 is 1.01 mIoU points and 1.75 water-IoU points below that row; joint is 2.43 and 4.19 points below. This is context, not a verified modality-matched reproduction gap. Local adaptations include sampling safeguards, single-process normalization and validation cadence.

Implementation provenance is recorded in `run.json` and [the UPerNet module](../src/croma_upernet_pangaea.py). Compare the [PANGAEA source at the recorded revision](https://github.com/VMarsocci/pangaea-bench/tree/ca19b837a16797cd809225225b671473c7c93768) for normalization, encoder selection, cropping, inference and checkpoint selection. Mean per-chip water IoU cannot be directly compared to binary mIoU.

### Next experiments

Keep these runs as documented baselines. To isolate causes, fix data membership, preprocessing, head, sampling, inference and checkpoint selection; vary one factor at a time and repeat over seeds. Useful questions include optical feature depth, alternative fusion, and performance by event, water extent and input coverage. Choose settings on training/validation data and reserve test data for evaluation.

The TerraMind S1 FCN notebooks are now ready for the next experiment, with model-specific normalization and patch handling documented above. Full training/evaluation remains pending. Weak-label/data-scarcity work remains a possible next study, not an existing outcome.

## Reproducibility and GitHub

To trace a result, start at evaluation `run.json`, follow its checkpoint to the training run, inspect that configuration/history, then the notebook and any historical snapshot. Full run identities are listed above.

| Report information | Local source |
| --- | --- |
| Validation checkpoints | Training `*training_history.json` |
| Selected test scores and confusion counts | Evaluation `metrics_summary.json` |
| AW/PW/FW | Evaluation `croma_aw_pw_fw_summary.csv` |
| Historical test scores | `metrics_summary.json`, or `non_bol/metrics.json` and `bol/metrics.json` |
| Settings / provenance | `run.json` and active source/notebook definitions |

Some older references were broken before reorganization. The local migration audit retains them; uncertain train/test links remain flagged rather than inferred.

This report, README, notebooks and source are intended for Git. Data, model binaries, generated runs, prediction rasters, large archives and other local reports remain excluded; `.gitignore` makes an exception for this report only. A GitHub clone documents the results but does not contain the local data/checkpoints needed to rerun them. The two TerraMind experiment notebooks are intended to be tracked; their generated runs remain excluded.

This report update documents the TerraMind notebook addition and its limited verification. It does not change the existing CROMA/authors notebooks, source modules, saved results or metric policies. All numerical performance tables above still refer to the previously saved CROMA runs.
