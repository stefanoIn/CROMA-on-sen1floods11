# Authors' Sen1Floods11 FCN evaluation

`Test_Modelsv0v0.ipynb` is the runnable local adaptation of the authors' evaluation workflow. `Test_Models_upstream.ipynb` is an unchanged copy from cloudtostreet/Sen1Floods11, `old_training/Test_Models.ipynb`. The runnable notebook builds on the restored Test_Modelsv0v0 local adaptation; its image transformation and water-category functions were checked against upstream ASTs.

All four distributed checkpoints were imported from `../Sen1Floods11_experiment/datasets/cnn_checkpoints/` into `pretrained/sen1floods11_authors/`, with copy hashes verified. `checkpoint_sources.json` records source paths, SHA256 hashes, and official storage URLs. These are the distributed v1.0-trained models, not the user's retrained checkpoints. Their training supervision differs: hand-labeled, S1 weak labels, S2 weak labels, and permanent water.

## Run

Use the existing THESIS D RTX3070 environment. From the repository root:

```powershell
python experiments/sen1floods11_reproduction/authors/run_authors.py
python experiments/sen1floods11_reproduction/authors/build_report.py runs/sen1floods11_reproduction/comparison_<timestamp>_<id>
```

The batch runner executes the notebook eight times: four models on original v1.0 test data, then the same models on v1.1. Each run saves its checkpoint provenance, native notebook metrics, and 105 georeferenced 512x512 predictions. Executed notebooks and the manifest live in the timestamped comparison folder. Single runs can be launched from the notebook by selecting MODEL_ID and DATASET_VERSION; optional FCNN_MODEL and FCNN_DATASET environment variables override these defaults.

Data references are relative: v1.0 uses `../Sen1Floods11_experiment/datasets`, and v1.1 uses this repository's dataset directory. Update the relative v1.0 path if that dataset moves. No external source files are edited. Existing Colab installation/download commands are replaced by local paths. Predictions reconstruct the original four 256x256 quadrants; the authors' model, normalization, and GroupNorm architecture are retained.

## Metric caveats

The historical PIL/ToTensor label conversion is deliberately preserved in the notebook. When an entire image contains no water, ignored label 255 can become label 1. The upstream loader also contained redundant unsigned arithmetic whose result was overwritten immediately; the local adaptation omits that unused operation for modern NumPy compatibility. The legacy commission formula is FP/(TN+FP), i.e. false positive rate, not FP/(TP+FP).

`original_notebook_metrics.csv` contains the native protocol's scores. `integer_label_summary.csv` and `integer_label_per_chip.csv` rescore saved predictions against raw integer labels, excluding -1/255, with the same AW/PW/FW category rules as the current CROMA tables. Empty-union chip IoU follows the existing epsilon convention (1); chips excluded by the existing category rules remain excluded. These are separate scores, not a silent correction of the authors' notebook.

The report shows every chip at native resolution, identical valid-pixel masking across models, and optional error maps and 2x zoom. CROMA is compared only on v1.1; its selected saved run is explicit in manifest.json. The current report builder selects the most recent saved own-preprocessing CROMA test run when building a new report. Original .tif masks are unmasked binary predictions; grey regions in the viewer only indicate ignored ground-truth pixels.

## Completed comparison

The initial completed suite is `runs/sen1floods11_reproduction/comparison_2026-10-20_18-31-06_611597_d03e130b/index.html`. It includes 840 newly generated authors' predictions and the existing 105 CROMA predictions. All 945 prediction rasters passed shape/CRS/transform and binary-value checks; CROMA integer-label metrics agree with its existing six category scores. All eight executed notebooks completed without errors.

The authors' FCN-ResNet50 outputs 32x32 logits for 256x256 input, while the CROMA head outputs 15x15 for 120x120: both have effective output stride 8 before bilinear upsampling. This can limit small-object and boundary detail, but the experiment does not isolate that cause: the backbone, training supervision, input context, and frozen-versus-trained features also differ. Architecture alone cannot explain the false-positive regions seen in the masks.
