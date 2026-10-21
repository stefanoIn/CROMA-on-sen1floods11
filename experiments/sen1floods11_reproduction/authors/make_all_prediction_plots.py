"""Export all scenes with per-method water IoU and separate error figures."""
from pathlib import Path
from datetime import datetime
import csv, json, sys, os, zipfile, html
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from sen1floods11_visualization import _load_case

suite = Path("runs/sen1floods11_reproduction/comparison_2026-10-20_18-31-06_611597_d03e130b")
manifest = json.loads((suite / "manifest.json").read_text())
auth = Path(next(r["directory"] for r in manifest["runs"] if r["model"] == "hand_labeled" and r["test_dataset"] == "v1.1"))
croma = Path(manifest["croma_comparison_run"])
mine = Path("fcnn_test/notebook_results/mine_train_v1_1_test_v1_1")
hand = Path("dataset/sen1floods11_hand_labeled_only/HandLabeled")
out = Path("reports") / ("all_chips_handlabelled_predictions_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))
out.mkdir(parents=True)
palette = np.array([[35,43,53], [0,190,225], [190,190,190]], dtype=np.uint8)
plt.style.use("default")
plt.rcParams["pdf.fonttype"] = 42

def read_mask(path, grid=None):
    with rasterio.open(path) as src:
        current = (src.shape, src.crs, src.transform)
        if grid is not None:
            assert current[:2] == grid[:2] and current[2].almost_equals(grid[2], precision=1e-10), path
        arr = src.read(1).astype(np.int16)
    assert np.isin(arr, [0, 1, -1, 255]).all(), path
    return arr, current

def color_mask(arr):
    # Preserve saved predictions, including outside the labelled GT area.
    # No error overlay and no masking of predictions by ground truth.
    return palette[np.where(np.isin(arr, [0,1]), arr, 2)]

error_palette = np.array([[35,43,53], [0,190,225], [255,145,0], [225,40,110], [190,190,190]], dtype=np.uint8)
method_names = ["v0v0 (authors FCN)", "v1.1 (my FCN)", "CROMA + FCN", "Otsu"]

def evaluate_prediction(truth, prediction):
    valid = np.isin(truth, [0,1]) & np.isin(prediction, [0,1])
    tp = valid & (truth == 1) & (prediction == 1)
    fp = valid & (truth == 0) & (prediction == 1)
    fn = valid & (truth == 1) & (prediction == 0)
    counts = {"tp":int(tp.sum()), "fp":int(fp.sum()), "fn":int(fn.sum()), "valid_pixels":int(valid.sum())}
    union = counts["tp"] + counts["fp"] + counts["fn"]
    counts["iou"] = None if not valid.any() else counts["tp"] / union if union else 1.0
    classes = np.full(truth.shape, 4, dtype=np.uint8)
    classes[valid] = 0
    classes[tp] = 1
    classes[fp] = 2
    classes[fn] = 3
    return counts, error_palette[classes]

def render_scene(chip, split, split_title, rgb, truth, predictions, stats, errors, error_mode=False):
    fig, axes = plt.subplots(2, 3, figsize=(16,9))
    fig.suptitle(f"{chip} - {split_title} - Sen1Floods11 v1.1" + (" - Errors" if error_mode else ""), fontsize=16)
    panels = [rgb, color_mask(truth)] + (errors if error_mode else [color_mask(p) for p in predictions])
    titles = ["Sentinel-2 (RGB)", "Ground truth"]
    for name, score in zip(method_names, stats):
        value = "N/A" if score["iou"] is None else f'{score["iou"]:.3f}'
        titles.append(f'{"Errors" if error_mode else "Prediction"} {name} | IoU {value}')
    for ax, arr, title in zip(axes.flat, panels, titles):
        ax.imshow(arr, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.subplots_adjust(left=.04, right=.96, top=.91, bottom=.10, wspace=.07, hspace=.19)
    colors = error_palette if error_mode else palette
    labels = ["Correct non-water (TN)", "Correct water (TP)", "False water (FP)", "Missed water (FN)", "Ignored / NoData"] if error_mode else ["Non-water", "Water", "Ignored / NoData in displayed layer"]
    fig.legend(handles=[Patch(facecolor=c/255, label=label) for c,label in zip(colors,labels)],
               loc="lower center", bbox_to_anchor=(.5,.035), ncol=len(labels), frameon=False, fontsize=11)
    fig.text(.5, .018, "Water IoU = TP / (TP + FP + FN); ignored GT and missing predictions excluded per method. N/A = no valid pixels.", ha="center", fontsize=10)
    stem = Path(split) / (chip + ("_error" if error_mode else ""))
    fig.savefig(out / stem.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(out / stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return stem.as_posix()

records = []
for split, filename, split_title, expected in [
    ("non_bol", "flood_test_data.csv", "Non-Bolivia", 90),
    ("bol", "flood_bolivia_data.csv", "Bolivia", 15),
]:
    (out / split).mkdir()
    with (hand.parent / "flood_handlabeled" / filename).open() as f:
        cases = list(csv.reader(f))
    assert len(cases) == expected
    for x, y in cases:
        chip = Path(x).stem.removesuffix("_S1Hand")
        truth, grid = read_mask(hand / "LabelHand" / Path(y).name)
        paths = [auth / "predictions" / split / f"{chip}_S1Hand_prediction.tif",
                 mine / "predictions" / split / f"{chip}_S1Hand_prediction.tif",
                 croma / split / f"{chip}_S1Hand_prediction.tif",
                 hand / "S1OtsuLabelHand" / f"{chip}_S1OtsuLabelHand.tif"]
        predictions = [read_mask(p, grid)[0] for p in paths]
        case_id, rgb, _, _, _ = _load_case(paths[0], hand, s2_scale=3000)
        assert case_id == chip and rgb.shape[:2] == truth.shape
        scored = [evaluate_prediction(truth, pred) for pred in predictions]
        stats = [item[0] for item in scored]
        errors = [item[1] for item in scored]
        stem = render_scene(chip, split, split_title, rgb, truth, predictions, stats, errors)
        error_stem = render_scene(chip, split, split_title, rgb, truth, predictions, stats, errors, error_mode=True)
        records.append({"chip":chip, "split":split, "figure":stem, "error_figure":error_stem,
                        "scores":dict(zip(method_names,stats)), "prediction_sources":[p.as_posix() for p in paths]})
        if len(records) % 10 == 0:
            print(f"Exported {len(records)}/105 scenes", flush=True)
assert len(records) == 105 and len({r["chip"] for r in records}) == 105
for split, count in [("non_bol",90),("bol",15)]:
    for ext in ["png","pdf"]:
        assert len(list((out/split).glob("*."+ext))) == 2 * count
provenance = {"authors_run":auth.as_posix(),"personal_fcn_run":mine.as_posix(),"croma_run":croma.as_posix(),
              "test_dataset":"v1.1", "rgb_bands":[4,3,2], "rgb_scale":3000,
              "prediction_display":"Raw saved class masks; no ground-truth masking; no errors; no scene selection",
              "iou_definition":"Per-image water TP/(TP+FP+FN), excluding ignored GT and missing predictions independently per method. Empty union with valid pixels = 1; no valid pixels = null (N/A).",
              "error_display":"TN dark; TP cyan; FP orange; FN pink; ignored GT or missing prediction grey.",
              "authors_note":"v0v0 is the notebook name; authors checkpoint trained on dataset v1.0", "figures":records}
(out/"provenance.json").write_text(json.dumps(provenance,indent=2))
page = ['<!doctype html><meta charset="utf-8"><title>All 105 prediction comparisons</title><style>body{font:18px Arial;background:#eef1f5;margin:30px}section{background:white;margin:24px 0;padding:18px}img{width:100%;max-width:1600px}a{margin-right:24px}</style><h1>All 105 scenes: hand-labelled prediction comparison</h1><p>Sentinel-2, ground truth, v0v0, v1.1, CROMA, Otsu. Per-method water IoU in titles; separate prediction and error figures for every scene.</p>']
for r in records:
    s=html.escape(r["figure"])
    e=html.escape(r["error_figure"])
    page.append(f'<section><h2>{html.escape(r["chip"])} ({r["split"]})</h2><a href="{s}.png">PNG</a><a href="{s}.pdf">PDF</a><br><img src="{s}.png" loading="lazy"><br><a href="{e}.png">Errors PNG</a><a href="{e}.pdf">Errors PDF</a><img src="{e}.png" loading="lazy"></section>')
(out/"index.html").write_text("\n".join(page),encoding="utf-8")
(out/"README.md").write_text("105 scenes (90 non-Bolivia, 15 Bolivia), each with four files: CHIP.png, CHIP.pdf, CHIP_error.png, CHIP_error.pdf. All use the same 2x3 Matplotlib layout: Sentinel-2 RGB, ground truth, authors hand-labelled FCN v0v0, personal hand-labelled FCN v1.1 (epoch051), CROMA + FCN, Otsu. PNG size 3200x1800. Predictions are shown as saved. Separate error maps use dark TN, cyan TP, orange FP, pink FN, grey ignored GT or missing prediction. Each model title includes per-image water IoU = TP/(TP+FP+FN), evaluated only on valid GT and valid predictions for that method. Otsu can have fewer valid pixels; counts are in provenance.json. Empty union with valid pixels scores 1; no valid pixels displays N/A. All scenes included, without cropping or selection. See provenance.json for run sources.\n",encoding="utf-8")
with zipfile.ZipFile(out/"all_105_prediction_figures.zip","w",zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.suffix != ".zip":
            archive.write(p,p.relative_to(out))
print("OUTPUT",out,flush=True)
