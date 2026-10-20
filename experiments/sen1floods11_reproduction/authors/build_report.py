"""Compare saved predictions at native resolution, with separate metric conventions."""
import csv, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, "src")
from sen1floods11_visualization import _load_case
suite = Path(sys.argv[1])
manifest = json.loads((suite / "manifest.json").read_text())
assert len(manifest["runs"]) == 8, "Wait for all eight evaluations."
candidates = sorted(Path("runs/croma_sen1floods11/own_preprocessing").glob("test_*/run.json"), key=lambda p: json.loads(p.read_text())["started_at"])
croma = candidates[-1].parent if candidates else None
manifest["croma_comparison_run"] = croma.as_posix() if croma else None
assets = suite / "assets"
assets.mkdir(exist_ok=True)
rows, native, viewer = [], [], []
checks = {"prediction_grids_verified": 0, "legacy_label_affected_images": [], "prediction_resampling": False}
colors = np.array([[35,43,53],[0,190,225],[190,190,190]], dtype=np.uint8)
def png(name, array):
    Image.fromarray(array).save(assets / name)
    return "assets/" + name
for version in ("v1.0", "v1.1"):
    data = Path("../Sen1Floods11_experiment/datasets") if version == "v1.0" else Path("dataset/sen1floods11_hand_labeled_only/HandLabeled")
    splits = data if version == "v1.0" else Path("dataset/sen1floods11_hand_labeled_only/flood_handlabeled")
    runs = [r for r in manifest["runs"] if r["test_dataset"] == version]
    for r in runs:
        files = list(Path(r["directory"]).glob("*_" + r["model"] + ".json"))
        assert len(files) == 1, files
        scores = json.loads(files[0].read_text())
        for key, value in scores["results"].items():
            native.append({"model":r["model"], "test_dataset":version, "category_split":key, **value})
    for split, csvname in (("non_bol", "flood_test_data.csv"), ("bol", "flood_bolivia_data.csv")):
        for xname, yname in csv.reader((splits / csvname).open()):
            if version == "v1.0":
                xp, yp = data / xname, data / yname
                pp = Path(str(yp).replace("QC_v2", "Perm").replace("QC", "Perm"))
            else:
                xp, yp = data / "S1Hand" / Path(xname).name, data / "LabelHand" / Path(yname).name
                pp = data / "JRCWaterHand" / Path(yname).name.replace("LabelHand", "JRCWaterHand")
            chip = xp.stem.removesuffix("_S1Hand").removesuffix("_S1")
            with rasterio.open(yp) as src:
                y = src.read(1).astype(np.int16)
                grid = (src.shape, src.crs, src.transform)
            with rasterio.open(pp) as src:
                assert src.shape == grid[0] and src.crs == grid[1] and src.transform.almost_equals(grid[2])
                perm = src.read(1)
            assert np.isin(y, [-1, 0, 1, 255]).all()
            valid = (y == 0) | (y == 1)
            if not (y == 1).any() and (~valid).any():
                checks["legacy_label_affected_images"].append({"test_dataset":version, "split":split, "chip":chip, "ignored_pixels":int((~valid).sum())})
            with rasterio.open(xp) as src:
                sar = src.read(); vv = sar[0]; bad = (~np.isfinite(sar)).any(axis=0)
            grey = (np.clip((np.nan_to_num(vv, nan=-30, posinf=-30, neginf=-30) + 30) / 30, 0, 1) * 255).astype(np.uint8)
            prefix = version + "_" + split + "_" + chip
            item = {"version":version, "split":split, "chip":chip, "panels":[]}
            item["panels"].append({"name":"VV (display -30 to 0)", "mask":png(prefix + "_vv.png", grey)})
            item["panels"].append({"name":"Ground truth", "mask":png(prefix + "_label.png", colors[np.where(valid, y, 2).astype(int)])})
            preds = [(r["model"], Path(r["directory"]) / "predictions" / split / f"{chip}_S1Hand_prediction.tif") for r in runs]
            if version == "v1.1" and croma:
                preds.append(("CROMA own float", croma / split / f"{chip}_S1Hand_prediction.tif"))
                _, rgb, _, _, _ = _load_case(preds[0][1], data, s2_scale=3000.)
                item["panels"].insert(0, {"name":"S2 RGB (context only)", "mask":png(prefix + "_rgb.png", (rgb * 255).astype(np.uint8))})
            for model, predpath in preds:
                with rasterio.open(predpath) as src:
                    assert src.shape == grid[0] and src.crs == grid[1] and src.transform.almost_equals(grid[2], precision=1e-10), predpath
                    pred = src.read(1)
                assert pred.shape == (512, 512) and np.isin(pred, [0, 1]).all()
                checks["prediction_grids_verified"] += 1
                fp = valid & (y == 0) & (pred == 1)
                fn = valid & (y == 1) & (pred == 0)
                tp = valid & (y == 1) & (pred == 1)
                err = np.full((*y.shape, 3), 35, dtype=np.uint8)
                err[~valid] = [190,190,190]; err[tp] = [0,190,225]
                err[fp] = [255,145,0]; err[fn] = [225,40,110]
                slug = model.replace(" ", "_")
                item["panels"].append({"name":model, "mask":png(prefix + "_" + slug + ".png", colors[np.where(valid, pred, 2).astype(int)]), "errors":png(prefix + "_" + slug + "_errors.png", err)})
                for cat in ("AW", "PW", "FW"):
                    target = y.copy()
                    if cat == "PW": target[(y == 1) & (perm == 0)] = 255
                    if cat == "FW": target[(y == 1) & (perm == 1)] = 255
                    if cat != "AW" and (y == 1).any() and not (target == 1).any(): continue
                    v = (target == 0) | (target == 1)
                    t = int((v & (target == 1) & (pred == 1)).sum())
                    f = int((v & (target == 0) & (pred == 1)).sum())
                    miss = int((v & (target == 1) & (pred == 0)).sum())
                    tn = int((v & (target == 0) & (pred == 0)).sum())
                    rows.append({"model":model, "test_dataset":version, "split":split, "category":cat, "chip":chip, "tp":t, "fp":f, "fn":miss, "tn":tn, "chip_iou":(t+1e-7)/(t+f+miss+1e-7), "fp_invalid_sar":int((fp & bad).sum()) if cat == "AW" else None})
            viewer.append(item)
frame = pd.DataFrame(rows)
summary = frame.groupby(["model","test_dataset","split","category"], sort=False).agg(chips=("chip","size"), mean_chip_iou=("chip_iou","mean"), tp=("tp","sum"), fp=("fp","sum"), fn=("fn","sum"), tn=("tn","sum")).reset_index()
summary["pooled_iou"] = summary.tp / (summary.tp + summary.fp + summary.fn)
summary["precision"] = summary.tp / (summary.tp + summary.fp)
summary["recall"] = summary.tp / (summary.tp + summary.fn)
frame.to_csv(suite / "integer_label_per_chip.csv", index=False)
summary.to_csv(suite / "integer_label_summary.csv", index=False)
pd.DataFrame(native).to_csv(suite / "original_notebook_metrics.csv", index=False)
(suite / "validation.json").write_text(json.dumps(checks, indent=2))
(suite / "manifest.json").write_text(json.dumps(manifest, indent=2))
page = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
page = page.replace("__TABLE__", summary[["model","test_dataset","split","category","chips","mean_chip_iou","pooled_iou","precision","recall"]].to_html(index=False, float_format=lambda x:f"{x:.4f}"))
page = page.replace("__DATA__", json.dumps(viewer))
(suite / "index.html").write_text(page, encoding="utf-8")
print("REPORT", suite / "index.html", flush=True)
print(summary.loc[summary.category == "AW", ["model","test_dataset","split","mean_chip_iou","pooled_iou"]].to_string(index=False), flush=True)
print("CHECKS", checks, flush=True)
