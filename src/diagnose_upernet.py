import sys,json
from pathlib import Path
import numpy as np,pandas as pd,rasterio
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from segmentation_metrics import *
DATA=ROOT/"dataset/sen1floods11_hand_labeled_only/HandLabeled"
OUT=ROOT/"reports/upernet_diagnostics"
OUT.mkdir(parents=True,exist_ok=True)
RUN=ROOT/"runs/croma_sen1floods11/UperNet_head/test_croma_large_upernet_pangaea_sar_2026-10-22_00-24-54_786448_eb1dbd8d"
OLD=ROOT/"runs/croma_sen1floods11/own_preprocessing/test_croma_own_preprocessing_float_2026-10-20_16-05-26_695085_48942d00_2026-10-20_18-17-34_175070_8956ee0b"
rows=[]; detail=[]; sizes=[]; totals={}
for model,run in [("upernet",RUN),("previous_own_preprocessing",OLD)]:
 for split in ["non_bol","bol"]:
  current=[]
  for p in sorted((run/split).glob("*_prediction.tif")):
   chip=p.name.replace("_S1Hand_prediction.tif","")
   with rasterio.open(p) as src: pred=src.read(1);grid=(src.shape,src.crs,src.transform)
   with rasterio.open(DATA/"LabelHand"/(chip+"_LabelHand.tif")) as src:
    target=src.read(1)
    assert grid[:2]==(src.shape,src.crs) and np.allclose(tuple(grid[2]),tuple(src.transform),rtol=0,atol=1e-10),chip
   valid=np.isin(target,[0,1]);water=target==1
   row=dict(model=model,split=split,chip_id=chip,region=chip.rsplit("_",1)[0],**binary_metrics(*confusion_counts(pred,target)))
   current.append(row);rows.append(row)
   distance=ndi.distance_transform_edt(~(valid&~water))
   near_ignore=ndi.binary_dilation(~valid,iterations=3)
   for name,mask in [("water_within_3px_of_land",water&(distance<=3)&~near_ignore),("water_farther_than_3px_from_land",water&(distance>3)&~near_ignore)]:
    detail.append(dict(model=model,split=split,zone=name,water_pixels=int(mask.sum()),missed_pixels=int((mask&(pred==0)).sum())))
   labels,n=ndi.label(water,structure=np.ones((3,3)))
   areas=np.bincount(labels.ravel());areas[0]=0
   for name,low,high in [("1-64_pixels",1,64),("65-1024_pixels",65,1024),("1025+_pixels",1025,10**9)]:
    mask=water&(areas[labels]>=low)&(areas[labels]<=high)
    sizes.append(dict(model=model,split=split,size=name,water_pixels=int(mask.sum()),missed_pixels=int((mask&(pred==0)).sum())))
   if model=="upernet":
    with rasterio.open(DATA/"S1Hand"/(chip+"_S1Hand.tif")) as src:sar=src.read()
    row["nonfinite_sar_valid_pixels"]=int((~np.isfinite(sar).all(axis=0)&valid).sum())
  aggregate=binary_metrics(*[sum(r[k] for r in current) for k in ["TP","FP","FN","TN"]])
  saved=json.loads((run/split/"metrics.json").read_text())
  for k in ["TP","FP","FN","TN"]:assert aggregate[k]==saved[k],(model,split,k)
  totals[model+"_"+split]=aggregate
  if model=="upernet":save_per_chip_metrics(current,run/split)
metric_table({s:totals["upernet_"+s] for s in ["non_bol","bol"]},RUN)
metric_table(totals,OUT)
frame=pd.DataFrame(rows);frame.to_csv(OUT/"per_chip.csv",index=False)
regions=[]
for (model,split),group in frame.groupby(["model","split"]):
 r=group_metrics(group.to_dict("records"),"region");r.insert(0,"split",split);r.insert(0,"model",model);regions.append(r)
pd.concat(regions).to_csv(OUT/"per_region.csv",index=False)
for name,records,key in [("shoreline",detail,"zone"),("water_body_size",sizes,"size")]:
 df=pd.DataFrame(records).groupby(["model","split",key])[["water_pixels","missed_pixels"]].sum()
 df["recall"]=1-df.missed_pixels/df.water_pixels
 df.to_csv(OUT/(name+".csv"))
print(pd.DataFrame(totals).loc[["water_iou","miou","precision","recall","accuracy"]].to_string())
print(pd.concat(regions).query('model=="upernet"')[["region","chips","water_iou","recall","FN"]].to_string(index=False))
print(pd.read_csv(OUT/"shoreline.csv").to_string(index=False))
print("Nonfinite valid pixels",frame.query('model=="upernet"').nonfinite_sar_valid_pixels.sum())
examples=[]
for split in ["non_bol","bol"]:
 examples.extend(frame.query('model=="upernet" and split==@split').sort_values("FN",ascending=False).head(2).chip_id.tolist())
fig,axes=plt.subplots(len(examples),6,figsize=(18,3.6*len(examples)))
for axrow,chip in zip(axes,examples):
 split="bol" if chip.startswith("Bolivia") else "non_bol"
 with rasterio.open(DATA/"S1Hand"/(chip+"_S1Hand.tif")) as src:sar=src.read()
 with rasterio.open(DATA/"LabelHand"/(chip+"_LabelHand.tif")) as src:gt=src.read(1)
 with rasterio.open(RUN/split/(chip+"_S1Hand_prediction.tif")) as src:pred=src.read(1)
 with rasterio.open(OLD/split/(chip+"_S1Hand_prediction.tif")) as src:old=src.read(1)
 valid=np.isin(gt,[0,1]);error=np.zeros((*gt.shape,3),dtype=float)+.12;error[~valid]=.7
 error[valid&(gt==1)&(pred==1)]=[.1,.7,.4]
 error[valid&(gt==1)&(pred==0)]=[.9,.15,.15]
 error[valid&(gt==0)&(pred==1)]=[1,.65,.1]
 for ax,im,title in zip(axrow,[sar[0],sar[1],np.where(valid,gt,np.nan),np.where(valid,old,np.nan),np.where(valid,pred,np.nan),error],["SAR VV","SAR VH","Label","Previous model","UPerNet","TP green / FN red / FP orange"]):
  if title.startswith("SAR"):ax.imshow(im,cmap="gray",vmin=-25 if title=="SAR VV" else -32,vmax=0 if title=="SAR VV" else -7)
  elif title.startswith("TP"):ax.imshow(im)
  else:ax.imshow(im,cmap="Blues",vmin=0,vmax=1)
  ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
 axrow[0].set_ylabel(chip,fontsize=10)
fig.tight_layout();fig.savefig(OUT/"error_examples.png",dpi=170);plt.close(fig)
save_json(OUT/"provenance.json",dict(test_run=str(RUN),previous_run=str(OLD),method="Pooled valid-pixel confusion counts; labels -1/255 ignored; raster grids verified to 1e-10 coordinate tolerance; saved confusion counts verified. Boundary analysis excludes water within 3 pixels of ignored areas; 8-connected water components in raster pixels. Comparisons between runs are not controlled head ablations."))
# Additional diagnostic strata use raw SAR values, not model-normalized values.
signal_rows=[]
for split in ["non_bol","bol"]:
 for p in (RUN/split).glob("*_prediction.tif"):
  chip=p.name.replace("_S1Hand_prediction.tif","")
  with rasterio.open(p) as s:pred=s.read(1)
  with rasterio.open(DATA/"LabelHand"/(chip+"_LabelHand.tif")) as s:gt=s.read(1)
  with rasterio.open(DATA/"S1Hand"/(chip+"_S1Hand.tif")) as s:sar=s.read()
  finite=np.isfinite(sar).all(axis=0);valid=np.isin(gt,[0,1])
  for name,mask in [("nonfinite_SAR",~finite),("finite_SAR",finite),("VV_below_minus15",finite&(sar[0]<-15)),("VV_minus15_to_minus10",finite&(sar[0]>=-15)&(sar[0]<-10)),("VV_at_least_minus10",finite&(sar[0]>=-10))]:
   target=np.where(valid&mask,gt,255)
   signal_rows.append(dict(split=split,chip_id=chip,group=name,**binary_metrics(*confusion_counts(pred,target))))
result=[]
for split in ["non_bol","bol"]:
 df=group_metrics([r for r in signal_rows if r["split"]==split],"group");df.insert(0,"split",split);result.append(df)
pd.concat(result).to_csv(OUT/"sar_signal_groups.csv",index=False)
splits={}
for p in (ROOT/"dataset/sen1floods11_hand_labeled_only").rglob("flood_*data.csv"):
 splits[p.stem]=set(pd.read_csv(p,header=None).iloc[:,0].astype(str))
save_json(OUT/"split_audit.json",dict(counts={k:len(v) for k,v in splits.items()},overlaps=[dict(a=a,b=b,count=len(splits[a]&splits[b])) for a in splits for b in splits if a<b]))
fig,ax=plt.subplots(figsize=(12,5))
keys=["water_iou","miou","accuracy","precision","recall","f1","omission_error","commission_error"]
pd.DataFrame({s:totals["upernet_"+s] for s in ["non_bol","bol"]}).loc[keys].plot.bar(ax=ax)
ax.set(ylabel="Fraction",ylim=(0,1));ax.grid(axis="y",alpha=.2)
fig.tight_layout();fig.savefig(OUT/"metrics.png",dpi=160);plt.close(fig)
# Keep comparison provenance and interpretation outside the notebooks.
def table(name):return pd.read_csv(OUT/name).to_html(index=False,float_format=lambda x:f"{x:.4f}")
html="""<!doctype html><html lang="en"><meta charset="utf-8"><title>CROMA UPerNet error analysis</title>
<style>body{font:16px system-ui;max-width:1300px;margin:36px auto;padding:0 24px;line-height:1.55;color:#183042}h1,h2{line-height:1.2}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #cbd5df;text-align:right}img{width:100%}.scroll{overflow-x:auto}.note{background:#eef4f8;padding:18px}a{color:#1263a0}</style>
<h1>CROMA + UPerNet: where the errors occur</h1>
<p>Analysis of the completed 80-epoch SAR-only experiment, checkpoint selected at epoch 75. Existing predictions were evaluated directly; no retraining or threshold changes were performed. All fractions below use pooled confusion counts over valid labelled pixels, not averages of per-image metrics.</p>
<div class="note"><b>Main finding:</b> the new run has higher precision but substantially lower water recall than the previous own-preprocessing run. Similar aggregate mIoU hides different error behaviour. Finer-looking maps do not establish improved small-water or shoreline detection.</div>
<h2>Current test metrics</h2><img src="metrics.png" alt="Water IoU, mIoU, accuracy, precision, recall, F1, omission and commission">
<div class="scroll">SUMMARY</div>
<p>Omission = FN/(TP+FN) = 1 − recall. Commission = FP/(TP+FP) = 1 − precision. False-positive rate = FP/(FP+TN), which is a different quantity. F1 is water Dice/F1. Accuracy includes both classes; non-water dominates the pixel count. mIoU is the mean of water and non-water IoU. Undefined divisions are NaN in tables and null in JSON, not artificially perfect scores.</p>
<h2>What the existing runs demonstrate</h2>
<ul><li>Non-Bolivia mIoU is almost identical: previous 78.9071%, UPerNet 78.9110%. Water precision rises from 72.8% to 87.8%, while recall falls from 84.8% to 69.5%. These are different prediction behaviours, not identical models.</li>
<li>Bolivia water IoU falls from 61.0% to 42.9%, with recall falling from 91.1% to 45.3%. The run is therefore not uniformly better, despite finer visible structures.</li>
<li>These runs also differ in preprocessing and training protocol. This comparison cannot attribute the changes to the head alone.</li></ul>
<h2>Geography matters</h2>
<p>UPerNet water IoU ranges from 11.7% in Pakistan to 89.1% in Nigeria. Pakistan contributes 174,923 missed water pixels; India contributes 219,648. The model can exceed 0.8 water IoU in some regions. The aggregate is not evidence of a universal architectural ceiling. Bolivia is a separate held-out event; its decline is consistent with a generalization problem, but the responsible environmental factors require further evidence.</p>
<div class="scroll">REGIONS</div>
<h2>Boundary and small-water errors</h2>
<p>On non-Bolivia, water recall within three pixels of known land is 41.9%, versus 80.2% farther inside water. For 8-connected labelled water components of 1–64 pixels, recall is 13.7%; for components larger than 1,024 pixels it is 74.1%. Nonetheless, 592,577 missed water pixels belong to the largest component group: small structures alone do not explain the total error.</p>
<p>The shoreline statistic is water recall in a spatial band, not a boundary F1 score. Water within three dilation steps of ignored labels is excluded. Component sizes are measured in raster pixels and are truncated at chip boundaries; these are descriptive strata, not independent objects or causal tests.</p>
<div class="scroll">SHORE</div><div class="scroll">SIZES</div>
<h2>SAR signal and invalid input</h2>
<p>Among finite-input non-Bolivia pixels, water recall is 89.9% for VV &lt; −15 dB, 47.7% for −15 ≤ VV &lt; −10 dB, and 30.7% for VV ≥ −10 dB. Thus the errors are strongly associated with radar-brighter labelled water. Radar scattering, vegetation, roughness, mixed pixels, and image/label timing are plausible explanations to check; this analysis does not identify which applies to any specific pixel. Thresholds were chosen for descriptive analysis, not model tuning.</p>
<p>73,408 valid non-Bolivia pixels and 43,381 valid Bolivia pixels contain nonfinite values in at least one SAR band. The current upstream-aligned loader fills NaNs with zero before fixed normalization, which can look bright in dB. Nonfinite input accounts for 50,530 of 783,512 non-Bolivia false negatives (6.45%). Restricting evaluation to finite SAR increases non-Bolivia water IoU from 63.33% to 64.48%; this is a subset diagnostic, not a replacement benchmark score or proof that changing the fill will recover those pixels.</p>
<div class="scroll">SIGNAL</div>
<h2>Readable examples using the actual input</h2>
<p>Each split contributes its two chips with the most missed water pixels. These are selected failure cases, not a representative sample. SAR is displayed with fixed VV/VH stretches; both channels are the model's actual sensor modality. Blue indicates water in the label/prediction panels; the error panel uses green for TP, red for FN, orange for FP, dark grey for TN, grey for ignored labels. The earlier notebook montage uses Sentinel-2 RGB, which the SAR-only model does not receive.</p>
<a href="error_examples.png"><img src="error_examples.png" alt="SAR channels, labels, old and new predictions, and error maps"></a>
<h2>Comparison with published work</h2>
<p><a href="https://arxiv.org/html/2412.04204v1#A4">PANGAEA Table A6</a> reports CROMA at 90.89% mIoU / 84.15% water IoU with 100% labels. That table does not identify the CROMA input branch. The paper also groups Sen1Floods11 with multispectral tasks. Matching the public SAR configuration and decoder does not establish that this run reproduces that table row. Historical modality, split/version, resolved configuration and checkpoint must be matched before interpreting the gap as a failed reproduction. Frozen-encoder training is also part of PANGAEA's main protocol, so freezing alone is not a sufficient explanation.</p>
<p>A clearly labelled SAR reference is <a href="https://openaccess.thecvf.com/content/ICCV2025/papers/Jakubik_TerraMind_Large-Scale_Generative_Multimodality_for_Earth_Observation_ICCV_2025_paper.pdf">TerraMind (ICCV 2025)</a>: Sentinel-1 full fine-tuning reports 68.00% water IoU and 81.06% mIoU. The numerical gaps from this non-Bolivia run are 4.67 and 2.15 percentage points. This is contextual evidence, not a controlled comparison or a claim to reproduce its split and training. It is not an exhaustive SOTA ranking.</p>
<p>The local PANGAEA reference snapshot came from <a href="https://github.com/VMarsocci/pangaea-bench/tree/ca19b837a16797cd809225225b671473c7c93768">the official public repository, pinned revision ca19b837a16797cd809225225b671473c7c93768</a>. It is source/configuration reference material, not the authors' trained task checkpoint.</p>
<h2>Checks and limitations</h2>
<ul><li>Recomputed TP/FP/FN/TN match the original saved test metrics exactly for both runs and both splits. Prediction and label raster shapes, CRS and affine transforms match within 1e-10 coordinate tolerance.</li>
<li>Local split sizes: train 252, validation 89, non-Bolivia test 90, Bolivia 15. No filename overlap between these splits. This does not establish geographic independence or equivalence with the historical PANGAEA release.</li>
<li>Existing implementation tests compare the intermediate CROMA features and UPerNet against the pinned upstream code and check the inference grid, normalization, scheduler and checkpoint selection.</li>
<li>Prior validation epochs do not contain the new metrics; they cannot be reconstructed from loss/IoU alone. Future training records the full metrics each validation interval (five epochs). Training metrics use sampled augmented crops, so they are not directly comparable to full-scene validation metrics.</li></ul>
<h2>Next controlled experiments</h2>
<ol><li>Evaluate probability thresholds on validation predictions only, then lock the selected threshold before testing. This will distinguish a decision-threshold issue from a ranking/representation issue. The saved hard masks cannot answer it.</li><li>Audit radar-bright and nonfinite-input failure cases with dates, original SAR and optical imagery, and labels. Test alternative missing-data handling as a separate run; do not silently exclude labelled pixels.</li><li>Hold splits, preprocessing, loss, schedule and seed fixed when comparing heads. Then vary class-aware sampling/loss or encoder fine-tuning one at a time, reporting per-region precision/recall and multiple seeds.</li><li>Resolve the authors' exact input modality and historical configuration before attempting the 90.89% PANGAEA reproduction. If that requires optical input, keep it as a separately named experiment.</li></ol>
<p><b>Explanation for a professor:</b> “UPerNet produces a different precision–recall tradeoff. It reduces false alarms but misses more labelled water, particularly near shorelines, in radar-bright areas and in several geographic regions. The pooled mIoU hides those differences. We have verified the metric calculation; the remaining explanations require controlled input, threshold and training ablations.”</p>
<p>Artifacts: <a href="metrics_summary.csv">all aggregate metrics</a> · <a href="per_chip.csv">per-chip metrics</a> · <a href="per_region.csv">per-region metrics</a> · <a href="provenance.json">run provenance</a> · <a href="split_audit.json">split audit</a>.</p></html>"""
for token,name in [("SUMMARY","metrics_summary.csv"),("REGIONS","per_region.csv"),("SHORE","shoreline.csv"),("SIZES","water_body_size.csv"),("SIGNAL","sar_signal_groups.csv")]:
 html=html.replace(token,table(name))
(OUT/"index.html").write_text(html,encoding="utf-8")
