"""Export reproducible 16:9 figures from the verified prediction comparison."""
import json, sys, os, zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[3]
os.chdir(ROOT)
source=Path(sys.argv[1])
out=Path(sys.argv[2]) if len(sys.argv)>2 else Path("reports")/("presentation_"+datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f"))
out.mkdir(parents=True,exist_ok=len(sys.argv)>2)
summary=pd.read_csv(source/"integer_label_summary.csv")
chips=pd.read_csv(source/"integer_label_per_chip.csv")
assets=source/"assets"
models=["hand_labeled","s1_weak","s2_weak","permanent_water","CROMA own float"]
names=["Authors: hand labels","Authors: S1 weak","Authors: S2 weak","Authors: permanent water","CROMA + FCN (own preprocessing)"]
short=["Hand labels","S1 weak","S2 weak","Permanent water","CROMA + FCN"]
colors=["#688494","#8D6AA8","#299E88","#BC8A4C","#0072CE"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":15,"axes.titlesize":20,"axes.labelsize":16,"xtick.labelsize":13,"ytick.labelsize":15,"pdf.fonttype":42,"ps.fonttype":42,"savefig.facecolor":"white"})
notes=[]
common="Test v1.1 | Raw integer-label scoring | Single checkpoints; training data and supervision differ"
slides=PdfPages(out/"all_figures.pdf")
def finish(fig,stem,title,caption,selection=None):
    fig.savefig(out/(stem+".png"),dpi=200)
    fig.savefig(out/(stem+".pdf"))
    slides.savefig(fig)
    plt.close(fig)
    notes.append({"file":stem,"title":title,"caption":caption,"selection":selection})
    print("EXPORTED",stem,flush=True)
def heading(fig,title,subtitle=common):
    fig.text(.045,.955,title,size=25,weight="bold",color="#15283c",va="top")
    fig.text(.045,.905,subtitle,size=12,color="#536477",va="top")
def footer(fig,text):fig.text(.045,.035,text,size=11,color="#536477",va="bottom")
def data(split,cat="AW",version="v1.1"):
    return summary[(summary.test_dataset==version)&(summary.split==split)&(summary.category==cat)].set_index("model").reindex(models)
for metric,stem,title,xlabel in [
 ("mean_chip_iou","01_mean_water_iou","CROMA is close to the strongest authors' model on Non-Bolivia","Mean per-image all-water IoU"),
 ("pooled_iou","02_pooled_water_iou","Pooled IoU tells a different story from the image average","Pooled all-water IoU")]:
    fig,axs=plt.subplots(1,2,figsize=(16,9))
    heading(fig,title)
    for ax,split,label in zip(axs,["non_bol","bol"],["Non-Bolivia (90 images)","Bolivia (15 images)"]):
        vals=data(split)[metric].to_numpy()
        bars=ax.barh(np.arange(5),vals,color=colors,height=.6)
        ax.set_yticks(np.arange(5),short);ax.invert_yaxis();ax.set_xlim(0,1)
        ax.set_xlabel(xlabel);ax.set_title(label,pad=18)
        ax.bar_label(bars,labels=[f"{v:.3f}" for v in vals],padding=7,fontsize=16)
        ax.grid(axis="x",alpha=.18);ax.set_axisbelow(True)
        for side in ["top","right","left"]:ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=.14,right=.97,bottom=.16,top=.81,wspace=.52)
    footer(fig,"Higher is better. Figures 01 and 02 use identical predictions, with different IoU aggregation.")
    finish(fig,stem,title,"AW = all water. Every image has equal weight in mean per-image IoU; pooled IoU sums pixel counts first. Scores are descriptive, without multi-seed uncertainty estimates.")
fig,axs=plt.subplots(1,2,figsize=(16,9));heading(fig,"Performance depends on the water category")
for ax,split,label in zip(axs,["non_bol","bol"],["Non-Bolivia","Bolivia"]):
    mat=np.column_stack([data(split,c).mean_chip_iou.to_numpy() for c in ["PW","FW","AW"]])
    im=ax.imshow(mat,vmin=0,vmax=.5,cmap="Blues",aspect="auto")
    ax.set_xticks(range(3),["Permanent (PW)","Flood (FW)","All water (AW)"]);ax.set_yticks(range(5),short)
    counts=[int(data(split,c).chips.iloc[0]) for c in ["PW","FW","AW"]]
    ax.set_title(label+" - mean per-image IoU",pad=20)
    for r in range(5):
        for c in range(3):ax.text(c,r,f"{mat[r,c]:.3f}",ha="center",va="center",fontsize=19,color="white" if mat[r,c]>.3 else "#15283c",weight="bold" if r==4 else "normal")
    ax.set_xlabel("Evaluated images: "+" / ".join(map(str,counts)),labelpad=16)
fig.subplots_adjust(left=.14,right=.96,bottom=.20,top=.81,wspace=.50)
footer(fig,"PW/FW follow the notebook's category masking and image-exclusion rules. Grey ignored labels are never scored.")
finish(fig,"03_water_categories","Performance depends on the water category","Mean per-image IoU for PW, FW and AW. PW uses fewer eligible images (54 Non-Bolivia; 7 Bolivia); category scores are not interchangeable.")
fig,axs=plt.subplots(1,2,figsize=(16,9));heading(fig,"CROMA's high recall comes with more false water predictions")
for ax,split,label in zip(axs,["non_bol","bol"],["Non-Bolivia","Bolivia"]):
    d=data(split)
    for i,m in enumerate(models):
        row=d.loc[m];ax.scatter(row.precision,row.recall,s=220,color=colors[i],marker="D" if i==4 else "o",edgecolor="white",linewidth=1.5,zorder=3)
        ax.annotate(str(i+1),(row.precision,row.recall),xytext=(-18,-18) if split=="bol" and i==0 else (9,7),textcoords="offset points",fontsize=15,weight="bold",zorder=5)
    ax.set(xlim=(.5,1),ylim=(.6,1),xlabel="Precision: fraction of predicted water that is correct",ylabel="Recall: fraction of labeled water found",title=label)
    ax.grid(alpha=.2)
fig.legend([Line2D([],[],marker="D" if i==4 else "o",color=c,linestyle="",markersize=10) for i,c in enumerate(colors)],[f"{i+1}. {x}" for i,x in enumerate(short)],loc="lower center",bbox_to_anchor=(.5,.07),ncol=3,frameon=False,fontsize=14)
fig.subplots_adjust(left=.08,right=.97,top=.81,bottom=.26,wspace=.28)
footer(fig,"Pooled AW precision/recall. Moving toward the upper-right improves both; these are fixed argmax predictions.")
finish(fig,"04_precision_recall","CROMA's high recall comes with more false water predictions","CROMA Bolivia: recall 0.911, precision 0.649. The comparison does not identify whether the cause is the decoder, preprocessing, loss weighting, or backbone training.")
aw=chips[(chips.test_dataset=="v1.1")&(chips.category=="AW")]
wide=aw.pivot(index=["split","chip"],columns="model",values="chip_iou")
nb=wide.loc["non_bol"].copy();nb["delta"]=nb["CROMA own float"]-nb.s2_weak
nbmedian=nb.index[np.argmin(np.abs(nb['CROMA own float']-nb['CROMA own float'].median()))]
bol=wide.loc['bol'];bolmedian=bol.index[np.argmin(np.abs(bol['CROMA own float']-bol['CROMA own float'].median()))]
cases=[("non_bol",nbmedian,"05_typical_non_bolivia","A scene near CROMA's median Non-Bolivia IoU","Nearest to median CROMA AW IoU; ties follow CSV/pivot order"),
       ("bol",bolmedian,"06_typical_bolivia","A scene near CROMA's median Bolivia IoU","Nearest to median CROMA AW IoU"),
       ("non_bol",nb.delta.idxmax(),"07_croma_advantage","A selected case where CROMA outperforms the S2-weak FCN","Largest CROMA minus S2-weak per-image AW IoU on Non-Bolivia; illustrative extreme, not representative"),
       ("non_bol",nb.delta.idxmin(),"08_croma_failure","A selected case where CROMA underperforms the S2-weak FCN","Smallest CROMA minus S2-weak per-image AW IoU on Non-Bolivia; illustrative extreme, not representative")]
def image(split,chip,suffix):return np.array(Image.open(assets/f"v1.1_{split}_{chip}_{suffix}.png"))
for split,chip,stem,title,rule in cases:
    fig,axs=plt.subplots(2,4,figsize=(16,9))
    heading(fig,title,f"{chip} | Test v1.1 | {'Non-Bolivia' if split=='non_bol' else 'Bolivia'} | AW masks; values are per-image IoU")
    panels=[("rgb","Sentinel-2 RGB (context only)",None),("vv","SAR VV (model input)",None),("label","Ground truth",None),("CROMA_own_float","CROMA + FCN","CROMA own float"),("hand_labeled","Authors: hand labels","hand_labeled"),("s1_weak","Authors: S1 weak","s1_weak"),("s2_weak","Authors: S2 weak","s2_weak"),("permanent_water","Authors: permanent water","permanent_water")]
    for ax,(suffix,title2,m) in zip(axs.flat,panels):
        arr=image(split,chip,suffix);ax.imshow(arr,cmap="gray" if arr.ndim==2 else None,vmin=0,vmax=255)
        if m is not None:title2+=f" | IoU {wide.loc[(split,chip),m]:.3f}"
        ax.set_title(title2,fontsize=13,pad=6,color="#0072CE" if m=="CROMA own float" else "#15283c",weight="bold");ax.axis('off')
    fig.subplots_adjust(left=.03,right=.97,top=.82,bottom=.10,wspace=.08,hspace=.15)
    footer(fig,"Cyan: water | Dark: non-water | Grey: ignored labels | Same extent and mask for every model; RGB is not model input")
    finish(fig,stem,title,f"Chip {chip}. {rule}. All panels show the complete 512x512 image; images are display-scaled only.",{"split":split,"chip":chip,"rule":rule})
# Close-ups use the same window in every model; pick maximum valid prediction disagreement.
fig,axs=plt.subplots(2,4,figsize=(16,9));heading(fig,"Close-ups separate missed boundaries from false water regions","Test v1.1 | Same 128x128 crop across panels; nearest-neighbor display | Selected disagreement regions")
rois=[]
for r,(split,chip,_,_,_) in enumerate(cases[-2:]):
    a=image(split,chip,'s2_weak');b=image(split,chip,'CROMA_own_float');label=image(split,chip,'label')
    valid=~np.all(label==[190,190,190],axis=-1);diff=np.any(a!=b,axis=-1)&valid
    choices=[(int(diff[y:y+128,x:x+128].sum()),y,x) for y in range(0,385,32) for x in range(0,385,32)]
    _,y,x=max(choices,key=lambda t:t[0]);rois.append({"chip":chip,"split":split,"top":y,"left":x,"size":128,"rule":"maximum valid mask disagreement between CROMA and S2-weak over a 32-pixel search grid"})
    for c,(suffix,title) in enumerate([('rgb','RGB context'),('label','Ground truth'),('s2_weak_errors','Authors S2-weak: errors'),('CROMA_own_float_errors','CROMA: errors')]):
        arr=image(split,chip,suffix)[y:y+128,x:x+128];axs[r,c].imshow(arr,interpolation='nearest');axs[r,c].axis('off');axs[r,c].set_title(f"{title}\n{chip}",fontsize=14)
fig.subplots_adjust(left=.04,right=.96,top=.81,bottom=.12,wspace=.12,hspace=.20)
footer(fig,"Error maps: cyan = correct water, orange = false water, pink = missed water, grey = ignored. Crops are illustrative extremes.")
finish(fig,"09_boundary_closeups","Close-ups separate missed boundaries from false water regions","Close-up positions maximize model disagreement, not representativeness. Model errors are measured against the same integer-label ground truth.",rois)
fig,axs=plt.subplots(1,2,figsize=(16,9));heading(fig,"Dataset version changes the evaluation result","Authors' identical checkpoints on v1.0 versus v1.1 | Raw integer-label mean per-image AW IoU")
for ax,split,label in zip(axs,['non_bol','bol'],['Non-Bolivia','Bolivia']):
    for i,m in enumerate(models[:4]):
        vals=[float(data(split,version=v).loc[m,'mean_chip_iou']) for v in ['v1.0','v1.1']]
        ax.plot([0,1],vals,'o-',color=colors[i],linewidth=2.5,markersize=9,label=short[i])
        ax.text(-.035,vals[0],f"{vals[0]:.3f}",ha='right',va='center',fontsize=12,color=colors[i])
        dy = (-9 if i==0 else 9 if i==1 else 0) if split=="non_bol" else (9 if i==0 else -9 if i==3 else 0)
        ax.annotate(f"{vals[1]:.3f}",(1,vals[1]),xytext=(14,dy),textcoords="offset points",ha="left",va="center",fontsize=12,color=colors[i])
    ax.set(xlim=(-.25,1.25),ylim=(0,0.6),xticks=[0,1],xticklabels=['Test v1.0','Test v1.1'],ylabel='Mean per-image AW IoU',title=label)
    ax.grid(axis='y',alpha=.2)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.08),ncol=4,frameon=False)
fig.subplots_adjust(left=.08,right=.97,top=.81,bottom=.23,wspace=.26)
footer(fig,"Appendix: dataset versions are separate evaluations. Do not attribute changes here to retraining or preprocessing changes.")
finish(fig,"10_dataset_version_appendix","Dataset version changes the evaluation result","The same four authors' checkpoints are evaluated on the two dataset releases. Different inputs/labels can change metrics; this is not a training comparison.")
slides.close()
(out/'figure_notes.json').write_text(json.dumps({'source_comparison':source.as_posix(),'figures':notes,'model_names':dict(zip(models,names))},indent=2))
lines=['# Presentation figures','', 'All figures are 16:9, exported as 3200x1800 PNG and vector PDF. `all_figures.pdf` contains the complete set.','', '## Suggested main slides','', 'Use 01 (mean IoU), 04 (precision/recall), 05 or 06 (typical scene), and 09 (close-ups). Keep 02/03/07/08/10 as discussion or appendix material.','', '## Interpretation safeguards','', '- Metrics use raw integer labels with identical ignore handling, not the legacy notebook label-conversion scores.','- Main comparisons use test v1.1. Authors weights were trained on v1.0 under differing supervision; CROMA uses the current own-preprocessing head. This does not isolate architecture or preprocessing.','- One checkpoint per model: no uncertainty bars or significance claims.','- Mean per-image IoU and pooled IoU are different metrics. Empty-union chip IoU follows the existing epsilon convention (1).','- Both decoders predict on an effective stride-8 grid; these results alone cannot identify the FCN as the cause.','- RGB provides context only. Models consume SAR, and optical/SAR appearances need not match.','']
for n in notes:lines += ['## '+n['file'],n['title'],'',n['caption'],'']
(out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
# Self-contained contact sheet/index for quick selection.
html='<html><head><meta charset="utf-8"><title>Presentation figures</title><style>body{font:18px Arial;background:#eef2f6;margin:32px}img{max-width:1100px;width:100%;display:block}section{background:white;padding:24px;margin:24px 0}a{margin-right:16px}</style></head><body><h1>Presentation figures</h1><p>16:9 PNGs and PDFs. See README for captions, methods and selection rules.</p>'
for n in notes:html+=f'<section><h2>{n["file"]}</h2><a href="{n["file"]}.png">PNG</a><a href="{n["file"]}.pdf">PDF</a><p>{n["caption"]}</p><img src="{n["file"]}.png" loading="lazy"></section>'
(out/'index.html').write_text(html+'</body></html>',encoding='utf-8')
with zipfile.ZipFile(out/'presentation_figures.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in out.iterdir():
        if p.suffix in ['.png','.pdf','.md','.json','.html']:z.write(p,p.name)
print('OUTPUT',out,flush=True)
