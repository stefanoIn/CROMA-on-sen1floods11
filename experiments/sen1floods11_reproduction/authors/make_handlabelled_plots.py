"""Four-way hand-labelled presentation: authors, personal FCN, CROMA, Otsu."""
from pathlib import Path
from datetime import datetime
import os,sys,csv,json,hashlib,zipfile
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[3];os.chdir(ROOT);sys.path.insert(0,"src")
from sen1floods11_visualization import _load_case
suite=Path("runs/sen1floods11_reproduction/comparison_2026-10-20_18-31-06_611597_d03e130b")
manifest=json.loads((suite/"manifest.json").read_text())
auth=Path(next(r["directory"] for r in manifest["runs"] if r["model"]=="hand_labeled" and r["test_dataset"]=="v1.1"))
croma=Path(manifest["croma_comparison_run"])
mine=Path("fcnn_test/notebook_results/mine_train_v1_1_test_v1_1")
hand=Path("dataset/sen1floods11_hand_labeled_only/HandLabeled")
methods=["authors","mine","croma","otsu"]
names=["Authors FCN (v0v0)","My FCN (v1.1)","CROMA + FCN","Otsu"]
colors=["#718096","#8b62aa","#0072ce","#d28a29"]
out=Path("reports")/("handlabelled_four_models_"+datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f"));out.mkdir(parents=True)
provenance={"models":dict(zip(methods,names)),"authors_run":auth.as_posix(),"personal_fcn_run":mine.as_posix(),"croma_run":croma.as_posix(),"otsu":"dataset/sen1floods11_hand_labeled_only/HandLabeled/S1OtsuLabelHand","test_version":"v1.1","authors_version_note":"v0v0 is the user's notebook name; authors checkpoint was trained on dataset v1.0","personal_checkpoint":"fcnn_test/fcnn_checkpoints/mine_train_v1_1/Sen1Floods11_hand_labeled_epoch051_valIoU0.539782.cp","personal_checkpoint_sha256":"d54e9e8eaa460bee4d5f3651dc545ff75ceb1722b29d7868210a9c3f070c673b"}
assert hashlib.sha256(Path(provenance['personal_checkpoint']).read_bytes()).hexdigest()==provenance['personal_checkpoint_sha256']
cache={};rows=[];coverage=[]
def read(p,grid=None):
 with rasterio.open(p) as src:
  if grid:assert src.shape==grid[0] and src.crs==grid[1] and src.transform.almost_equals(grid[2],precision=1e-10),p
  a=src.read(1).astype(np.int16);g=(src.shape,src.crs,src.transform)
 a[a==-1]=255
 return a,g
for split,file in [("non_bol","flood_test_data.csv"),("bol","flood_bolivia_data.csv")]:
 for x,yname in csv.reader((hand.parent/"flood_handlabeled"/file).open()):
  chip=Path(x).stem.removesuffix('_S1Hand')
  y,grid=read(hand/'LabelHand'/Path(yname).name)
  perm,_=read(hand/'JRCWaterHand'/f'{chip}_JRCWaterHand.tif',grid)
  preds={}
  for m in methods:
   path=(hand/'S1OtsuLabelHand'/f'{chip}_S1OtsuLabelHand.tif') if m=='otsu' else ((auth/'predictions'/split if m=='authors' else mine/'predictions'/split if m=='mine' else croma/split)/f'{chip}_S1Hand_prediction.tif')
   preds[m],_=read(path,grid)
   assert np.isin(preds[m],[0,1,255] if m=='otsu' else [0,1]).all()
  common=np.isin(y,[0,1])&np.logical_and.reduce([np.isin(p,[0,1]) for p in preds.values()])
  coverage.append({'split':split,'chip':chip,'label_valid':int(np.isin(y,[0,1]).sum()),'common_valid':int(common.sum())})
  cache[(split,chip)]={'y':y,'perm':perm,'preds':preds,'common':common}
  for cat in ['PW','FW','AW']:
   target=y.copy()
   if cat=='PW':target[(y==1)&(perm==0)]=255
   if cat=='FW':target[(y==1)&(perm==1)]=255
   if cat!='AW' and (y==1).any() and not (target==1).any():continue
   for protocol in ['common_valid','native_masks']:
    for m,pred in preds.items():
     valid=np.isin(target,[0,1]) & (common if protocol=='common_valid' else np.isin(pred,[0,1]))
     if protocol=='common_valid' and not valid.any():continue
     tp=int((valid&(target==1)&(pred==1)).sum());fp=int((valid&(target==0)&(pred==1)).sum());fn=int((valid&(target==1)&(pred==0)).sum());tn=int((valid&(target==0)&(pred==0)).sum())
     rows.append({'protocol':protocol,'split':split,'category':cat,'model':m,'chip':chip,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'valid_pixels':int(valid.sum()),'chip_iou':(tp+1e-7)/(tp+fp+fn+1e-7)})
frame=pd.DataFrame(rows)
summary=frame.groupby(['protocol','split','category','model'],sort=False).agg(chips=('chip','size'),mean_chip_iou=('chip_iou','mean'),tp=('tp','sum'),fp=('fp','sum'),fn=('fn','sum'),tn=('tn','sum'),valid_pixels=('valid_pixels','sum')).reset_index()
summary['pooled_iou']=summary.tp/(summary.tp+summary.fp+summary.fn)
for _,g in summary[summary.protocol=='common_valid'].groupby(['split','category']):
 assert g.chips.nunique()==1 and g.valid_pixels.nunique()==1 and (g.tp+g.fn).nunique()==1
frame.to_csv(out/'per_chip_scores.csv',index=False);summary.to_csv(out/'summary.csv',index=False);pd.DataFrame(coverage).to_csv(out/'pixel_coverage.csv',index=False)
# Verify current CROMA rescore reproduces its own saved table under its original convention.
old=pd.read_csv(croma/'croma_aw_pw_fw_summary.csv').set_index(['split','category'])
new=summary[(summary.model=='croma')&(summary.protocol=='native_masks')].set_index(['split','category']).loc[old.index]
np.testing.assert_allclose(old.mean_chip_iou,new.mean_chip_iou,atol=1e-12,rtol=0)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.titlesize':18,'axes.labelsize':15,'pdf.fonttype':42})
book=PdfPages(out/'all_figures.pdf');notes=[]
pltcolors=np.array([[35,43,53],[0,190,225],[190,190,190]],dtype=np.uint8)
def head(fig,title,subtitle):
 fig.text(.04,.96,title,va='top',fontsize=25,weight='bold',color='#15283c')
 fig.text(.04,.905,subtitle,va='top',fontsize=12,color='#536477')
def foot(fig,text):fig.text(.04,.035,text,fontsize=11,color='#536477')
def save(fig,name,caption,selection=None):
 fig.savefig(out/(name+'.png'),dpi=200);fig.savefig(out/(name+'.pdf'));book.savefig(fig);plt.close(fig)
 notes.append({'file':name,'caption':caption,'selection':selection});print('EXPORTED',name,flush=True)
def scores(split,cat='AW'):
 return summary[(summary.protocol=='common_valid')&(summary.split==split)&(summary.category==cat)].set_index('model').reindex(methods)
fig,axs=plt.subplots(1,2,figsize=(16,9));head(fig,'Hand-labelled comparison: authors, my FCN, CROMA and Otsu','Same test set: Sen1Floods11 v1.1 | Same valid pixels for all four models | Mean per-image water IoU')
for ax,split,title in zip(axs,['non_bol','bol'],['Non-Bolivia','Bolivia']):
 d=scores(split);bars=ax.barh(range(4),d.mean_chip_iou,color=colors,height=.6)
 ax.set_yticks(range(4),names);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_title(f'{title} ({int(d.chips.iloc[0])} evaluated images)');ax.set_xlabel('Mean per-image AW IoU')
 ax.bar_label(bars,labels=[f'{v:.3f}' for v in d.mean_chip_iou],padding=7,fontsize=16);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
 for side in ['top','right','left']:ax.spines[side].set_visible(False)
fig.subplots_adjust(left=.18,right=.97,top=.81,bottom=.15,wspace=.8)
foot(fig,'AW = all water. Grey / missing pixels excluded for all models; entirely unscored images omitted. One checkpoint per model.')
save(fig,'01_handlabelled_scores','Exactly four methods. Shared valid pixels exclude Otsu missing predictions for all models. This is not the legacy notebook label-conversion metric; native-mask scores are also available in summary.csv.')
fig,axs=plt.subplots(1,2,figsize=(16,9));head(fig,'Hand-labelled comparison by water category','Sen1Floods11 v1.1 | Shared valid pixels | Mean per-image IoU | Same four methods')
for ax,split,title in zip(axs,['non_bol','bol'],['Non-Bolivia','Bolivia']):
 mat=np.column_stack([scores(split,c).mean_chip_iou for c in ['PW','FW','AW']]);ax.imshow(mat,vmin=0,vmax=1,cmap='Blues',aspect='auto')
 ax.set_xticks(range(3),['Permanent (PW)','Flood (FW)','All water (AW)']);ax.set_yticks(range(4),names);ax.set_title(title,pad=18)
 for r in range(4):
  for c in range(3):ax.text(c,r,f'{mat[r,c]:.3f}',ha='center',va='center',fontsize=19,color='white' if mat[r,c]>.55 else '#15283c')
 ax.set_xlabel('Evaluated images: '+' / '.join(str(int(scores(split,c).chips.iloc[0])) for c in ['PW','FW','AW']),labelpad=15)
fig.subplots_adjust(left=.18,right=.97,top=.81,bottom=.19,wspace=.8)
foot(fig,'PW/FW use the existing category masks and exclusion rules. Otsu is the thresholding baseline, not a trained model.')
save(fig,'02_handlabelled_categories','PW, FW and AW evaluated with the same valid-pixel mask for the four methods; different categories have different eligible images.')
# Selection uses labels only, never model scores. Include visually informative water-bearing cases.
rng=np.random.default_rng(42);selected=[]
for split,count in [('non_bol',4),('bol',2)]:
 eligible=sorted(chip for (sp,chip),c in cache.items() if sp==split and c['common'].sum()>=10000 and (c['common']&(c['y']==1)).sum()>=1000)
 selected.extend((split,str(x)) for x in rng.choice(eligible,count,replace=False))
def rgb(split,chip):
 return _load_case(auth/'predictions'/split/f'{chip}_S1Hand_prediction.tif',hand,s2_scale=3000)[1]
def mask(arr,valid):return pltcolors[np.where(valid,arr,2).astype(int)]
for idx,(split,chip) in enumerate(selected,3):
 c=cache[(split,chip)];valid=c['common'];fig,axs=plt.subplots(2,3,figsize=(16,9))
 head(fig,'Hand-labelled models and Otsu: same scene, same extent',f'{chip} | {"Non-Bolivia" if split=="non_bol" else "Bolivia"} | Test v1.1 | All-water predictions')
 panels=[(rgb(split,chip),'RGB context (not model input)'),(mask(c['y'],valid),'Ground truth')]
 for m,name in zip(methods,names):
  value=frame[(frame.protocol=='common_valid')&(frame.split==split)&(frame.category=='AW')&(frame.chip==chip)&(frame.model==m)].chip_iou.iloc[0]
  panels.append((mask(c['preds'][m],valid),f'{name} | IoU {value:.3f}'))
 for ax,(arr,title) in zip(axs.flat,panels):ax.imshow(arr,interpolation='nearest');ax.set_title(title,fontsize=15,pad=6);ax.axis('off')
 fig.subplots_adjust(left=.04,right=.96,top=.81,bottom=.10,wspace=.07,hspace=.18)
 foot(fig,'Cyan = water | Dark = non-water | Grey = excluded for all methods | Authors v0v0 checkpoint: trained on dataset v1.0')
 save(fig,f'{idx:02d}_{split}_{chip}',f'Full 512x512 scene {chip}. Only the four requested methods. RGB is contextual; models use SAR.',{'chip':chip,'split':split,'seed':42,'rule':'uniform label-only selection: >=10000 common-valid pixels and >=1000 water pixels; no score ranking'})
# One zoom per split; same crop for all models, selected by four-way disagreement.
for idx,split in enumerate(['non_bol','bol'],9):
 chip=next(ch for sp,ch in selected if sp==split);c=cache[(split,chip)];valid=c['common'];stack=np.stack([c['preds'][m] for m in methods]);disagree=(stack.min(axis=0)!=stack.max(axis=0))&valid
 _,top,left=max([(int(disagree[y:y+128,x:x+128].sum()),y,x) for y in range(0,385,32) for x in range(0,385,32)],key=lambda t:t[0]);crop=np.s_[top:top+128,left:left+128]
 fig,axs=plt.subplots(2,3,figsize=(16,9));head(fig,'Where the four methods disagree',f'{chip} | Test v1.1 | Identical 128x128 crop | {split} | Diagnostic close-up')
 panels=[(rgb(split,chip)[crop],'RGB context'),(mask(c['y'],valid)[crop],'Ground truth')]
 for m,name in zip(methods,names):
  pred=c['preds'][m];err=np.full((*valid.shape,3),35,dtype=np.uint8);err[~valid]=[190,190,190];err[valid&(c['y']==1)&(pred==1)]=[0,190,225];err[valid&(c['y']==0)&(pred==1)]=[255,145,0];err[valid&(c['y']==1)&(pred==0)]=[225,40,110]
  panels.append((err[crop],name+' - errors'))
 for ax,(arr,title) in zip(axs.flat,panels):ax.imshow(arr,interpolation='nearest');ax.set_title(title,fontsize=15);ax.axis('off')
 fig.subplots_adjust(left=.04,right=.96,top=.81,bottom=.11,wspace=.07,hspace=.18)
 foot(fig,'Cyan = correct water | Orange = false water | Pink = missed water | Grey = ignored | Crop maximizes model disagreement')
 save(fig,f'{idx:02d}_closeup_{split}',f'Four-method error comparison in one common crop. Diagnostic disagreement crop, not a representative performance estimate.',{'split':split,'chip':chip,'top':top,'left':left,'size':128,'rule':'maximum four-model valid-pixel disagreement on a 32-pixel search grid'})
book.close();provenance['figures']=notes;provenance['selection_seed']=42
(out/'provenance.json').write_text(json.dumps(provenance,indent=2))
text=['# Corrected comparison: only four requested methods','','1. Authors hand-labelled FCN (v0v0 notebook; checkpoint trained on dataset v1.0).','2. Your original hand-labelled FCN v1.1, epoch051. Not the separately corrected epoch72 model.','3. Your current hand-labelled CROMA + FCN, own float preprocessing.','4. Stored Otsu baseline.','','All evaluated on v1.1. No S1-weak, S2-weak or permanent-water-trained model is included.','','Ten 16:9 figures: scores, category scores, six full-scene comparisons, two error close-ups. Each is a 3200x1800 PNG and PDF. `all_figures.pdf` contains all ten.','','Figures use common valid pixels (valid labels and a prediction from every method). Otsu missing predictions are excluded for all models. Fully unscored images are omitted. Therefore values may differ from earlier tables. summary.csv also includes native-mask scores matching the existing CROMA convention. Truly dry empty-union IoU is 1. Category exclusion rules are preserved.','','Case selection: fixed seed42 among chips with >=10000 common-valid pixels and >=1000 labeled water pixels. No prediction scores are used to select full scenes. Close-ups explicitly maximize disagreement within selected scenes.','','PNG/PDF titles are in English for presentation use. CROMA and personal FCN run identifiers and checkpoint provenance are in provenance.json.']
(out/'README.md').write_text('\n'.join(text),encoding='utf-8')
html='<html><head><meta charset="utf-8"><title>Hand-labelled: four-model comparison</title><style>body{font:18px Arial;margin:32px;background:#edf2f6}section{background:white;padding:24px;margin:20px 0}img{width:100%;max-width:1200px}a{margin-right:20px}</style></head><body><h1>Hand-labelled: authors / my FCN v1.1 / CROMA / Otsu</h1><p>Only the four requested methods. Same v1.1 test images and valid pixels.</p>'
for n in notes:html+=f'<section><h2>{n["file"]}</h2><a href="{n["file"]}.png">PNG</a><a href="{n["file"]}.pdf">PDF</a><img src="{n["file"]}.png" loading="lazy"><p>{n["caption"]}</p></section>'
(out/'index.html').write_text(html+'</body></html>',encoding='utf-8')
with zipfile.ZipFile(out/'handlabelled_four_models.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in out.iterdir():
  if p.suffix!='.zip':z.write(p,p.name)
print('OUTPUT',out,flush=True);print(summary[(summary.protocol=='common_valid')&(summary.category=='AW')][['split','model','chips','mean_chip_iou','pooled_iou']].to_string(index=False),flush=True)
