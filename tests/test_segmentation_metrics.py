import json,math,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from segmentation_metrics import *

class MetricTests(unittest.TestCase):
 def test_counts_ignore_and_formulas(self):
  pred=np.array([1,1,0,0,1,0]);gt=np.array([1,0,1,0,255,-1])
  self.assertEqual(confusion_counts(pred,gt),(1,1,1,1))
  m=binary_metrics(3,1,2,4)
  self.assertAlmostEqual(m["precision"],.75)
  self.assertAlmostEqual(m["recall"],.6)
  self.assertAlmostEqual(m["omission_error"],.4)
  self.assertAlmostEqual(m["commission_error"],.25)
  self.assertAlmostEqual(m["false_positive_rate"],.2)
  self.assertAlmostEqual(m["accuracy"],.7)
  self.assertAlmostEqual(m["miou"],(3/6+4/7)/2)
 def test_undefined_metrics_are_null_json(self):
  m=binary_metrics(0,0,0,10)
  self.assertTrue(math.isnan(m["precision"]))
  self.assertEqual(m["accuracy"],1)
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/"metrics.json";save_json(p,m)
   self.assertIsNone(json.loads(p.read_text())["recall"])
   self.assertNotIn("NaN",p.read_text())
 def test_grouping_pools_counts(self):
  rows=[dict(region="A",**binary_metrics(1,1,0,1)),dict(region="A",**binary_metrics(9,0,9,0))]
  m=group_metrics(rows,"region").iloc[0]
  self.assertAlmostEqual(m.water_iou,10/20)
 def test_actual_notebook_test_function_writes_consistent_metrics(self):
  p=ROOT/"experiments/croma_sen1floods11/UperNet_head/02C_croma_on_sen1floods11_test_UperNet.ipynb"
  cells=json.loads(p.read_text(encoding="utf-8"))["cells"]
  source=next("".join(c["source"]) for c in cells if "def test(" in "".join(c["source"]))
  ns=dict(torch=torch,Path=Path,np=np,json=json,math=math,TILE_BATCH_SIZE=8,CROP_SIZE=120,IGNORE_INDEX=255,binary_metrics=binary_metrics,confusion_counts=confusion_counts,save_per_chip_metrics=save_per_chip_metrics)
  exec(source,ns)
  ns["sliding_window_logits"]=lambda model,image,device,**kwargs:image
  image=torch.tensor([[[0.,0.],[1.,1.]],[[1.,1.],[0.,0.]]])
  target=torch.tensor([[1,0],[1,255]])
  loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(image[None],target[None]),batch_size=1)
  with tempfile.TemporaryDirectory() as td:
   m=ns["test"](torch.nn.Identity(),loader,torch.nn.CrossEntropyLoss(ignore_index=255),"cpu",save_dir=td)
   self.assertEqual([m[k] for k in ("TP","FP","FN","TN")],[1,1,1,0])
   self.assertTrue((Path(td)/"metrics_per_chip.csv").exists())
   self.assertEqual(json.loads((Path(td)/"metrics.json").read_text())["valid_pixels"],3)

if __name__=="__main__":unittest.main()
