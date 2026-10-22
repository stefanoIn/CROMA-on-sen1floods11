import sys,tempfile,unittest
from pathlib import Path
import numpy as np,rasterio
from rasterio.transform import from_origin
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from sar_input_quality import audit_sar_pairs,report_input_quality

class InputQualityTests(unittest.TestCase):
 def test_missing_partial_and_finite_zero_inputs(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);pairs=[]
   profile=dict(driver="GTiff",height=2,width=2,transform=from_origin(100,100,10,10))
   for name,raw in [("missing",np.full((2,2,2),np.nan)),("zero",np.zeros((2,2,2))),("partial",np.array([[[np.nan,0],[0,0]],[[0,0],[0,0]]]))]:
    x=root/(name+"_S1Hand.tif");y=root/(name+"_LabelHand.tif")
    with rasterio.open(x,"w",count=2,dtype="float32",nodata=np.nan,**profile) as s:s.write(raw.astype("float32"))
    with rasterio.open(y,"w",count=1,dtype="int16",**profile) as s:s.write(np.ones((2,2),dtype="int16"),1)
    with rasterio.open(root/(x.stem+"_prediction.tif"),"w",count=1,dtype="uint8",**profile) as s:s.write(np.zeros((2,2),dtype="uint8"),1)
    pairs.append((x,y))
   audit=audit_sar_pairs(pairs)
   self.assertEqual(audit.exclude_missing_sar.tolist(),[True,False,False])
   self.assertEqual(audit.sar_usable_pixels.tolist(),[0,4,3])
   m=report_input_quality(pairs,root)
   self.assertEqual(m["excluded_missing_sar_images"],1)
   self.assertEqual(m["valid_pixels"],8)
   self.assertEqual(m["FN"],8)
   self.assertTrue((root/"excluded_missing_sar.csv").exists())
if __name__=="__main__":unittest.main()
