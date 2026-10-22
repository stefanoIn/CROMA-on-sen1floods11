import ast
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from croma_upernet_pangaea import (
    UPerNetHead, CROMAUPerNet, prepare_sample, PREPROCESSING,
    latest_completed_checkpoint, SCHEMA, Sen1Floods11Dataset,
    random_labeled_crop, filter_labeled_training_pairs,
)
from use_croma import ViT, get_2dalibi

REFERENCE = ROOT / 'experiments/croma_sen1floods11/UperNet_head/pangaea_reference'


def upstream_classes():
    source = (REFERENCE / 'pangaea__decoders__upernet.py').read_text()
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name in {'SegUPerNet', 'PPM', 'Feature2Pyramid'}]
    class Decoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
    namespace = dict(torch=torch, nn=nn, F=F, Decoder=Decoder, Encoder=nn.Module)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<pinned upstream>', 'exec'), namespace)
    return namespace


class TinyCROMA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_dim = 32
        self.num_patches = 16
        self.attn_bias = get_2dalibi(16, 16)
        self.s1_encoder = ViT(dim=32, depth=12, in_channels=2)


class PangaeaProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)


    def test_crop_rejects_ignored_and_preserves_land_supervision(self):
        image = np.arange(2*16*16,dtype=np.float32).reshape(2,16,16)
        mask = np.full((16,16),255,dtype=np.int64)
        with self.assertRaises(ValueError):
            random_labeled_crop(image,mask,8)
        mask[15,15] = 0
        torch.manual_seed(234)
        for _ in range(20):
            crop,label = random_labeled_crop(image,mask,8,max_attempts=0)
            self.assertEqual(crop.shape,(2,8,8))
            self.assertEqual(int((label==0).sum()),1)
            self.assertEqual(crop[0][label==0].item(),image[0,15,15])
        dry = np.zeros((16,16),dtype=np.int64)
        _, label = random_labeled_crop(image,dry,8)
        self.assertTrue((label==0).all())

    def test_training_filter_keeps_valid_dry_images(self):
        import rasterio
        from rasterio.transform import from_origin
        with tempfile.TemporaryDirectory() as td:
            pairs = []
            for name,value in [('ignored',-1),('dry',0),('water',1)]:
                path=Path(td)/(name+'.tif')
                with rasterio.open(path,'w',driver='GTiff',height=2,width=2,count=1,
                                   dtype='int16',transform=from_origin(100,100,10,10)) as dst:
                    dst.write(np.full((2,2),value,dtype='int16'),1)
                pairs.append((path,path))
            kept,excluded=filter_labeled_training_pairs(pairs)
            self.assertEqual(kept,pairs[1:])
            self.assertEqual(excluded,pairs[:1])

    def test_decoder_matches_upstream_eval_and_training(self):
        upstream = upstream_classes()
        encoder = nn.Module()
        encoder.output_layers = [3, 5, 7, 11]
        encoder.output_dim = [32] * 4
        encoder.pyramid_output = False
        reference = upstream['SegUPerNet'](encoder, 2, False, channels=16)
        local = UPerNetHead(in_channels=32, channels=16)
        local.load_state_dict(reference.state_dict(), strict=True)
        features = [torch.randn(2, 32, 8, 8) for _ in range(4)]
        for training in [False, True]:
            reference.train(training); local.train(training)
            reference.dropout.p = local.dropout.p = 0
            expected = reference.conv_seg(reference._forward_feature(reference.neck(features)))
            actual = local(features)
            torch.testing.assert_close(actual, expected)
            if training:
                actual.sum().backward()
                self.assertTrue(all(p.grad is not None for p in local.parameters()))
        self.assertEqual([x.shape[-1] for x in local.neck(features)], [32, 16, 8, 4])

    def test_intermediate_features_match_upstream_transformer(self):
        croma = TinyCROMA()
        model = CROMAUPerNet(croma, UPerNetHead(in_channels=32, channels=16))
        source = (REFERENCE / 'pangaea__encoders__croma_encoder.py').read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'BaseTransformer')
        forward = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        namespace = {}
        exec(compile(ast.Module(body=[forward], type_ignores=[]), '<upstream features>', 'exec'), namespace)
        image = torch.randn(2, 2, 32, 32)
        from einops import rearrange
        x = croma.s1_encoder.linear_input(rearrange(image, 'b c (h i) (w j) -> b (h w) (c i j)', i=8, j=8))
        expected = namespace['forward'](croma.s1_encoder.transformer, x, croma.attn_bias, [3,5,7,11])
        actual = model.extract_features(image)
        for a,b in zip(actual, expected):
            torch.testing.assert_close(a.flatten(2).transpose(1,2), b)
        model.train()
        self.assertFalse(croma.training)
        model(image).sum().backward()
        self.assertTrue(all(p.grad is None for p in croma.parameters()))

    def test_raw_loading_and_fixed_normalization(self):
        import rasterio
        from rasterio.transform import from_origin
        with tempfile.TemporaryDirectory() as td:
            x = Path(td) / 'x.tif'; y = Path(td) / 'y.tif'
            raw = np.array([[[-60, 5], [np.nan, -10]], [[-80, 8], [-20, -15]]], dtype='float32')
            labels = np.array([[0,1],[-1,1]], dtype='int16')
            profile = dict(driver='GTiff',height=2,width=2,transform=from_origin(100,100,10,10))
            with rasterio.open(x,'w',count=2,dtype='float32',**profile) as dst: dst.write(raw)
            with rasterio.open(y,'w',count=1,dtype='int16',**profile) as dst: dst.write(labels,1)
            dataset = Sen1Floods11Dataset([(x,y)],prepare_sample)
            image, mask = dataset[0]
            mean = torch.tensor(PREPROCESSING['mean'])[:,None,None]
            std = torch.tensor(PREPROCESSING['std'])[:,None,None]
            torch.testing.assert_close(image,(torch.from_numpy(np.nan_to_num(raw))-mean)/std)
            self.assertEqual(mask[1,0].item(),255)
            self.assertLess(image.min().item(),0)
            self.assertGreater(image.max().item(),1)

    def test_only_completed_new_schema_runs_are_selected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name,schema,epochs in [('old','croma_sar_upernet_v1',80),('partial',SCHEMA,79),('good',SCHEMA,80)]:
                run=root/('train_'+name); run.mkdir()
                (run/'model.pt').touch()
                (run/'run.json').write_text(json.dumps(dict(schema=schema,epochs=80,checkpoint='model.pt',started_at=name)))
                (run/'croma_training_history.json').write_text(json.dumps(dict(epochs_completed=epochs)))
            self.assertEqual(latest_completed_checkpoint(root),root/'train_good/model.pt')

    def test_run_cell_selects_miou_and_completes_history(self):
        import contextlib
        import io
        from datetime import datetime
        from uuid import uuid4
        path = ROOT / 'experiments/croma_sen1floods11/UperNet_head/01C_croma_on_sen1floods11_training_UperNet.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        source = next(''.join(c['source']) for c in notebook['cells']
                      if ''.join(c['source']).startswith('model_name ='))
        head = nn.Conv2d(2,2,1)
        class Sampler:
            def set_epoch(self, epoch): pass
        loader = type('Loader', (), {'sampler': Sampler()})()
        values = iter([dict(miou=0.80,water_iou=0.60,val_loss=0.2),
                       dict(miou=0.79,water_iou=0.70,val_loss=0.1)])
        with tempfile.TemporaryDirectory() as td:
            from segmentation_metrics import METRIC_KEYS, clean_json, metric_table
            import pandas as pd
            namespace = dict(torch=torch, Path=Path, datetime=datetime, uuid4=uuid4, json=json,
                METRIC_KEYS=METRIC_KEYS,clean_json=clean_json,metric_table=metric_table,pd=pd,display=lambda *args:None,
                EXPERIMENT_NAME=td, NOTEBOOK_PATH=str(path), SCHEMA=SCHEMA, excluded_unlabeled_train=[], excluded_missing_sar_train=[],
                train_input_audit=pd.DataFrame(),val_input_audit=pd.DataFrame(),
                PANGAEA_REVISION='test', pretrained_path='pretrained/CROMA_large.pt',
                HEAD_CONFIG={}, PREPROCESSING=PREPROCESSING, CROP_SIZE=120, EPOCHS=2,
                EVAL_INTERVAL=1, SEED=234, BATCH_SIZE=8, IGNORE_INDEX=255, LR=1e-4,
                WEIGHT_DECAY=0.05, LR_MILESTONES=[1,2], model=type('Model',(),{'head':head})(),
                optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4), criterion=None,
                train_loader=loader, valid_loader=None, device='cpu',
                train_one_epoch=lambda *args: 0.3, validate=lambda *args: next(values),
                metrics=dict(miou=0.4,water_iou=0.,val_loss=0.5))
            with contextlib.redirect_stdout(io.StringIO()): exec(source,namespace)
            selected=latest_completed_checkpoint(td)
            checkpoint=torch.load(selected,map_location='cpu',weights_only=True)
            self.assertEqual(checkpoint['epoch'],1)
            self.assertEqual(checkpoint['best_val_miou'],0.8)
            history=json.loads(namespace['history_path'].read_text())
            self.assertEqual(history['epochs_completed'],2)
            self.assertEqual(history['history']['val_water_iou'],[0.6,0.7])

    def test_notebook_grid_and_scheduler(self):
        p=ROOT/'experiments/croma_sen1floods11/UperNet_head/01C_croma_on_sen1floods11_training_UperNet.ipynb'
        n=json.loads(p.read_text(encoding='utf-8'))
        source=next(''.join(c['source']) for c in n['cells'] if 'def sliding_window_logits(' in ''.join(c['source']))
        ns=dict(torch=torch,TILE_BATCH_SIZE=8)
        exec(source,ns)
        class IdentityLogits(nn.Module):
            def __init__(self): super().__init__(); self.tiles=0
            def forward(self,x): self.tiles+=x.shape[0]; return x
        model=IdentityLogits(); image=torch.randn(2,512,512)
        result=ns['sliding_window_logits'](model,image,'cpu')
        torch.testing.assert_close(result,image)
        self.assertEqual(model.tiles,25)
        model=IdentityLogits(); image=torch.randn(2,241,363)
        torch.testing.assert_close(ns['sliding_window_logits'](model,image,'cpu'),image)
        head=nn.Linear(2,2)
        ns=dict(torch=torch,nn=nn,model=type('Model',(),{'head':head})(),IGNORE_INDEX=255,train_loader=[None]*31)
        source=next(''.join(c['source']) for c in n['cells'] if ''.join(c['source']).startswith('EPOCHS ='))
        exec(source,ns)
        optimizer=ns['optimizer']; scheduler=ns['scheduler']
        for step in range(1,2481):
            optimizer.step(); scheduler.step()
            expected=1e-4*(0.1**sum(step>=m for m in [1488,2232]))
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'],expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
