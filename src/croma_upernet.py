from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


# UPerNet: pyramid pooling + top-down FPN + concatenated multiscale fusion.
# Architecture references (implementation uses only PyTorch):
# https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/uper_head.py
# https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/necks/featurepyramid.py
SCHEMA = 'croma_sar_upernet_v1'
PREPROCESSING = 'helper.load_image_pair+prepare_sample:clip_-50_1_then_per_image_mean_2std_float'


def conv_block(in_channels, channels, kernel=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, channels, kernel, padding=kernel // 2, bias=False),
        nn.GroupNorm(8, channels),
        nn.ReLU(inplace=False),
    )


class UPerNetHead(nn.Module):
    def __init__(self, in_channels=1024, channels=128, num_classes=2,
                 out_indices=(2, 5, 8, 11), pool_scales=(1, 2, 3, 6)):
        super().__init__()
        if channels < 16 or channels % 8:
            raise ValueError('channels must be a multiple of 8 and at least 16.')
        if len(out_indices) != 4 or tuple(sorted(set(out_indices))) != tuple(out_indices):
            raise ValueError('Select four distinct transformer blocks in increasing order.')
        self.config = dict(in_channels=in_channels, channels=channels, num_classes=num_classes,
                           out_indices=list(out_indices), pool_scales=list(pool_scales))
        self.out_indices = tuple(out_indices)
        self.norms = nn.ModuleList([nn.LayerNorm(in_channels) for _ in range(4)])
        # Project before spatial expansion to keep GPU memory use manageable.
        self.projections = nn.ModuleList([conv_block(in_channels, channels, 1) for _ in range(4)])
        self.resamplers = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(channels, channels, 2, 2),
                          nn.GroupNorm(8, channels), nn.GELU(),
                          nn.ConvTranspose2d(channels, channels, 2, 2)),
            nn.ConvTranspose2d(channels, channels, 2, 2),
            nn.Identity(), nn.MaxPool2d(2, 2),
        ])
        # GroupNorm also works with batch size one and the 1x1 pooling branch.
        self.ppm = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(scale), conv_block(channels, channels, 1))
            for scale in pool_scales
        ])
        self.ppm_fuse = conv_block(channels * (len(pool_scales) + 1), channels)
        self.laterals = nn.ModuleList([conv_block(channels, channels, 1) for _ in range(3)])
        self.smooth = nn.ModuleList([conv_block(channels, channels) for _ in range(3)])
        self.fuse = conv_block(4 * channels, channels)
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def make_pyramid(self, features):
        if len(features) != 4:
            raise ValueError('UPerNet requires four intermediate feature maps.')
        pyramid = []
        for feature, norm, project, resample in zip(features, self.norms, self.projections, self.resamplers):
            feature = norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            pyramid.append(resample(project(feature)))
        return pyramid

    def forward(self, features):
        pyramid = self.make_pyramid(features)
        deepest = pyramid[-1]
        pooled = [deepest] + [F.interpolate(branch(deepest), size=deepest.shape[-2:],
                                            mode='bilinear', align_corners=False) for branch in self.ppm]
        laterals = [layer(feature) for layer, feature in zip(self.laterals, pyramid[:-1])]
        laterals.append(self.ppm_fuse(torch.cat(pooled, dim=1)))
        for i in range(3, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='bilinear', align_corners=False)
        outputs = [layer(feature) for layer, feature in zip(self.smooth, laterals[:-1])] + [laterals[-1]]
        outputs = [F.interpolate(x, size=outputs[0].shape[-2:], mode='bilinear', align_corners=False)
                   for x in outputs]
        return self.classifier(self.dropout(self.fuse(torch.cat(outputs, dim=1))))


class CROMAUPerNet(nn.Module):
    def __init__(self, croma, head):
        super().__init__()
        self.croma = croma.requires_grad_(False)
        self.head = head
        self.croma.eval()
        if max(head.out_indices) >= len(croma.s1_encoder.transformer.layers):
            raise ValueError('Requested blocks exceed the SAR encoder depth.')
        if head.config['in_channels'] != croma.encoder_dim:
            raise ValueError('Decoder channel count does not match CROMA.')

    def train(self, mode=True):
        super().train(mode)
        self.croma.eval()
        return self

    @torch.no_grad()
    def extract_features(self, images):
        encoder = self.croma.s1_encoder
        p = encoder.patch_size
        h, w = images.shape[-2:]
        if h != w or h % p or (h // p) * (w // p) != self.croma.num_patches:
            raise ValueError('Input tile must match the square CROMA image_resolution.')
        x = rearrange(images, 'b c (h i) (w j) -> b (h w) (c i j)', i=p, j=p)
        x = encoder.linear_input(x)
        bias = self.croma.attn_bias.to(device=x.device, dtype=x.dtype)
        features = []
        # Capture the full residual stream AFTER attention and FFN, not FFN outputs alone.
        # Each selected block has its own trainable LayerNorm in the decoder.
        for index, (attention, ffn) in enumerate(encoder.transformer.layers):
            x = x + attention(x, bias)
            x = x + ffn(x)
            if index in self.head.out_indices:
                features.append(x.transpose(1, 2).reshape(x.shape[0], x.shape[2], h // p, w // p))
            if index == self.head.out_indices[-1]:
                break
        return features

    def forward(self, images):
        logits = self.head(self.extract_features(images))
        return F.interpolate(logits, size=images.shape[-2:], mode='bilinear', align_corners=False)


def latest_completed_checkpoint(run_root):
    candidates = []
    for manifest in Path(run_root).glob('train_*/run.json'):
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        if metadata.get('schema') != SCHEMA:
            continue
        history_path = manifest.parent / 'croma_training_history.json'
        if not history_path.is_file():
            continue
        history = json.loads(history_path.read_text(encoding='utf-8'))
        checkpoint = manifest.parent / Path(metadata['checkpoint']).name
        if history.get('epochs_completed') == metadata.get('epochs') and checkpoint.is_file():
            candidates.append((metadata['started_at'], checkpoint))
    if not candidates:
        raise FileNotFoundError('No completed UPerNet run found. Run the training notebook first.')
    return max(candidates, key=lambda item: item[0])[1]
