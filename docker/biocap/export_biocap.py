#!/usr/bin/env python3
"""Export imageomics/biocap (ViT-B/16) vision and text encoders to FP32 ONNX."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

EXPORT_IMAGE = "sha256:fb47fdf2a242ea97c5b0ae0cebc18d44fc53ab0e7bf2055cb121c00b559ae3b4"

INNER_SCRIPT = r"""
import json, os, shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from PIL import Image

out = Path('/out')
out.mkdir(parents=True, exist_ok=True)

model, _, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/biocap')
model = model.eval().cpu().float()
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/biocap')

class VisionWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, image):
        return F.normalize(self.m.encode_image(image), dim=-1)

class TextWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, text):
        return F.normalize(self.m.encode_text(text), dim=-1)

vw = VisionWrapper(model).eval()
tw = TextWrapper(model).eval()

dummy_img = torch.randn(1, 3, 224, 224, dtype=torch.float32)
dummy_tok = tokenizer(["an eastern fox squirrel"])

with torch.no_grad():
    torch.onnx.export(
        vw, (dummy_img,), str(out / 'biocap-vision.onnx'),
        input_names=['image'], output_names=['embedding'],
        opset_version=17, dynamo=False, do_constant_folding=True,
    )
    torch.onnx.export(
        tw, (dummy_tok,), str(out / 'biocap-text.onnx'),
        input_names=['text'], output_names=['embedding'],
        opset_version=17, dynamo=False, do_constant_folding=True,
    )

# Copy BPE vocabulary used by OpenCLIP SimpleTokenizer
bpe_src = Path(open_clip.__file__).resolve().parent / 'bpe_simple_vocab_16e6.txt.gz'
shutil.copyfile(bpe_src, out / 'bpe_simple_vocab_16e6.txt.gz')

# Extract exact preprocessing mean/std/size from preprocess_val
mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]
for t in getattr(preprocess_val, 'transforms', []):
    if hasattr(t, 'mean') and hasattr(t, 'std'):
        mean = [float(x) for x in t.mean]
        std = [float(x) for x in t.std]
(out / 'preprocess.json').write_text(json.dumps({
    'model': 'imageomics/biocap',
    'image_size': 224,
    'context_length': 77,
    'embedding_dim': 512,
    'mean': mean,
    'std': std,
}, indent=2) + '\n')

# Save PyTorch reference embeddings for parity validation
sample_img = Image.open('/sample/sample.jpg').convert('RGB')
img_t = preprocess_val(sample_img).unsqueeze(0)
texts = ["an eastern fox squirrel", "a gray fox", "a raccoon in the backyard", "empty yard"]
toks = tokenizer(texts)
with torch.no_grad():
    ref_img = vw(img_t).numpy()
    ref_txt = tw(toks).numpy()
np.savez(
    out / 'ref_check.npz',
    img_tensor=img_t.numpy(),
    tokens=toks.numpy(),
    ref_img=ref_img,
    ref_txt=ref_txt,
    texts=np.array(texts),
)
print('Exported BioCAP ONNX models + tokenizer vocab + reference parity vectors')
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/biocap"))
    p.add_argument(
        "--sample-image",
        type=Path,
        default=Path(
            "artifacts/megadetector-extra/captures/1794926438.280011-c3jpp7.jpg"
        ),
    )
    p.add_argument("--export-image", default=EXPORT_IMAGE)
    args = p.parse_args()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sample = args.sample_image.resolve(strict=True)

    with tempfile.TemporaryDirectory(prefix="biocap-export-") as td:
        tmp = Path(td)
        (tmp / "inner.py").write_text(INNER_SCRIPT)
        deps = tmp / "deps"
        deps.mkdir()
        hf = tmp / "hf"
        hf.mkdir()
        sample_dir = tmp / "sample"
        sample_dir.mkdir()
        shutil.copyfile(sample, sample_dir / "sample.jpg")

        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-e",
                "HOME=/tmp",
                "-e",
                "HF_HOME=/work/hf",
                "-v",
                f"{tmp}:/work",
                "-v",
                f"{out}:/out",
                "-v",
                f"{sample_dir}:/sample:ro",
                "--entrypoint",
                "bash",
                args.export_image,
                "-c",
                "python3 -m pip install --quiet --target /work/deps open_clip_torch onnx onnxscript && "
                "PYTHONPATH=/work/deps python3 /work/inner.py",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
