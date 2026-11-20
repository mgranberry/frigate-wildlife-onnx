"""Lightweight TensorRT FP16 BioCAP embedding sidecar compatible with Frigate's llama.cpp GenAI client."""

import base64
import gzip
import html
import io
import json
import os
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import ftfy
import numpy as np
import onnxruntime as ort
import regex as re
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from PIL import Image

MODEL_DIR = Path(os.environ.get("BIOCAP_MODEL_DIR", "/models"))
MEDIA_MARKER = "<__media__>"
MODEL_ID = os.environ.get("BIOCAP_MODEL_ID", "imageomics/biocap")
PORT = int(os.environ.get("PORT", "8090"))


@lru_cache()
def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def _whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


class CLIPSimpleTokenizer:
    """Pure-Python OpenAI/OpenCLIP BPE tokenizer matching bpe_simple_vocab_16e6."""

    def __init__(self, bpe_path: Path, context_length: int = 77) -> None:
        self.byte_encoder = _bytes_to_unicode()
        merges_raw = gzip.open(bpe_path, "rt", encoding="utf-8").read().split("\n")
        merges_raw = merges_raw[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(m.split()) for m in merges_raw]
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<start_of_text>", "<end_of_text>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<start_of_text>": "<start_of_text>",
            "<end_of_text>": "<end_of_text>",
        }
        self.pat = re.compile(
            r"""<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.sot_token = self.encoder["<start_of_text>"]
        self.eot_token = self.encoder["<end_of_text>"]
        self.context_length = context_length

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        joined = " ".join(word)
        self.cache[token] = joined
        return joined

    def encode_one(self, text: str) -> np.ndarray:
        cleaned = _whitespace_clean(_basic_clean(text)).lower()
        bpe_tokens = []
        for token in re.findall(self.pat, cleaned):
            utf8_tok = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[b] for b in self.bpe(utf8_tok).split(" "))
        tokens = (
            [self.sot_token] + bpe_tokens[: self.context_length - 2] + [self.eot_token]
        )
        arr = np.zeros((1, self.context_length), dtype=np.int64)
        arr[0, : len(tokens)] = tokens
        return arr


class BioCAPEngine:
    def __init__(self, model_dir: Path) -> None:
        meta = json.loads((model_dir / "preprocess.json").read_text())
        self.image_size = int(meta["image_size"])
        self.mean = np.asarray(meta["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(meta["std"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.tokenizer = CLIPSimpleTokenizer(
            model_dir / "bpe_simple_vocab_16e6.txt.gz",
            context_length=int(meta["context_length"]),
        )
        trt_cache = model_dir / "trt-cache"
        trt_cache.mkdir(parents=True, exist_ok=True)
        use_fp16 = os.environ.get("BIOCAP_TRT_FP16", "true").lower() != "false"

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        providers: list[Any] = []
        if "TensorrtExecutionProvider" in ort.get_available_providers():
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": 0,
                        "trt_fp16_enable": use_fp16,
                        "trt_timing_cache_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_timing_cache_path": str(trt_cache),
                        "trt_engine_cache_path": str(trt_cache),
                    },
                )
            )
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append(("CUDAExecutionProvider", {"device_id": 0}))
        providers.append("CPUExecutionProvider")

        self.vision = ort.InferenceSession(
            str(model_dir / "biocap-vision.onnx"),
            sess_options=opts,
            providers=providers,
        )
        self.text = ort.InferenceSession(
            str(model_dir / "biocap-text.onnx"), sess_options=opts, providers=providers
        )
        self.vision_in = self.vision.get_inputs()[0].name
        self.text_in = self.text.get_inputs()[0].name
        self._lock = threading.Lock()

        # Warm up TensorRT engines on startup
        self.vision.run(
            None,
            {
                self.vision_in: np.zeros(
                    (1, 3, self.image_size, self.image_size), dtype=np.float32
                )
            },
        )
        self.text.run(
            None,
            {self.text_in: np.zeros((1, int(meta["context_length"])), dtype=np.int64)},
        )

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        size = self.image_size
        if w < h:
            nw, nh = size, int(size * h / w)
        else:
            nw, nh = int(size * w / h), size
        img = img.resize((nw, nh), Image.Resampling.BICUBIC)
        left = int(round((nw - size) / 2.0))
        top = int(round((nh - size) / 2.0))
        img = img.crop((left, top, left + size, top + size))
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray((arr - self.mean) / self.std, dtype=np.float32)

    def embed_image(self, image_bytes: bytes) -> list[float]:
        x = self.preprocess_image(image_bytes)
        with self._lock:
            out = self.vision.run(None, {self.vision_in: x})[0][0].astype(np.float32)
        norm = float(np.linalg.norm(out))
        if norm > 0:
            out = out / norm
        return out.tolist()

    def embed_text(self, text: str) -> list[float]:
        tok = self.tokenizer.encode_one(text)
        with self._lock:
            out = self.text.run(None, {self.text_in: tok})[0][0].astype(np.float32)
        norm = float(np.linalg.norm(out))
        if norm > 0:
            out = out / norm
        return out.tolist()


engine: BioCAPEngine | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine
    engine = BioCAPEngine(MODEL_DIR)
    yield


app = FastAPI(title="BioCAP Embeddings Sidecar", lifespan=lifespan)


def _get_engine() -> BioCAPEngine:
    if engine is None:
        raise HTTPException(status_code=503, detail="BioCAP engine not initialized")
    return engine


@app.get("/health")
def health() -> dict[str, Any]:
    eng = _get_engine()
    return {
        "status": "ok",
        "model": MODEL_ID,
        "vision_providers": eng.vision.get_providers(),
        "text_providers": eng.text.get_providers(),
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "data": [
            {
                "id": MODEL_ID,
                "architecture": {"input_modalities": ["text", "image"]},
                "meta": {"n_ctx": 77},
            }
        ]
    }


@app.get("/props")
def props() -> dict[str, Any]:
    return {
        "default_generation_settings": {"n_ctx": 77},
        "modalities": {"vision": True},
        "media_marker": MEDIA_MARKER,
    }


@app.post("/embeddings")
async def embeddings(request: Request) -> dict[str, Any]:
    eng = _get_engine()
    body = await request.json()
    content = body.get("content", [])
    results = []
    for item in content:
        if isinstance(item, dict) and "multimodal_data" in item:
            media = item["multimodal_data"]
            if media:
                for b64 in media:
                    vec = eng.embed_image(base64.b64decode(b64))
                    results.append({"embedding": vec})
            elif "prompt_string" in item and item["prompt_string"] != MEDIA_MARKER:
                vec = eng.embed_text(str(item["prompt_string"]))
                results.append({"embedding": vec})
        elif isinstance(item, dict) and "prompt_string" in item:
            vec = eng.embed_text(str(item["prompt_string"]))
            results.append({"embedding": vec})
        elif isinstance(item, str):
            vec = eng.embed_text(item)
            results.append({"embedding": vec})
    return {"data": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
