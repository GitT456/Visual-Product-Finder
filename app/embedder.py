"""Turns a PIL image into an embedding: ResNet50 with the classifier removed.

All settings come from `model_config.json` (written by the notebooks), so queries are processed
exactly like the catalog images were. PyTorch is imported here only, so the rest of the
application can be tested without it.
"""
from __future__ import annotations

import logging
import threading

import numpy as np
from PIL import Image

logger = logging.getLogger("vpf.embedder")


class Embedder:
    def __init__(self, config: dict, device: str = "auto"):
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        from torchvision.models import ResNet50_Weights, resnet50

        self._torch = torch
        pre = config["preprocessing"]

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # "ResNet50_Weights.IMAGENET1K_V2" -> ResNet50_Weights["IMAGENET1K_V2"]
        weights_name = str(config["weights"]).split(".")[-1]
        try:
            weights = ResNet50_Weights[weights_name]
        except KeyError as exc:
            raise RuntimeError(f"Unknown weights '{config['weights']}' in model_config.json.") from exc

        self._check_against_weights(weights, pre)

        model = resnet50(weights=weights)       # downloads the weights on the first run (cached afterwards)
        model.fc = nn.Identity()                # the 2048-number vector before the classifier is the embedding
        self.model = model.to(self.device).eval()

        self.preprocess = T.Compose([
            T.Resize(pre["resize_size"], interpolation=T.InterpolationMode(pre["interpolation"])),
            T.CenterCrop(pre["crop_size"]),
            T.ToTensor(),
            T.Normalize(mean=pre["mean"], std=pre["std"]),
        ])
        self.crop_size = pre["crop_size"]
        self.dim = int(config["embedding_dim"])
        # fp16 only on GPU, and only if the catalog was built with it
        self.use_amp = bool(config.get("mixed_precision_fp16", False)) and self.device.type == "cuda"
        self._lock = threading.Lock()           # one forward pass at a time

        logger.info("Embedder ready on %s (fp16: %s)", self.device, self.use_amp)

    @staticmethod
    def _check_against_weights(weights, pre: dict) -> None:
        """Warn if the saved preprocessing differs from what the pretrained weights document."""
        official = weights.transforms()
        same = (official.resize_size[0] == pre["resize_size"]
                and official.crop_size[0] == pre["crop_size"]
                and np.allclose(official.mean, pre["mean"])
                and np.allclose(official.std, pre["std"]))
        if not same:
            logger.warning("Preprocessing in model_config.json differs from the official one for %s. "
                           "model_config.json is used, because the catalog was built with it.", weights)

    @property
    def description(self) -> str:
        return f"ResNet50 on {self.device}" + (" (fp16)" if self.use_amp else " (fp32)")

    def embed(self, image: Image.Image) -> np.ndarray:
        """L2-normalized float32 vector of shape (dim,)."""
        torch = self._torch
        x = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self._lock, torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            features = self.model(x)
        vector = torch.nn.functional.normalize(features.float(), dim=1)[0].cpu().numpy()
        return vector.astype(np.float32)

    def warmup(self) -> None:
        """One dummy pass, so that the first real request is not slow."""
        self.embed(Image.new("RGB", (self.crop_size * 2, self.crop_size * 2), (255, 255, 255)))
