"""
Frozen embedding helpers for V1 (Semantic-Temporal Target Memory).

  * CLIP text encoder  -> semantic field (expression embedding)
  * DINOv2 image encoder -> appearance field (masked / boxed crop embedding)

Both are frozen, no training. Outputs are L2-normalised float32 numpy vectors,
so cosine similarity is just a dot product.
"""
from __future__ import annotations

import numpy as np

CLIP_ID = "openai/clip-vit-base-patch32"
DINO_ID = "facebook/dinov2-small"


def _as_tensor(out, *names):
    """transformers versions differ: some return a Tensor, some an output object."""
    import torch
    if isinstance(out, torch.Tensor):
        return out
    for n in names:
        v = getattr(out, n, None)
        if v is not None:
            return v
    raise TypeError(f"cannot get a tensor from {type(out)}")


def _l2(x):
    import torch
    return torch.nn.functional.normalize(x.float(), dim=-1)


class TextEmbedder:
    def __init__(self, device: str = "cuda", model_id: str = CLIP_ID):
        from transformers import CLIPModel, CLIPTokenizer
        self.device = device
        self.tok = CLIPTokenizer.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()

    def __call__(self, texts: list[str]) -> np.ndarray:
        import torch
        with torch.no_grad():
            enc = self.tok(texts, padding=True, truncation=True, max_length=77,
                           return_tensors="pt").to(self.device)
            out = self.model.get_text_features(**enc)
            feats = _as_tensor(out, "text_embeds", "pooler_output")
            return _l2(feats).cpu().numpy().astype(np.float32)

    def free(self):
        import torch
        del self.model
        torch.cuda.empty_cache()


class CropEmbedder:
    """DINOv2 CLS embedding of an RGB crop. `image` is a PIL.Image."""

    def __init__(self, device: str = "cuda", model_id: str = DINO_ID):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    def embed_images(self, images: list) -> np.ndarray:
        import torch
        with torch.no_grad():
            inp = self.proc(images=images, return_tensors="pt").to(self.device)
            out = self.model(**inp)
            cls = out.last_hidden_state[:, 0]
            return _l2(cls).cpu().numpy().astype(np.float32)

    def embed_box(self, image, box) -> np.ndarray:
        """Embed the crop of `image` inside box=[x1,y1,x2,y2(,score)]."""
        w, h = image.size
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, max(int(x2), x1 + 2)), min(h, max(int(y2), y1 + 2))
        return self.embed_images([image.crop((x1, y1, x2, y2))])[0]

    def embed_masked(self, image, mask: np.ndarray, pad: int = 8):
        """Embed the tight crop around a boolean mask with background greyed out.
        Returns None if the mask is empty."""
        from PIL import Image
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None
        h, w = mask.shape
        y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
        x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
        arr = np.array(image.convert("RGB"))
        grey = np.full_like(arr, 124)
        arr = np.where(mask[..., None], arr, grey)
        crop = Image.fromarray(arr[y1:y2, x1:x2])
        return self.embed_images([crop])[0]

    def free(self):
        import torch
        del self.model
        torch.cuda.empty_cache()
