"""
Realism Score (Phone Photo) — ComfyUI custom node
Scores a generated image on how much it looks like a real smartphone photo.

Outputs (all floats):
  final_score    — weighted composite, 0–1  (higher = more phone-like)
  realism_score  — CLIP semantic score, 0–1
  aesthetic_score— aesthetic predictor, 0–1 (requires aesthetic-predictor-v2-5)
  texture_score  — mid-sharpness preference, 0–1
  noise_score    — moderate noise preference, 0–1

Dependencies:
  pip install open_clip_torch
  pip install aesthetic-predictor-v2-5   ← optional but recommended

Installation:
  Place this file + __init__.py in:
    ComfyUI/custom_nodes/comfyui-realism-scorer/
"""

import torch
import numpy as np
from PIL import Image

# ── lazy-loaded globals (populated on first inference, not at import time) ──
_clip_model    = None
_clip_preprocess = None
_clip_tokenizer  = None
_aesthetic_model = None
_aesthetic_preprocessor = None
_device = None


def _get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def _load_clip():
    """Load CLIP ViT-L/14 on first use."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return
    import open_clip
    device = _get_device()
    print("[RealismScorer] Loading CLIP ViT-L/14 …")
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    _clip_model = _clip_model.to(device).eval()
    _clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
    print("[RealismScorer] CLIP loaded.")


def _load_aesthetic():
    """
    Load Aesthetic Predictor V2.5 on first use.
    Uses the correct API: convert_v2_5_from_siglip()
    Falls back gracefully if the package isn't installed.
    """
    global _aesthetic_model, _aesthetic_preprocessor
    if _aesthetic_model is not None:
        return True   # already loaded
    try:
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        device = _get_device()
        print("[RealismScorer] Loading Aesthetic Predictor V2.5 …")
        _aesthetic_model, _aesthetic_preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        _aesthetic_model = _aesthetic_model.to(torch.bfloat16).to(device).eval()
        print("[RealismScorer] Aesthetic Predictor loaded.")
        return True
    except Exception as e:
        print(f"[RealismScorer] Aesthetic Predictor not available ({e}). "
              "Install with: pip install aesthetic-predictor-v2-5")
        _aesthetic_model = "unavailable"   # sentinel so we don't retry
        return False


# ── prompts ────────────────────────────────────────────────────────────────

POS_PROMPTS = [
    "a casual smartphone photo",
    "an amateur phone picture",
    "a natural unedited photo",
    "a real candid photograph",
    "a photo taken on an iPhone",
]

NEG_PROMPTS = [
    "AI generated image",
    "CGI render",
    "digital art",
    "studio portrait with lighting",
    "artificially generated face",
]


# ── helpers (numpy-only, no cv2) ────────────────────────────────────────────

def _to_gray(image_np: np.ndarray) -> np.ndarray:
    """RGB uint8 → float32 grayscale via standard luminance weights."""
    return (
        0.2989 * image_np[:, :, 0].astype(np.float32)
        + 0.5870 * image_np[:, :, 1].astype(np.float32)
        + 0.1140 * image_np[:, :, 2].astype(np.float32)
    )


def _laplacian_variance(image_np: np.ndarray) -> float:
    """Sharpness proxy via discrete Laplacian variance (no cv2 needed)."""
    gray = _to_gray(image_np)
    # Simple 3×3 Laplacian kernel convolution via numpy slicing
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:,  1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(lap))


def _estimate_noise(image_np: np.ndarray) -> float:
    """
    High-frequency noise estimate: subtract a 3×3 mean-blurred version
    and take the std of the residual — more accurate than raw std.
    """
    gray = _to_gray(image_np)
    # 3×3 box blur via slicing average
    blurred = (
        gray[:-2, :-2] + gray[:-2, 1:-1] + gray[:-2, 2:]
        + gray[1:-1, :-2] + gray[1:-1, 1:-1] + gray[1:-1, 2:]
        + gray[2:,  :-2] + gray[2:,  1:-1] + gray[2:,  2:]
    ) / 9.0
    residual = gray[1:-1, 1:-1] - blurred
    return float(np.std(residual))


def _gaussian_preference(x: float, target: float, sigma: float) -> float:
    """Bell curve centered at `target`. Returns 0–1."""
    return float(np.exp(-((x - target) ** 2) / (2 * sigma ** 2)))


def _clip_realism(image_pil: Image.Image) -> float:
    """
    CLIP cosine similarity: positive prompts minus negative prompts,
    remapped from roughly [-0.3, 0.3] → [0, 1].
    """
    _load_clip()
    device = _get_device()

    image_tensor = _clip_preprocess(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        img_feat = _clip_model.encode_image(image_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        pos_tokens = _clip_tokenizer(POS_PROMPTS).to(device)
        pos_feat   = _clip_model.encode_text(pos_tokens)
        pos_feat   = pos_feat / pos_feat.norm(dim=-1, keepdim=True)
        pos_sim    = (img_feat @ pos_feat.T).mean().item()

        neg_tokens = _clip_tokenizer(NEG_PROMPTS).to(device)
        neg_feat   = _clip_model.encode_text(neg_tokens)
        neg_feat   = neg_feat / neg_feat.norm(dim=-1, keepdim=True)
        neg_sim    = (img_feat @ neg_feat.T).mean().item()

    raw = pos_sim - neg_sim   # typically in [-0.3, 0.3]

    # Remap to [0, 1]: assume practical range is [-0.25, 0.25]
    remapped = (raw + 0.25) / 0.50
    return float(np.clip(remapped, 0.0, 1.0))


def _aesthetic_score(image_pil: Image.Image) -> float:
    """
    Returns aesthetic score normalised to 0–1.
    Falls back to 0.5 (neutral) if the model isn't installed.
    """
    available = _load_aesthetic()
    if not available or _aesthetic_model == "unavailable":
        return 0.5   # neutral fallback — doesn't skew final score

    device = _get_device()
    pixel_values = (
        _aesthetic_preprocessor(images=image_pil, return_tensors="pt")
        .pixel_values
        .to(torch.bfloat16)
        .to(device)
    )
    with torch.inference_mode():
        score = _aesthetic_model(pixel_values).logits.squeeze().float().cpu().item()

    # Model outputs roughly 1–10; normalise to 0–1
    return float(np.clip((score - 1.0) / 9.0, 0.0, 1.0))


# ── ComfyUI node ─────────────────────────────────────────────────────────────

class RealismScoreNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES  = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES  = (
        "final_score",
        "realism_score",
        "aesthetic_score",
        "texture_score",
        "noise_score",
    )
    FUNCTION  = "score"
    CATEGORY  = "image/analysis"

    def score(self, image: torch.Tensor):
        # ComfyUI tensor: (B, H, W, C) float32 0-1
        image_np  = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np, mode="RGB")

        # 1. CLIP semantic realism (0–1)
        realism = _clip_realism(image_pil)

        # 2. Aesthetic Predictor V2.5 (0–1)
        aesthetic = _aesthetic_score(image_pil)

        # 3. Texture / sharpness — Gaussian centred at target laplacian variance
        #    ~150 = pleasantly sharp phone photo; too low = blurry, too high = CGI
        sharpness = _laplacian_variance(image_np)
        texture   = _gaussian_preference(sharpness, target=150.0, sigma=80.0)

        # 4. Noise — Gaussian centred at target residual std
        #    ~4–6 = mild sensor noise; 0 = AI-smooth, >15 = over-noisy
        noise_std   = _estimate_noise(image_np)
        noise_score = _gaussian_preference(noise_std, target=5.0, sigma=3.0)

        # Weighted composite — all inputs are now guaranteed 0–1
        final = (
            0.45 * realism
            + 0.35 * aesthetic
            + 0.10 * texture
            + 0.10 * noise_score
        )

        return (
            float(np.clip(final,        0.0, 1.0)),
            float(np.clip(realism,      0.0, 1.0)),
            float(np.clip(aesthetic,    0.0, 1.0)),
            float(np.clip(texture,      0.0, 1.0)),
            float(np.clip(noise_score,  0.0, 1.0)),
        )


# ── registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RealismScoreNode": RealismScoreNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RealismScoreNode": "📷 Realism Score (Phone Photo)",
}
