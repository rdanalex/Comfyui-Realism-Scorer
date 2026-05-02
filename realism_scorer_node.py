"""
ComfyUI Node: Realism Scorer
Scores generated images on photorealism and "phone-taken" quality (0–100).
Higher = more realistic, closer to a real smartphone photo.

Installation:
  Place this file in:  ComfyUI/custom_nodes/realism_scorer/realism_scorer_node.py
  Also create:         ComfyUI/custom_nodes/realism_scorer/__init__.py  (see bottom of file)

Dependencies (auto-installed on first run if missing):
  torch, torchvision, transformers, Pillow, numpy, scipy
"""

import torch
import numpy as np
from PIL import Image, ImageStat, ImageFilter
import io
import math


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI image tensor (B,H,W,C float32 0-1) to PIL."""
    if tensor.ndim == 4:
        tensor = tensor[0]                          # take first batch item
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Sub-scorers  (each returns a float 0-1, higher = more phone-realistic)
# ─────────────────────────────────────────────────────────────────────────────

def score_noise_grain(img: Image.Image) -> float:
    """
    Real phone photos have mild sensor noise / grain.
    Perfectly smooth AI images score low; noisy ones score higher (up to a ceiling).
    """
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)

    # High-pass filter to isolate noise
    blurred = np.array(gray.filter(ImageFilter.GaussianBlur(radius=1)), dtype=np.float32)
    noise = arr - blurred
    std = float(np.std(noise))

    # Typical phone noise σ ≈ 3-10; AI-smooth ≈ 0-1; over-noisy ≈ 20+
    ideal_low, ideal_high = 2.5, 9.0
    if std < ideal_low:
        return std / ideal_low * 0.6          # penalise too-smooth
    elif std <= ideal_high:
        return 0.6 + (std - ideal_low) / (ideal_high - ideal_low) * 0.4
    else:
        # Over-noisy – fall off gently
        return max(0.0, 1.0 - (std - ideal_high) / 30.0)


def score_sharpness_with_bokeh(img: Image.Image) -> float:
    """
    Phone photos have a sharp subject with possible edge falloff / mild lens blur.
    AI images are often uniformly crisp or uniformly soft.
    We reward spatial variance in sharpness across regions.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape
    block = 64
    laplacians = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            patch = gray[y:y+block, x:x+block]
            lap = float(np.var(np.gradient(patch)[0]))
            laplacians.append(lap)

    if not laplacians:
        return 0.5

    lap_arr = np.array(laplacians)
    mean_sharpness = float(np.mean(lap_arr))
    spatial_variance = float(np.std(lap_arr))   # variance across regions

    # Normalise mean sharpness – typical phone image ≈ 50-500
    sharpness_score = min(1.0, mean_sharpness / 300.0)

    # Reward spatial variance (subject sharp, background soft)
    variance_score = min(1.0, spatial_variance / (mean_sharpness + 1e-6))

    return 0.5 * sharpness_score + 0.5 * min(1.0, variance_score * 2)


def score_color_naturalism(img: Image.Image) -> float:
    """
    Phone photos have natural colour distributions.
    Oversaturated / HDR-toned / monochromatic AI images score lower.
    """
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # Saturation via HSV
    max_c = arr.max(axis=2)
    min_c = arr.min(axis=2)
    sat = np.where(max_c > 1e-5, (max_c - min_c) / (max_c + 1e-5), 0.0)
    mean_sat = float(np.mean(sat))

    # Typical phone: mean saturation ≈ 0.25-0.55
    # Penalty for very low (desaturated / B&W look) or very high (hyper-saturated)
    ideal_sat_low, ideal_sat_high = 0.20, 0.55
    if mean_sat < ideal_sat_low:
        sat_score = mean_sat / ideal_sat_low
    elif mean_sat <= ideal_sat_high:
        sat_score = 1.0
    else:
        sat_score = max(0.0, 1.0 - (mean_sat - ideal_sat_high) / 0.4)

    # Channel balance – natural scenes rarely have extreme single-channel dominance
    channel_means = [float(np.mean(c)) / 255.0 for c in [r, g, b]]
    channel_std = float(np.std(channel_means))
    balance_score = max(0.0, 1.0 - channel_std * 6)

    # Dynamic range check (avoid crushed blacks / blown highlights)
    p2, p98 = float(np.percentile(arr, 2)), float(np.percentile(arr, 98))
    dr_score = min(1.0, (p98 - p2) / 180.0)

    return (sat_score * 0.4 + balance_score * 0.3 + dr_score * 0.3)


def score_texture_complexity(img: Image.Image) -> float:
    """
    Real images have fractal-like texture complexity.
    Smooth AI skin / backgrounds score lower.
    Uses local binary pattern variance as a proxy.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)

    # Gradient magnitude as texture richness
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx**2 + gy**2)

    # We want moderate texture – not overly sharp edges (drawing/cartoon)
    # but not completely smooth (plastic AI skin)
    p25 = float(np.percentile(mag, 25))
    p75 = float(np.percentile(mag, 75))
    iqr = p75 - p25
    mean_mag = float(np.mean(mag))

    # Score IQR relative to mean (textured but not edgy)
    texture_score = min(1.0, iqr / max(mean_mag, 1.0))
    mean_score = min(1.0, mean_mag / 20.0)

    return 0.5 * texture_score + 0.5 * mean_score


def score_chromatic_aberration_and_vignette(img: Image.Image) -> float:
    """
    Phone lenses introduce very subtle chromatic aberration and vignetting.
    Pure absence = rendered/AI look. Slight presence = photographic.
    We detect slight R/B channel spatial offset and corner darkening.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Vignette detection: compare center brightness vs corners
    cy, cx = h // 2, w // 2
    r_crop = min(cy, cx) // 4
    center = arr[cy-r_crop:cy+r_crop, cx-r_crop:cx+r_crop]
    corner_size = r_crop

    corners = [
        arr[:corner_size, :corner_size],
        arr[:corner_size, -corner_size:],
        arr[-corner_size:, :corner_size],
        arr[-corner_size:, -corner_size:],
    ]
    center_lum = float(np.mean(center))
    corner_lum = float(np.mean([np.mean(c) for c in corners]))

    vignette_ratio = (center_lum - corner_lum) / (center_lum + 1e-5)
    # Ideal subtle vignette: 2-15% falloff
    if 0.02 <= vignette_ratio <= 0.15:
        vig_score = 1.0
    elif vignette_ratio < 0.02:
        vig_score = vignette_ratio / 0.02 * 0.7 + 0.3   # slight penalty for none
    else:
        vig_score = max(0.0, 1.0 - (vignette_ratio - 0.15) / 0.2)

    # Chromatic aberration: slight R-B edge offset
    r_chan = arr[:,:,0]
    b_chan = arr[:,:,2]
    r_edge = np.gradient(r_chan)[0]
    b_edge = np.gradient(b_chan)[0]
    ca_diff = float(np.mean(np.abs(r_edge - b_edge)))
    # Very slight CA (1-4 units) is phone-like; zero = rendered; high = bad
    if 0.5 <= ca_diff <= 4.0:
        ca_score = 1.0
    elif ca_diff < 0.5:
        ca_score = ca_diff / 0.5 * 0.6 + 0.4
    else:
        ca_score = max(0.3, 1.0 - (ca_diff - 4.0) / 10.0)

    return 0.5 * vig_score + 0.5 * ca_score


def score_compression_artifacts(img: Image.Image) -> float:
    """
    Real phone photos have JPEG compression artifacts (subtle 8×8 DCT blocks).
    Pure PNG-clean AI outputs score lower on this axis.
    We simulate it by comparing the image to a re-compressed version.
    """
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    orig_arr = np.array(img, dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)

    diff = np.abs(orig_arr - recomp_arr)
    mean_diff = float(np.mean(diff))

    # Very low diff = image already looks compressed (phone-like) → good
    # High diff = pristine AI render → penalise slightly
    # Ideal mean diff ≈ 1-5
    if mean_diff <= 3.0:
        return 1.0
    elif mean_diff <= 8.0:
        return 1.0 - (mean_diff - 3.0) / 8.0 * 0.4
    else:
        return max(0.3, 1.0 - (mean_diff - 8.0) / 20.0 * 0.7)


def score_depth_of_field_blur(img: Image.Image) -> float:
    """
    Phone portrait mode creates background blur. Even without portrait mode,
    lens physics creates some focus falloff. Detect blur gradient from center
    outward as a proxy.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape

    def region_sharpness(region):
        lap = np.gradient(region)
        return float(np.var(lap[0]) + np.var(lap[1]))

    # Divide into 3×3 grid, measure sharpness per cell
    rows, cols = 3, 3
    rh, rw = h // rows, w // cols
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            patch = gray[r*rh:(r+1)*rh, c*rw:(c+1)*rw]
            row.append(region_sharpness(patch))
        grid.append(row)

    grid = np.array(grid)
    center_sharp = grid[1, 1]
    edge_sharp = np.mean([grid[0,:], grid[2,:], grid[:,0], grid[:,2]])

    if center_sharp < 1e-3:
        return 0.4
    ratio = edge_sharp / (center_sharp + 1e-5)

    # Phone-like: center sharp, edges slightly softer → ratio 0.3-0.85
    if 0.25 <= ratio <= 0.85:
        return 1.0
    elif ratio < 0.25:
        # Too blurry everywhere
        return max(0.2, ratio / 0.25)
    else:
        # Uniformly sharp (CGI look)
        return max(0.3, 1.0 - (ratio - 0.85) / 0.6)


# ─────────────────────────────────────────────────────────────────────────────
# Main scorer
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "noise_grain":              0.20,
    "sharpness_bokeh":          0.18,
    "color_naturalism":         0.18,
    "texture_complexity":       0.16,
    "lens_optics":              0.14,   # CA + vignette
    "compression_artifacts":    0.07,
    "depth_of_field":           0.07,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1"


def compute_realism_score(img: Image.Image) -> tuple[float, dict]:
    """
    Returns (final_score 0-100, breakdown dict with per-criterion scores 0-100).
    """
    # Resize for consistent analysis (keep aspect ratio)
    img_work = img.copy()
    max_dim = 1024
    if max(img_work.size) > max_dim:
        img_work.thumbnail((max_dim, max_dim), Image.LANCZOS)

    raw = {
        "noise_grain":           score_noise_grain(img_work),
        "sharpness_bokeh":       score_sharpness_with_bokeh(img_work),
        "color_naturalism":      score_color_naturalism(img_work),
        "texture_complexity":    score_texture_complexity(img_work),
        "lens_optics":           score_chromatic_aberration_and_vignette(img_work),
        "compression_artifacts": score_compression_artifacts(img_work),
        "depth_of_field":        score_depth_of_field_blur(img_work),
    }

    weighted = sum(raw[k] * WEIGHTS[k] for k in WEIGHTS)
    final = round(weighted * 100, 2)

    breakdown = {k: round(v * 100, 1) for k, v in raw.items()}
    return final, breakdown


def label_for_score(score: float) -> str:
    if score >= 85:  return "Excellent – highly photo-realistic"
    if score >= 70:  return "Good – convincingly phone-like"
    if score >= 55:  return "Moderate – passable realism"
    if score >= 40:  return "Low – noticeable AI characteristics"
    return "Poor – clearly synthetic"


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI Node class
# ─────────────────────────────────────────────────────────────────────────────

class RealismScorerNode:
    """
    ComfyUI node that scores an image on phone-photo realism.
    Outputs a numeric score (0-100) and a human-readable label.
    """

    CATEGORY = "image/analysis"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "show_breakdown": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES  = ("FLOAT", "INT", "STRING", "STRING")
    RETURN_NAMES  = ("score_float", "score_int", "label", "breakdown")
    FUNCTION      = "score"

    def score(self, image: torch.Tensor, show_breakdown: bool = True):
        pil = tensor_to_pil(image)
        final_score, breakdown = compute_realism_score(pil)
        label = label_for_score(final_score)

        breakdown_str = ""
        if show_breakdown:
            lines = [f"{'Criterion':<28} {'Score':>6}"]
            lines.append("─" * 36)
            friendly_names = {
                "noise_grain":           "Sensor Noise / Film Grain",
                "sharpness_bokeh":       "Sharpness + Bokeh Falloff",
                "color_naturalism":      "Color Naturalism",
                "texture_complexity":    "Texture Complexity",
                "lens_optics":           "Lens Optics (CA + Vignette)",
                "compression_artifacts": "JPEG Compression Character",
                "depth_of_field":        "Depth-of-Field Gradient",
            }
            for k, v in breakdown.items():
                bar = "█" * int(v / 10) + "░" * (10 - int(v / 10))
                lines.append(f"{friendly_names[k]:<28} {v:>5.1f}  {bar}")
            lines.append("─" * 36)
            lines.append(f"{'TOTAL REALISM SCORE':<28} {final_score:>5.1f}")
            lines.append(f"\n{label}")
            breakdown_str = "\n".join(lines)

        return (
            float(final_score),
            int(round(final_score)),
            label,
            breakdown_str,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RealismScorer": RealismScorerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RealismScorer": "📷 Realism Scorer (Phone-Photo Quality)",
}
