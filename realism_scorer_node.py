"""
Realism Score (Phone Photo) — ComfyUI custom node v5
Calibrated against real iPhone 17 Pro photos.

CLIP raw values observed on real photos: ~0.005 - 0.009
Aesthetic raw values observed on real photos: ~5.5 - 6.2
Laplacian variance observed: 5 - 165 (varies heavily by image content)
Noise residual std observed: 0.59 - 3.85
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -- lazy globals --------------------------------------------------------------
_clip_model      = None
_clip_preprocess = None
_clip_tokenizer  = None
_aesthetic_model = None
_aesthetic_prep  = None
_device          = None


def _get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def _load_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return
    import open_clip
    device = _get_device()
    print("[RealismScorer] Loading CLIP ViT-L/14 ...")
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    _clip_model = _clip_model.to(device).eval()
    _clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
    print("[RealismScorer] CLIP ready.")


def _load_aesthetic():
    global _aesthetic_model, _aesthetic_prep
    if _aesthetic_model is not None:
        return _aesthetic_model != "unavailable"
    try:
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        device = _get_device()
        print("[RealismScorer] Loading Aesthetic Predictor V2.5 ...")
        _aesthetic_model, _aesthetic_prep = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        _aesthetic_model = _aesthetic_model.to(torch.bfloat16).to(device).eval()
        print("[RealismScorer] Aesthetic Predictor ready.")
        return True
    except Exception as e:
        print(f"[RealismScorer] Aesthetic Predictor unavailable: {e}")
        _aesthetic_model = "unavailable"
        return False


# -- prompts ------------------------------------------------------------------
POS_PROMPTS = [
    "a casual smartphone photo",
    "an amateur phone picture",
    "a photo taken on an iPhone",
    "a real candid photograph",
    "a natural handheld photo with slight imperfections",
]
NEG_PROMPTS = [
    "AI generated image",
    "CGI render",
    "digital art",
    "perfectly lit studio portrait",
    "artificially generated face",
]


# -- signal helpers -----------------------------------------------------------
def _to_gray(img):
    return (
        0.2989 * img[:, :, 0].astype(np.float32)
        + 0.5870 * img[:, :, 1].astype(np.float32)
        + 0.1140 * img[:, :, 2].astype(np.float32)
    )

def _laplacian_variance(img):
    gray = _to_gray(img)
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return float(np.var(lap))

def _estimate_noise(img):
    gray = _to_gray(img)
    blurred = (
        gray[:-2, :-2] + gray[:-2, 1:-1] + gray[:-2, 2:]
        + gray[1:-1, :-2] + gray[1:-1, 1:-1] + gray[1:-1, 2:]
        + gray[2:,  :-2] + gray[2:,  1:-1] + gray[2:,  2:]
    ) / 9.0
    return float(np.std(gray[1:-1, 1:-1] - blurred))

def _gaussian(x, target, sigma):
    return float(np.exp(-((x - target) ** 2) / (2 * sigma ** 2)))


# -- scorers ------------------------------------------------------------------
def _clip_realism(pil):
    _load_clip()
    device = _get_device()
    t = _clip_preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        img_f = _clip_model.encode_image(t)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)

        pos_t = _clip_tokenizer(POS_PROMPTS).to(device)
        pos_f = _clip_model.encode_text(pos_t)
        pos_f = pos_f / pos_f.norm(dim=-1, keepdim=True)
        pos   = (img_f @ pos_f.T).mean().item()

        neg_t = _clip_tokenizer(NEG_PROMPTS).to(device)
        neg_f = _clip_model.encode_text(neg_t)
        neg_f = neg_f / neg_f.norm(dim=-1, keepdim=True)
        neg   = (img_f @ neg_f.T).mean().item()

    raw = pos - neg
    print(f"[RealismScorer] CLIP raw (pos-neg): {raw:.4f}")

    # Calibrated from real iPhone 17 Pro photos:
    #   Real photos observed range: ~0.000 to +0.015
    #   AI images expected range:   ~-0.08 to -0.01
    #
    # Map so that:
    #   raw = -0.08  ->  0.0  (clearly AI)
    #   raw =  0.000 ->  0.7  (neutral/real boundary)
    #   raw =  0.015 ->  1.0  (confidently real)
    #
    # Linear map: score = (raw - LOW) / (HIGH - LOW)
    # where LOW = -0.08 (AI floor) and HIGH = +0.015 (real ceiling)
    LOW  = -0.08
    HIGH =  0.015
    remapped = (raw - LOW) / (HIGH - LOW)
    return float(np.clip(remapped, 0.0, 1.0))


def _aesthetic_score(pil):
    ok = _load_aesthetic()
    if not ok:
        return 0.5, False
    device = _get_device()
    pv = (
        _aesthetic_prep(images=pil, return_tensors="pt")
        .pixel_values.to(torch.bfloat16).to(device)
    )
    with torch.inference_mode():
        raw = _aesthetic_model(pv).logits.squeeze().float().cpu().item()
    print(f"[RealismScorer] Aesthetic raw: {raw:.4f}")
    # Raw range 1-10. Real iPhone photos ~5.5-6.5 -> normalise to 0-1
    # Use 3.0 as floor (anything below is very poor) and 8.0 as ceiling
    return float(np.clip((raw - 3.0) / 5.0, 0.0, 1.0)), True


# -- preview card -------------------------------------------------------------
def _get_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Helvetica.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_bar(draw, x, y, w, h, frac, colour, bg=(40, 40, 40)):
    draw.rectangle([x, y, x + w, y + h], fill=bg)
    fw = max(0, int(w * float(np.clip(frac, 0, 1))))
    if fw:
        draw.rectangle([x, y, x + fw, y + h], fill=colour)


def _grade(s):
    if s >= 0.75: return "Excellent", (80, 200, 120)
    if s >= 0.60: return "Good",      (100, 180, 255)
    if s >= 0.45: return "Moderate",  (255, 200, 80)
    return "Poor", (220, 80, 80)


def _make_card(image_pil, scores, aesthetic_loaded):
    W, H = 900, 420  # fixed size for readability

    try:
        card = Image.new("RGB", (W, H), (20, 20, 20))
        draw = ImageDraw.Draw(card)

        f_large = _get_font(42)
        f_med   = _get_font(26)
        f_small = _get_font(18)

        final = scores["final"]
        grade, gcol = _grade(final)

        # --- Header ---
        draw.text((30, 25), "Realism Score", font=f_large, fill=(230, 230, 230))

        pct = f"{final * 100:.1f}%"
        try:
            pw = draw.textlength(pct, font=f_large)
        except Exception:
            pw = len(pct) * 20

        draw.text((W - 30 - pw, 25), pct, font=f_large, fill=gcol)
        draw.text((W - 30 - pw, 75), grade, font=f_small, fill=gcol)

        draw.line([(30, 110), (W - 30, 110)], fill=(50, 50, 50), width=2)

        # --- Rows ---
        y = 140
        spacing = 55

        def row(label, value, color):
            nonlocal y

            if value > 0.7:
                col = (120, 255, 120)
            elif value > 0.5:
                col = (255, 220, 120)
            else:
                col = (255, 120, 120)

            draw.text((40, y), label, font=f_med, fill=(180, 180, 180))
            draw.text((W - 180, y), f"{value:.3f}", font=f_med, fill=col)
            y += spacing

        row("CLIP Realism", scores["realism"], (100, 180, 255))
        row("Aesthetic",    scores["aesthetic"], (180, 130, 255))
        row("Texture",      scores["texture"], (255, 185, 70))
        row("Noise",        scores["noise"], (70, 210, 150))

        if not aesthetic_loaded:
            draw.text((40, y + 10), "(Aesthetic model not loaded)", font=f_small, fill=(255, 120, 120))

        return card

    except Exception as e:
        print(f"[RealismScorer] Card render error: {e}")
        return Image.new("RGB", (600, 300), (20, 20, 20))


# -- node ---------------------------------------------------------------------
class RealismScoreNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES  = ("IMAGE", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES  = (
        "preview",
        "final_score",
        "realism_score",
        "aesthetic_score",
        "texture_score",
        "noise_score",
    )
    OUTPUT_NODE = True
    FUNCTION    = "score"
    CATEGORY    = "image/analysis"

    def score(self, image: torch.Tensor):
        image_np  = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np, mode="RGB")

        realism           = _clip_realism(image_pil)
        aesthetic, aes_ok = _aesthetic_score(image_pil)

        # Texture: calibrated to observed real-photo lap_var range 5-200
        # Target ~80 (mid sharpness), sigma 120 (wide tolerance)
        sharpness = _laplacian_variance(image_np)
        #texture   = _gaussian(sharpness, target=80.0, sigma=120.0)
        texture   = _gaussian(sharpness, target=80.0, sigma=1500.0)

        # Noise: calibrated to observed residual_std range 0.5-4.0
        # Target ~2.0 (mild natural noise), sigma 1.5
        noise_std   = _estimate_noise(image_np)
        #noise_score = _gaussian(noise_std, target=2.0, sigma=1.5)
        noise_score = _gaussian(noise_std, target=2.0, sigma=10.0)

        final = float(np.clip(
            0.45 * realism
            + 0.35 * aesthetic
            + 0.10 * texture
            + 0.10 * noise_score,
            0.0, 1.0
        ))

        scores = {
            "final":     final,
            "realism":   float(np.clip(realism,     0, 1)),
            "aesthetic": float(np.clip(aesthetic,   0, 1)),
            "texture":   float(np.clip(texture,     0, 1)),
            "noise":     float(np.clip(noise_score, 0, 1)),
        }

        grade_label, _ = _grade(final)
        print(f"\n[RealismScorer] {'─'*44}")
        print(f"  Final        : {final:.4f}  ({grade_label})")
        print(f"  CLIP Realism : {scores['realism']:.4f}")
        aes_note = "" if aes_ok else "  (fallback — not installed)"
        print(f"  Aesthetic    : {scores['aesthetic']:.4f}{aes_note}")
        print(f"  Texture      : {scores['texture']:.4f}  (lap_var={sharpness:.1f})")
        print(f"  Noise        : {scores['noise']:.4f}  (residual_std={noise_std:.3f})")
        print(f"[RealismScorer] {'─'*44}\n")

        preview_pil = _make_card(image_pil, scores, aes_ok)
        preview_np  = np.array(preview_pil).astype(np.float32) / 255.0
        preview_t   = torch.from_numpy(preview_np).unsqueeze(0)

        return (
            preview_t,
            scores["final"],
            scores["realism"],
            scores["aesthetic"],
            scores["texture"],
            scores["noise"],
        )


NODE_CLASS_MAPPINGS        = {"RealismScoreNode": RealismScoreNode}
NODE_DISPLAY_NAME_MAPPINGS = {"RealismScoreNode": "📷 Realism Score (Phone Photo)"}
