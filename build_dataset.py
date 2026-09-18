"""
Build a trainable dataset from what download_data.py fetched.

Combines:  FiveK DNGs (many different camera bodies)  x  HaldCLUT looks (film/camera styles)
        -> perfectly aligned (RAW crop, styled sRGB crop) pairs.

Key design point: style_key is the LOOK, not the camera. The same look is rendered across
many different camera bodies, and the front-end is conditioned on each file's own CCM +
EXIF, so the model learns "apply this look to any sensor" rather than "imitate one camera."

  python build_dataset.py --list-sources
  python build_dataset.py --scenes 300 --looks 12 --crops 4
  python build_dataset.py --scenes 50 --looks 4 --crops 2 --crop-size 512   # quick smoke test

Output (default /mnt/data/prepared):
  crops/<scene>__<look>__<k>.npz   bayer(float16) + ccm(3x3) + meta
  crops/<scene>__<look>__<k>.jpg   styled target, same pixels
  luts/<look>.npy                  (3,33,33,33) float32, ready for the training script
  manifest.jsonl
"""

import os, io, sys, glob, json, random, argparse, warnings
import numpy as np

warnings.filterwarnings("ignore")

DATA_ROOT = os.environ.get("CAMERA_LOOK_DATA", "/mnt/data")
LUT_N = 33


# ---------------------------------------------------------------- discovery

def find_dngs(root):
    pats = ["**/*.dng", "**/*.DNG", "**/*.nef", "**/*.NEF", "**/*.cr2", "**/*.CR2",
            "**/*.arw", "**/*.ARW", "**/*.raf", "**/*.RAF"]
    out = []
    for p in pats:
        out += glob.glob(os.path.join(root, p), recursive=True)
    return sorted(set(out))


def find_haldcluts(root):
    out = []
    for p in ["**/*.png", "**/*.PNG", "**/*.tif", "**/*.TIF", "**/*.tiff"]:
        out += glob.glob(os.path.join(root, p), recursive=True)
    keep = []
    for f in sorted(set(out)):
        try:
            from PIL import Image
            with Image.open(f) as im:
                w, h = im.size
            if w == h and w >= 64:
                n = round(w ** (1 / 3))
                if n ** 3 == w:
                    keep.append(f)
        except Exception:
            continue
    return keep


# ---------------------------------------------------------------- hald clut -> 3D LUT

def hald_to_cube(path):
    """HaldCLUT image (side n^3, cube side L=n^2) -> float32 cube [L,L,L,3] indexed [b,g,r]."""
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    side = arr.shape[1]
    n = round(side ** (1 / 3))
    if n ** 3 != side:
        raise ValueError(f"not a hald clut: {path}")
    L = n * n
    flat = arr.reshape(-1, 3)
    if flat.shape[0] != L ** 3:
        raise ValueError(f"unexpected hald size: {path}")
    return flat.reshape(L, L, L, 3)


def resample_cube(cube, out_n=LUT_N):
    """Trilinear-resample an [L,L,L,3] cube to [3,out_n,out_n,out_n] laid out for grid_sample."""
    import torch
    L = cube.shape[0]
    src = torch.from_numpy(cube).permute(3, 0, 1, 2).unsqueeze(0)  # 1,3,L,L,L  (D=b,H=g,W=r)
    lin = torch.linspace(-1, 1, out_n)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    grid = torch.stack([xx, yy, zz], -1).unsqueeze(0)
    out = torch.nn.functional.grid_sample(src, grid, mode="bilinear", align_corners=True)
    return out.squeeze(0).numpy().astype(np.float32)  # 3,out_n,out_n,out_n


def apply_cube(rgb, lut):
    """rgb HxWx3 float [0,1]; lut 3xNxNxN -> styled HxWx3."""
    import torch
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    vol = torch.from_numpy(lut).unsqueeze(0)
    grid = (t.clamp(0, 1) * 2 - 1).permute(0, 2, 3, 1)
    h, w = rgb.shape[:2]
    grid = grid.reshape(1, h * w, 1, 1, 3)
    out = torch.nn.functional.grid_sample(vol, grid, mode="bilinear", align_corners=True)
    return out.reshape(3, h, w).permute(1, 2, 0).numpy().clip(0, 1)


BRANDS = {
    "fuji": "Fujifilm", "fujifilm": "Fujifilm", "velvia": "Fujifilm", "provia": "Fujifilm",
    "astia": "Fujifilm", "superia": "Fujifilm", "sensia": "Fujifilm", "reala": "Fujifilm",
    "kodak": "Kodak", "portra": "Kodak", "ektar": "Kodak", "gold": "Kodak",
    "tri-x": "Kodak", "trix": "Kodak", "ektachrome": "Kodak", "kodachrome": "Kodak",
    "agfa": "Agfa", "agfacolor": "Agfa", "polaroid": "Polaroid", "ilford": "Ilford",
    "canon": "Canon", "nikon": "Nikon", "sony": "Sony", "leica": "Leica",
    "olympus": "Olympus", "panasonic": "Panasonic", "pentax": "Pentax",
}


def detect_brand(name):
    low = name.lower()
    for key, brand in BRANDS.items():
        if key in low:
            return brand
    return None


def bt709_decode(y):
    y = np.clip(y, 0.0, 1.0)
    return np.where(y < 4.5 * 0.018, y / 4.5, ((np.maximum(y, 4.5 * 0.018) + 0.099) / 1.099) ** 2.222)


def bt709_encode(x):
    x = np.clip(x, 1e-6, 1.0)
    return np.where(x < 0.018, 4.5 * x, 1.099 * np.maximum(x, 0.018) ** (1 / 2.222) - 0.099)


LUM = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _warm(lin, k):
    return lin * np.array([1 + k, 1 + 0.15 * k, 1 - k], dtype=np.float32)


def _expose(lin, k):
    return lin * (2.0 ** k)


def _contrast(lin, k):
    y = bt709_encode(lin)
    y = np.clip(0.5 + (y - 0.5) * (1 + k), 0, 1)
    return bt709_decode(y)


def _saturate(lin, k):
    g = (lin * LUM).sum(-1, keepdims=True)
    return np.clip(g + (lin - g) * (1 + k), 0, None)


def _mono(lin, k):
    g = (lin * LUM).sum(-1, keepdims=True)
    return np.clip(g + (lin - g) * (1 - k), 0, None)


# name -> (fn, signed strengths, phrase for +k, phrase for -k)
EDITS = {
    "warm":     (_warm,     [0.08, 0.18, 0.30], "warmer",          "cooler"),
    "expose":   (_expose,   [0.25, 0.50, 0.85], "brighter",        "darker"),
    "contrast": (_contrast, [0.20, 0.45, 0.75], "more contrasty",  "flatter"),
    "satur":    (_saturate, [0.20, 0.45, 0.80], "more saturated",  "more muted"),
}

LEVELS = ["slightly ", "", "much "]

EDIT_TEMPLATES = [
    "make this {p}",
    "make it {p}",
    "{p}",
    "can you make this {p}",
    "I want this {p}",
    "edit this to be {p}",
]

MONO_PROMPTS = [
    "black and white", "make this black and white", "convert to monochrome",
    "high contrast black and white", "desaturate this completely", "make it greyscale",
]


def make_edit_pairs(base_srgb, rng, n=2):
    """Relative-instruction pairs. Without these, 'make this warmer' is out-of-distribution:
    the look prompts only ever say 'apply named look X', so the model has no way to learn
    what a *relative* instruction means, and warmer/cooler collapse to the same output."""
    lin = bt709_decode(base_srgb)
    out = []
    keys = list(EDITS)
    rng.shuffle(keys)
    for key in keys[:n]:
        fn, strengths, pos, neg = EDITS[key]
        li = rng.randrange(len(strengths))
        k = strengths[li]
        sign = 1 if rng.random() < 0.5 else -1
        phrase = pos if sign > 0 else neg
        prompt = rng.choice(EDIT_TEMPLATES).format(p=LEVELS[li] + phrase)
        styled = bt709_encode(np.clip(fn(lin, sign * k), 0, 1))
        out.append((styled, prompt, f"edit_{key}_{'p' if sign > 0 else 'n'}{li}"))
    if rng.random() < 0.35:
        k = rng.choice([0.85, 1.0])
        prompt = rng.choice(MONO_PROMPTS)
        if "high contrast" in prompt:
            styled = bt709_encode(np.clip(_contrast(_mono(lin, k), 0.5), 0, 1))
            key = "edit_mono_hc"
        else:
            styled = bt709_encode(np.clip(_mono(lin, k), 0, 1))
            key = "edit_mono"
        out.append((styled, prompt, key))
    return out


_PROBE_RGB = None


def _probe_colors(n=4096, seed=0):
    global _PROBE_RGB
    if _PROBE_RGB is None:
        rng = np.random.default_rng(seed)
        _PROBE_RGB = rng.uniform(0.02, 0.98, (n, 3)).astype(np.float32)
    return _PROBE_RGB


def _srgb_to_lin_np(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lab_np(rgb):
    lin = _srgb_to_lin_np(np.clip(rgb, 0, 1))
    M = np.array([[.4124564, .3575761, .1804375],
                  [.2126729, .7151522, .0721750],
                  [.0193339, .1191920, .9503041]], dtype=np.float32)
    xyz = lin @ M.T
    xyz = xyz / np.array([.95047, 1.0, 1.08883], dtype=np.float32)
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(np.maximum(xyz, 1e-8)), xyz / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def _sample_cube_np(cube, rgb):
    """Trilinear sample of a [3,N,N,N] cube (indexed [c,b,g,r]) at rgb points, in numpy.
    Mirrors grid_sample(align_corners=True) so it matches the torch path used for rendering."""
    C, D, H, W = cube.shape
    x = np.clip(rgb[:, 0], 0, 1) * (W - 1)
    y = np.clip(rgb[:, 1], 0, 1) * (H - 1)
    z = np.clip(rgb[:, 2], 0, 1) * (D - 1)
    x0, y0, z0 = np.floor(x).astype(int), np.floor(y).astype(int), np.floor(z).astype(int)
    x1, y1, z1 = np.clip(x0 + 1, 0, W - 1), np.clip(y0 + 1, 0, H - 1), np.clip(z0 + 1, 0, D - 1)
    x0, y0, z0 = np.clip(x0, 0, W - 1), np.clip(y0, 0, H - 1), np.clip(z0, 0, D - 1)
    tx, ty, tz = x - x0, y - y0, z - z0
    out = np.zeros((len(rgb), C), dtype=np.float32)
    for c in range(C):
        v = cube[c]
        c00 = v[z0, y0, x0] * (1 - tx) + v[z0, y0, x1] * tx
        c01 = v[z0, y1, x0] * (1 - tx) + v[z0, y1, x1] * tx
        c10 = v[z1, y0, x0] * (1 - tx) + v[z1, y0, x1] * tx
        c11 = v[z1, y1, x0] * (1 - tx) + v[z1, y1, x1] * tx
        c0 = c00 * (1 - ty) + c01 * ty
        c1 = c10 * (1 - ty) + c11 * ty
        out[:, c] = c0 * (1 - tz) + c1 * tz
    return out


def look_mono_score(cube):
    """How much a look collapses chroma: 1.0 = fully monochrome, 0.0 = chroma preserved.

    The RawTherapee film-simulation pack is heavily black-and-white (Ilford, Tri-X, T-Max,
    Neopan, APX ...). If a large share of the chosen looks are monochrome, the director
    learns that a film or camera prompt usually means DESATURATE, and unrelated prompts come
    back grey. Difficulty normalisation equalises gradient weight per sample but cannot fix a
    skewed prior -- if most look targets really are grey, grey is the correct average.
    """
    rgb = _probe_colors()
    out = _sample_cube_np(cube, rgb)
    c_in = np.abs(rgb - rgb.mean(axis=-1, keepdims=True)).mean()
    c_out = np.abs(out - out.mean(axis=-1, keepdims=True)).mean()
    return float(np.clip(1.0 - c_out / max(c_in, 1e-6), 0.0, 1.0))


def look_strength(cube):
    """Mean dE between a colour and that colour through the LUT: how far the look moves things.

    The RawTherapee pack is a CREATIVE CLUT collection, not a set of camera profiles. It
    contains effects like day-for-night that no tone curve + hue/sat map can reproduce, and
    they dominate the loss while being unlearnable. A real camera picture style sits around
    5-10 dE from neutral; anything past ~15 is an effect, not a look.
    """
    rgb = _probe_colors()
    out = _sample_cube_np(cube, rgb)
    return float(np.linalg.norm(_lab_np(rgb) - _lab_np(out), axis=-1).mean())


def _read_tiff_raw(path):
    """Return (float array in [0,1], icc_profile_or_None).

    PIL's handling of multi-channel 16-bit TIFF is unreliable -- depending on the file it
    may hand back an 8-bit truncation, mode 'I;16' or mode 'F', and a later convert('RGB')
    then silently produces garbage (black or grey). tifffile/imageio read the real samples,
    so prefer them and fall back to PIL only when neither is installed.
    """
    icc = None
    try:
        from PIL import Image as _I
        with _I.open(path) as im:
            icc = im.info.get("icc_profile")
    except Exception:
        pass
    for mod in ("tifffile", "imageio.v3"):
        try:
            if mod == "tifffile":
                import tifffile
                a = tifffile.imread(path)
            else:
                import imageio.v3 as iio
                a = iio.imread(path)
            a = np.asarray(a)
            if a.ndim == 3 and a.shape[-1] > 3:
                a = a[..., :3]
            if a.dtype == np.uint16:
                a = a.astype(np.float32) / 65535.0
            elif a.dtype == np.uint8:
                a = a.astype(np.float32) / 255.0
            else:
                a = a.astype(np.float32)
                if a.max() > 1.5:
                    a = a / a.max()
            return np.clip(a, 0, 1), icc
        except ImportError:
            continue
        except Exception as e:
            print(f"    {mod} failed on {os.path.basename(path)}: {e}")
            continue
    from PIL import Image as _I
    im = _I.open(path)
    if im.mode not in ("RGB", "RGBA"):
        raise RuntimeError(
            f"{os.path.basename(path)} is mode {im.mode}; PIL cannot read this 16-bit TIFF "
            f"reliably. Install tifffile:  pip install tifffile")
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return np.clip(a, 0, 1), icc


def _read_tiff_rgb(path):
    """Read a 16-bit RGB TIFF as float [0,1], HxWx3.

    PIL has no 16-bit-per-channel RGB mode: such a file can load as 'I;16', a SINGLE
    channel, and .convert('RGB') then replicates luminance across R/G/B -- a grey image,
    with no error raised. Prefer tifffile/imageio; accept PIL only if it really gave 3ch.
    """
    arr = None
    try:
        import tifffile
        arr = tifffile.imread(path)
    except Exception:
        try:
            import imageio.v3 as iio
            arr = iio.imread(path)
        except Exception:
            from PIL import Image
            im = Image.open(path)
            if im.mode not in ("RGB", "RGBA"):
                raise RuntimeError(
                    f"{os.path.basename(path)}: PIL loaded mode '{im.mode}', not RGB. "
                    f"16-bit TIFFs need a real reader:  pip install tifffile")
            arr = np.asarray(im.convert("RGB"))
    arr = np.asarray(arr)
    if arr.ndim == 2:
        raise RuntimeError(f"{os.path.basename(path)}: single-channel read {arr.shape}; "
                           f"this is the grey-target failure. pip install tifffile")
    arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / arr.max()
    return np.clip(arr, 0, 1)


def validate_target(arr, label):
    """Reject degenerate targets before they poison training. A black or fully-desaturated
    target teaches the director to emit black/grey for every prompt, which only becomes
    visible much later, at evaluation."""
    mean = float(arr.mean())
    chroma = float(np.abs(arr - arr.mean(axis=-1, keepdims=True)).mean())
    if mean < 0.02:
        raise RuntimeError(f"{label}: near-black target (mean {mean:.4f})")
    if mean > 0.98:
        raise RuntimeError(f"{label}: near-white target (mean {mean:.4f})")
    if chroma < 0.002:
        raise RuntimeError(f"{label}: no chroma (mean |dev| {chroma:.5f}) -- almost certainly "
                           f"a single-channel read collapsed to grey")
    return True


def load_expert_tiff(path, size_hw):
    """FiveK expert retouch -> sRGB float, matched to our base renders.

    16-bit ProPhoto RGB. Treating it as sRGB shifts every colour (much wider primaries), so
    convert via the embedded ICC profile when present, else an explicit matrix.
    """
    from PIL import Image, ImageCms
    arr = _read_tiff_rgb(path)
    converted = False
    try:
        icc = Image.open(path).info.get("icc_profile")
    except Exception:
        icc = None
    if icc:
        try:
            im8 = Image.fromarray((arr * 255).astype(np.uint8), "RGB")
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst = ImageCms.createProfile("sRGB")
            arr = np.asarray(ImageCms.profileToProfile(im8, src, dst, outputMode="RGB"),
                             dtype=np.float32) / 255.0
            converted = True
        except Exception:
            pass
    if not converted:
        lin = np.where(arr < 16 * (1 / 512), arr / 16, np.clip(arr, 1e-8, None) ** 1.8)
        M_pro = np.array([[0.7976749, 0.1351917, 0.0313534],
                          [0.2880402, 0.7118741, 0.0000857],
                          [0.0000000, 0.0000000, 0.8252100]], dtype=np.float32)
        M_srgb_inv = np.array([[3.2404542, -1.5371385, -0.4985314],
                               [-0.9692660, 1.8760108, 0.0415560],
                               [0.0556434, -0.2040259, 1.0572252]], dtype=np.float32)
        brad = np.array([[0.9555766, -0.0230393, 0.0631636],
                         [-0.0282895, 1.0099416, 0.0210077],
                         [0.0122982, -0.0204830, 1.3299098]], dtype=np.float32)
        xyz = lin @ M_pro.T @ brad.T
        rgb = np.clip(xyz @ M_srgb_inv.T, 0, 1)
        arr = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)
    validate_target(arr, os.path.basename(path))
    h, w = size_hw
    if arr.shape[0] != h or arr.shape[1] != w:
        from PIL import Image as _I
        arr = np.asarray(_I.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)
                                      ).resize((w, h), _I.LANCZOS), dtype=np.float32) / 255.0
    return np.clip(arr, 0, 1)


EXPERT_PROMPTS = {
    "a": ["expert A retouch", "retouch this like a professional", "give this a clean professional edit"],
    "b": ["expert B retouch", "a tasteful photographic edit", "edit this like a pro photographer"],
    "c": ["expert C retouch", "professionally retouched", "give this a natural professional grade",
          "make this look professionally edited"],
    "d": ["expert D retouch", "a polished photographic look", "retouch this photo nicely"],
    "e": ["expert E retouch", "a refined photo edit", "make this look like a finished photograph"],
}


def look_name(path):
    return os.path.splitext(os.path.basename(path))[0].replace(" ", "_").replace("/", "_")


def pretty_name(name):
    return name.replace("_", " ").replace("-", " ").strip()


def make_prompts(name):
    """Natural-language phrasings for one look, from literal to vague."""
    p = pretty_name(name)
    brand = detect_brand(name)
    out = [
        f"{p}",
        f"{p} film look",
        f"apply the {p} look",
        f"{p} film simulation",
        f"give this photo a {p} feel",
        f"emulate {p}",
        f"grade this like {p}",
        f"process this with the {p} profile",
    ]
    if brand:
        out += [
            f"make this look like it was shot on a {brand} camera",
            f"make it look like {brand} film",
            f"shot on {brand}, {p}",
            f"render this the way a {brand} body would",
            f"I want that classic {brand} colour",
            f"{brand} {p} picture style",
            f"can you make this feel more {brand}",
        ]
    return out


def look_prompt(name):
    return make_prompts(name)[1]


# ---------------------------------------------------------------- raw handling

def alignment_score(bayer, base):
    """Pearson correlation between bayer-derived luma and the rendered luma.
    ~1.0 = aligned, ~0 = unrelated content (wrong orientation, wrong file, offset).
    Cheap insurance: misaligned pairs train a model to output a grey blur."""
    bl = 0.25 * (bayer[0::2, 0::2] + bayer[0::2, 1::2] + bayer[1::2, 0::2] + bayer[1::2, 1::2])
    gl = base.mean(axis=2)
    gl = 0.25 * (gl[0::2, 0::2] + gl[0::2, 1::2] + gl[1::2, 0::2] + gl[1::2, 1::2])
    h = min(bl.shape[0], gl.shape[0])
    w = min(bl.shape[1], gl.shape[1])
    a = np.sqrt(np.clip(bl[:h, :w], 0, 1)).ravel()   # rough gamma so scales are comparable
    b = gl[:h, :w].ravel()
    if a.size > 200000:
        step = a.size // 200000 + 1
        a, b = a[::step], b[::step]
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else 0.0


def load_raw(path):
    """-> (bayer HxW float32 [0,1], base sRGB HxWx3 float32, ccm 3x3, meta dict) or None."""
    import rawpy
    with rawpy.imread(path) as raw:
        if raw.raw_image_visible.ndim != 2:
            return None
        bayer = raw.raw_image_visible.astype(np.float32)
        black = float(np.mean(raw.black_level_per_channel))
        white = float(raw.white_level)
        bayer = np.clip((bayer - black) / max(white - black, 1.0), 0, 1)
        try:
            cm = np.asarray(raw.color_matrix[:3, :3], dtype=np.float32)
            if not np.isfinite(cm).all() or np.allclose(cm, 0):
                cm = np.eye(3, dtype=np.float32)
        except Exception:
            cm = np.eye(3, dtype=np.float32)
        # user_flip=0 is CRITICAL: postprocess() otherwise applies the camera's EXIF
        # orientation (rotating portrait shots 90 deg) while raw_image_visible stays in
        # trained on them learns a grey blur (the conditional mean of all targets).
        base = raw.postprocess(no_auto_bright=True, output_bps=8, user_flip=0,
                               use_camera_wb=True, gamma=(2.222, 4.5))
    base = base.astype(np.float32) / 255.0
    h = min(bayer.shape[0], base.shape[0])
    w = min(bayer.shape[1], base.shape[1])
    h -= h % 2
    w -= w % 2
    return bayer[:h, :w], base[:h, :w], cm, {}


def read_exif(path):
    meta = {"iso": 100.0, "exposure": 1 / 60, "fnumber": 4.0,
            "focal_length": 35.0, "camera": "unknown"}
    try:
        import exifread
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        def num(key, default):
            v = tags.get(key)
            if v is None:
                return default
            try:
                r = v.values[0]
                return float(r.num) / float(r.den) if hasattr(r, "num") else float(r)
            except Exception:
                return default

        meta["iso"] = num("EXIF ISOSpeedRatings", 100.0)
        meta["exposure"] = num("EXIF ExposureTime", 1 / 60)
        meta["fnumber"] = num("EXIF FNumber", 4.0)
        meta["focal_length"] = num("EXIF FocalLength", 35.0)
        make = str(tags.get("Image Make", "")).strip()
        model = str(tags.get("Image Model", "")).strip()
        if make or model:
            meta["camera"] = f"{make} {model}".strip()
    except Exception:
        pass
    return meta


# ---------------------------------------------------------------- build

def check_aligned(base, target, label, thresh=0.7):
    """Base and target are the same pixels under a colour transform, so their luminance
    STRUCTURE must correlate almost perfectly. A low correlation means they are different
    crops -- a mismatch no colour metric reveals (it just looks like a very strong look) and
    which no colour transform can fit, so it silently inflates every downstream number."""
    a = base.mean(-1)
    b = target.mean(-1)
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    if den < 1e-8:
        return True
    corr = float((a * b).sum() / den)
    # ABSOLUTE correlation: an inverting look (the pack contains a "Negative" CLUT) gives
    # corr ~ -1 while being perfectly aligned. Only the magnitude of the relationship tells
    # us the two images share structure.
    if abs(corr) < thresh:
        raise RuntimeError(f"{label}: base/target structure correlation {corr:.2f} "
                           f"(|{abs(corr):.2f}| < {thresh}) -- these are not the same crop")
    return True


def audit_manifest(path, sample=400, seed=0):
    """What do the targets actually look like, in aggregate?

    A director that ignores the prompt outputs something near the dataset MEAN target. So if
    evaluation shows one fixed dark/grey result for every prompt, the first thing to check is
    whether the mean target is itself dark and grey -- in which case that output is the
    loss-minimising prompt-independent answer and the model is behaving rationally.
    """
    from PIL import Image
    items = [json.loads(l) for l in open(path) if l.strip()]
    if not items:
        print("empty manifest")
        return
    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:sample]

    by_key = {}
    misaligned = 0
    tb, tc, bb, bc = [], [], [], []
    for it in items:
        try:
            t = np.asarray(Image.open(it["jpeg_path"]).convert("RGB"), dtype=np.float32) / 255.0
        except Exception:
            continue
        bmean = bchroma = float("nan")
        if "base_path" in it and os.path.exists(it["base_path"]):
            b = np.asarray(Image.open(it["base_path"]).convert("RGB"), dtype=np.float32) / 255.0
            bmean = float(b.mean())
            bchroma = float(np.abs(b - b.mean(-1, keepdims=True)).mean())
            bb.append(bmean); bc.append(bchroma)
        if not np.isnan(bmean):
            try:
                check_aligned(b, t, it.get("style_key", "?"))
            except RuntimeError:
                misaligned += 1
        tmean = float(t.mean())
        tchroma = float(np.abs(t - t.mean(-1, keepdims=True)).mean())
        tb.append(tmean); tc.append(tchroma)
        k = it.get("style_key", "?")
        d = by_key.setdefault(k, {"n": 0, "mean": 0.0, "chroma": 0.0})
        d["n"] += 1; d["mean"] += tmean; d["chroma"] += tchroma

    print(f"\naudited {len(tb)} targets from {path}")
    if bb:
        print(f"  base   mean {np.mean(bb):.3f}  chroma {np.mean(bc):.4f}")
    print(f"  target mean {np.mean(tb):.3f}  chroma {np.mean(tc):.4f}")
    dark = sum(1 for v in tb if v < 0.15)
    grey = sum(1 for v in tc if v < 0.02)
    print(f"  dark targets (mean<0.15): {dark}/{len(tb)} ({100*dark/len(tb):.0f}%)")
    print(f"  grey targets (chroma<0.02): {grey}/{len(tc)} ({100*grey/len(tc):.0f}%)")
    if misaligned:
        print(f"  misaligned base/target pairs: {misaligned}/{len(tb)} "
              f"({100*misaligned/len(tb):.0f}%) -- rebuild with --fresh")

    print(f"\n{'style_key':<34}{'n':>5}{'mean':>8}{'chroma':>9}")
    print("-" * 56)
    rows = sorted(by_key.items(), key=lambda kv: kv[1]["mean"] / kv[1]["n"])
    for k, d in rows:
        m, c = d["mean"] / d["n"], d["chroma"] / d["n"]
        flag = "  <-- dark" if m < 0.15 else ("  <-- grey" if c < 0.02 else "")
        print(f"{k[:34]:<34}{d['n']:5d}{m:8.3f}{c:9.4f}{flag}")

    print()
    if np.mean(tb) < 0.25 or np.mean(tc) < 0.04:
        print("warning: mean target is dark/desaturated; rebalance the look mix "
              "(--max-look-de, fewer effect looks, more expert pairs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--raw-dirs", default="fivek_dng,fivek_full,fivek_hf,raws",
                    help="comma-separated subdirs of data-root to scan for RAW files")
    ap.add_argument("--lut-dir", default="luts")
    ap.add_argument("--out", default="prepared")
    ap.add_argument("--scenes", type=int, default=300)
    ap.add_argument("--looks", type=int, default=12)
    ap.add_argument("--crops", type=int, default=4)
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--expert-dir", default="fivek_experts",
                    help="subdir of data-root holding tiff16_<x> expert retouches")
    ap.add_argument("--experts", default="c",
                    help="which FiveK expert retouches to use as targets (a-e), or '' for none")
    ap.add_argument("--max-mono-frac", type=float, default=0.2,
                    help="max share of monochrome looks (default 0.2). The film-sim pack "
                         "is heavily B&W; too many teaches the director to desaturate "
                         "for any film/camera prompt.")
    ap.add_argument("--look-sampling", choices=["stratified", "mild", "all"],
                    default="stratified",
                    help="stratified: spread across the strength range (keeps style coverage); "
                         "mild: weakest N only (safest for theta-only); all: no selection")
    ap.add_argument("--max-look-de", type=float, default=0.0,
                    help="drop looks whose mean dE from identity exceeds this. Default 0 "
                         "(keep everything): theta is now bounded to physical ranges and the "
                         "dE loss is normalised per-sample difficulty, so strong effect looks "
                         "no longer dominate the gradient or drive degenerate theta. Set e.g. "
                         "20 if you still see extreme theta in eval_prompts.")
    ap.add_argument("--edits", type=int, default=2,
                    help="relative-instruction pairs per crop (warmer/cooler/contrast/etc); 0 disables")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-align", type=float, default=0.5,
                    help="reject scenes whose bayer/render correlation is below this")
    ap.add_argument("--list-sources", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="delete prepared/crops before building (avoids stale crops)")
    ap.add_argument("--audit", default=None,
                    help="audit an existing manifest's target statistics and exit")
    args = ap.parse_args()

    if args.audit:
        audit_manifest(args.audit)
        return

    root = args.data_root
    random.seed(args.seed)
    np.random.seed(args.seed)

    raws = []
    scanned = []
    for d in args.raw_dirs.split(","):
        p = os.path.join(root, d.strip())
        found = find_dngs(p)
        scanned.append((p, os.path.isdir(p), len(found)))
        raws += found
    raws = sorted(set(raws))
    luts = find_haldcluts(os.path.join(root, args.lut_dir))

    print(f"found {len(raws)} RAW files")
    for p, exists, n in scanned:
        print(f"  {'ok ' if exists else 'MISSING'} {p}  ({n} raw)")
    print(f"found {len(luts)} HaldCLUT looks")
    if args.list_sources:
        for f in raws[:10]:
            print("  raw :", f)
        for f in luts[:10]:
            print("  look:", f)
        return

    if not raws:
        print(f"\nNo RAW files in any scanned dir under {root}.")
        try:
            entries = sorted(os.listdir(root))
            print(f"What is actually in {root}: {entries}")
            for e in entries:
                sub = os.path.join(root, e)
                if os.path.isdir(sub):
                    n = len(find_dngs(sub))
                    if n:
                        print(f"  -> {n} RAW files found in '{e}'. Re-run with:")
                        print(f"     python build_dataset.py --raw-dirs {e}")
        except Exception:
            pass
        print("\nIf you have not downloaded RAWs yet:")
        print("  python download_data.py --targets fivek_dng --max-files 200 --max-gb 15")
        sys.exit(1)
    if not luts:
        print(f"\nNo HaldCLUT looks under {os.path.join(root, args.lut_dir)}. Run:")
        print("  python download_data.py --targets haldclut")
        sys.exit(1)

    expert_dirs = {}
    for ex in [e.strip() for e in args.experts.split(",") if e.strip()]:
        d = os.path.join(root, args.expert_dir, f"tiff16_{ex}")
        if os.path.isdir(d):
            expert_dirs[ex] = d
    if expert_dirs:
        print(f"expert retouches: {sorted(expert_dirs)} "
              f"({sum(len(os.listdir(d)) for d in expert_dirs.values())} files)")
    elif args.experts:
        print(f"no expert retouches under {os.path.join(root, args.expert_dir)}; "
              f"get them with:  python download_data.py --targets fivek_experts")

    out_root = os.path.join(root, args.out)
    crops_dir = os.path.join(out_root, "crops")
    if args.fresh and os.path.isdir(crops_dir):
        import shutil
        print(f"--fresh: removing {crops_dir}")
        shutil.rmtree(crops_dir)
    luts_dir = os.path.join(out_root, "luts")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(luts_dir, exist_ok=True)

    random.shuffle(raws)
    raws = raws[:args.scenes]

    print(f"\nscreening {len(luts)} candidate looks ...")
    scored = []
    for lp in luts:
        name = look_name(lp)
        npy_path = os.path.join(luts_dir, f"{name}.npy")
        try:
            if os.path.exists(npy_path):
                cube = np.load(npy_path)
            else:
                cube = resample_cube(hald_to_cube(lp))
                np.save(npy_path, cube)
            scored.append((look_strength(cube), name, cube, npy_path,
                           look_mono_score(cube)))
        except Exception as e:
            print(f"  skip {name}: {e}")
    if not scored:
        print("no usable looks")
        sys.exit(1)

    scored.sort(key=lambda r: r[0])
    s_vals = [r[0] for r in scored]
    print(f"  look strength (mean dE from identity) over {len(scored)} looks:")
    print(f"    min {s_vals[0]:.1f}  p25 {np.percentile(s_vals,25):.1f}  "
          f"median {np.percentile(s_vals,50):.1f}  p75 {np.percentile(s_vals,75):.1f}  "
          f"max {s_vals[-1]:.1f}")
    if args.max_look_de and args.max_look_de > 0:
        keep = [r for r in scored if r[0] <= args.max_look_de]
        dropped = [r for r in scored if r[0] > args.max_look_de]
        print(f"  --max-look-de {args.max_look_de}: keeping {len(keep)}/{len(scored)}")
        if dropped:
            print(f"    dropped as effects (theta can only mimic these by going degenerate): "
                  f"{[(round(d[0],1), d[1][:22]) for d in dropped[-3:]]}")
        if not keep:
            print("    !! nothing survived the cut; raise --max-look-de")
        scored = keep or scored[:args.looks]

    # How to choose which looks to train on.
    #
    # "mild" keeps the weakest N. That is the safe option for theta-only training, because
    # strongest looks versus 3-4 dE on mild ones -- those targets are not representable, so
    # their gradient is closer to label noise than to a hard example.
    #
    # "stratified" (default) instead spreads the choice evenly across the strength range.
    # Dropping every strong look narrows the style range the director has ever seen, which
    # is a real generalisation cost; stratifying keeps coverage while stopping the
    # unrepresentable end from dominating the loss. Per-look strength is written into the
    # manifest so evaluation can report results per strength bucket.
    n_keep = min(args.looks, len(scored))
    if args.look_sampling == "all":
        chosen = scored
    elif args.look_sampling == "mild":
        chosen = scored[:n_keep]
    else:
        idx = np.linspace(0, len(scored) - 1, n_keep).round().astype(int)
        chosen = [scored[i] for i in sorted(set(idx.tolist()))]

    cubes = {}
    # Cap the monochrome share. Without this, a pack that is ~40% B&W teaches the director
    # that film/camera prompts mean desaturate, and unrelated prompts render grey.
    mono = [c for c in chosen if c[4] >= 0.6]
    colour = [c for c in chosen if c[4] < 0.6]
    max_mono = max(1, int(round(args.max_mono_frac * len(chosen)))) if chosen else 0
    if len(mono) > max_mono:
        pool = [c for c in scored if c[4] < 0.6 and c not in colour]
        dropped_mono = mono[max_mono:]
        mono = mono[:max_mono]
        need = len(dropped_mono)
        colour = colour + pool[:need]
        print(f"  mono cap: {len(dropped_mono)} B&W looks replaced with colour looks "
              f"({max_mono}/{len(chosen)} mono allowed)")
    chosen = sorted(mono + colour, key=lambda c: c[0])

    for strength, name, cube, npy_path, mscore in chosen:
        cubes[name] = (cube, npy_path, strength)
        tag = "  [B&W]" if mscore >= 0.6 else ""
        print(f"  keep {strength:6.1f}  mono {mscore:.2f}  {name}{tag}")
    kept = [c[0] for c in chosen]
    print(f"  {len(cubes)} looks, strength {min(kept):.1f}-{max(kept):.1f} "
          f"(sampling: {args.look_sampling})")
    hard = sum(1 for k in kept if k > 15)
    if hard:
        print(f"  note: {hard} of them are above 15 dE. theta alone will not reach those; "
              f"they are kept for style coverage. Use --use_lut for the colour twists.")

    from PIL import Image
    try:
        import tifffile  # noqa: F401
    except ImportError:
        if expert_dirs:
            print("\nwarning: tifffile not installed; 16-bit expert TIFFs may load "
                   "incorrectly. pip install tifffile\n")
    manifest_path = os.path.join(root, "manifest.jsonl")
    n_written = 0
    n_edits = 0
    n_expert = 0
    n_bad = 0
    n_misaligned = 0
    align_scores = []
    cameras = set()
    cs = args.crop_size

    with open(manifest_path, "w") as mf:
        for si, rp in enumerate(raws):
            scene = os.path.splitext(os.path.basename(rp))[0]
            try:
                loaded = load_raw(rp)
            except Exception as e:
                print(f"[{si+1}/{len(raws)}] skip {scene}: {e}")
                continue
            if loaded is None:
                print(f"[{si+1}/{len(raws)}] skip {scene}: unsupported raw layout")
                continue
            bayer, base, ccm, _ = loaded
            align = alignment_score(bayer, base)
            if align < args.min_align:
                n_misaligned += 1
                print(f"[{si+1}/{len(raws)}] SKIP {scene}: alignment {align:.3f} "
                      f"< {args.min_align} (bayer and render do not match)")
                continue
            align_scores.append(align)
            exif = read_exif(rp)
            cameras.add(exif["camera"])
            H, W = bayer.shape
            if H < cs or W < cs:
                print(f"[{si+1}/{len(raws)}] skip {scene}: too small ({H}x{W})")
                continue

            positions = []
            for _ in range(args.crops):
                y = random.randrange(0, (H - cs) // 2 + 1) * 2
                x = random.randrange(0, (W - cs) // 2 + 1) * 2
                positions.append((y, x))

            for look, (cube, npy_path, strength) in cubes.items():
                for k, (y, x) in enumerate(positions):
                    bc = bayer[y:y + cs, x:x + cs]
                    tc = base[y:y + cs, x:x + cs]
                    base_path = os.path.join(crops_dir, f"{scene}__base__{k}.jpg")
                    # ALWAYS overwrite. Skipping when the file exists left stale crops from a
                    # previous build beside freshly generated targets: crop positions come from
                    # the RNG stream, which shifts whenever the number of random calls changes,
                    # so base and target ended up showing DIFFERENT pixels of the scene. That is
                    # unfittable by any colour transform and inflates every metric.
                    Image.fromarray((tc * 255).astype(np.uint8)).save(base_path, quality=95)
                    styled = apply_cube(tc, cube)
                    try:
                        validate_target(styled, f"{scene}/{look}")
                    except RuntimeError as e:
                        print(f"    skip degenerate target: {e}")
                        n_bad += 1
                        continue
                    stem = f"{scene}__{look}__{k}"
                    npz_path = os.path.join(crops_dir, stem + ".npz")
                    jpg_path = os.path.join(crops_dir, stem + ".jpg")
                    np.savez_compressed(npz_path,
                                        bayer=bc.astype(np.float16),
                                        ccm=ccm.astype(np.float32))
                    Image.fromarray((styled * 255).astype(np.uint8)).save(jpg_path, quality=95)
                    mf.write(json.dumps({
                        "raw_path": npz_path,
                        "jpeg_path": jpg_path,
                        "base_path": base_path,
                        "lut_path": npy_path,
                        "camera": exif["camera"],
                        "prompt": random.choice(make_prompts(look)),
                        "style_key": look,
                        "look_de": round(float(strength), 2),
                        "iso": exif["iso"],
                        "exposure": exif["exposure"],
                        "fnumber": exif["fnumber"],
                        "focal_length": exif["focal_length"],
                    }) + "\n")
                    n_written += 1

            # expert-retouch pairs: real human edits of this exact photo
            for ex in expert_dirs:
                tif = os.path.join(expert_dirs[ex], f"{scene}.tif")
                if not os.path.exists(tif):
                    continue
                try:
                    exp_full = load_expert_tiff(tif, (H, W))
                except Exception as e:
                    print(f"    expert {ex} failed: {e}")
                    continue
                for k, (y, x) in enumerate(positions):
                    bc = bayer[y:y + cs, x:x + cs]
                    tc = base[y:y + cs, x:x + cs]
                    ec = exp_full[y:y + cs, x:x + cs]
                    base_path = os.path.join(crops_dir, f"{scene}__base__{k}.jpg")
                    # ALWAYS overwrite. Skipping when the file exists left stale crops from a
                    # previous build beside freshly generated targets: crop positions come from
                    # the RNG stream, which shifts whenever the number of random calls changes,
                    # so base and target ended up showing DIFFERENT pixels of the scene. That is
                    # unfittable by any colour transform and inflates every metric.
                    Image.fromarray((tc * 255).astype(np.uint8)).save(base_path, quality=95)
                    stem = f"{scene}__expert_{ex}__{k}"
                    npz_path = os.path.join(crops_dir, stem + ".npz")
                    jpg_path = os.path.join(crops_dir, stem + ".jpg")
                    np.savez_compressed(npz_path, bayer=bc.astype(np.float16),
                                        ccm=ccm.astype(np.float32))
                    Image.fromarray((ec * 255).astype(np.uint8)).save(jpg_path, quality=95)
                    mf.write(json.dumps({
                        "raw_path": npz_path, "jpeg_path": jpg_path, "base_path": base_path,
                        "camera": exif["camera"],
                        "prompt": random.choice(EXPERT_PROMPTS.get(ex, ["professional retouch"])),
                        "style_key": f"expert_{ex}",
                        "iso": exif["iso"], "exposure": exif["exposure"],
                        "fnumber": exif["fnumber"], "focal_length": exif["focal_length"],
                    }) + "\n")
                    n_written += 1
                    n_expert += 1

            # relative-instruction pairs, once per crop (not per look)
            for k, (y, x) in enumerate(positions):
                if args.edits <= 0:
                    break
                bc = bayer[y:y + cs, x:x + cs]
                tc = base[y:y + cs, x:x + cs]
                base_path = os.path.join(crops_dir, f"{scene}__base__{k}.jpg")
                Image.fromarray((tc * 255).astype(np.uint8)).save(base_path, quality=95)
                for styled, prompt, ekey in make_edit_pairs(tc, random, n=args.edits):
                    stem = f"{scene}__{ekey}__{k}"
                    npz_path = os.path.join(crops_dir, stem + ".npz")
                    jpg_path = os.path.join(crops_dir, stem + ".jpg")
                    np.savez_compressed(npz_path, bayer=bc.astype(np.float16),
                                        ccm=ccm.astype(np.float32))
                    Image.fromarray((styled * 255).astype(np.uint8)).save(jpg_path, quality=95)
                    mf.write(json.dumps({
                        "raw_path": npz_path,
                        "jpeg_path": jpg_path,
                        "base_path": base_path,
                        "camera": exif["camera"],
                        "prompt": prompt,
                        "style_key": ekey,
                        "iso": exif["iso"],
                        "exposure": exif["exposure"],
                        "fnumber": exif["fnumber"],
                        "focal_length": exif["focal_length"],
                    }) + "\n")
                    n_written += 1
                    n_edits += 1
            print(f"[{si+1}/{len(raws)}] {scene} -> {len(cubes)*len(positions)} pairs "
                  f"({exif['camera']}, align {align:.3f})")

    print(f"\nwrote {n_written} pairs to {manifest_path}  ({n_edits} relative-instruction, {n_expert} expert)")
    if n_bad:
        print(f"  skipped {n_bad} degenerate targets")
    if align_scores:
        a = np.array(align_scores)
        print(f"alignment: mean {a.mean():.3f}  min {a.min():.3f}  "
              f"(1.0 = perfect, <0.5 rejected)")
    if n_misaligned:
        print(f"REJECTED {n_misaligned} misaligned scenes")
    print(f"looks (style_key): {len(cubes)}")
    print(f"distinct camera bodies: {len(cameras)}")
    for c in sorted(cameras)[:15]:
        print(f"  - {c}")
    print(f"\nnext:\n  python camera_look_isp.py --gpu48 --phase frontend --manifest {manifest_path}")


if __name__ == "__main__":
    main()
