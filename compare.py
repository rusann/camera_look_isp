"""
Comparison against baselines on the standard FiveK expert-C protocol.

The published LUT-enhancement numbers (Zeng et al., AdaInt, SepLUT, CLUT-Net, Kosugi) are
all measured on one specific setup: FiveK, expert C as the target, short edge 480 px, a
fixed 4500/500 split, and -- critically -- a SPECIFIC 8-bit sRGB input render, not a RAW
decode. Reproducing the resolution but not the input is what makes a protocol check fail,
so --fivek-480p reads the authors' preprocessed pairs directly.

Design: PREDICTIONS ARE CACHED, not just metrics. Every row writes its output per image to
--work, keyed by a fingerprint of that row's configuration. Consequences:
  * adding or changing a metric       -> zero recomputation, only rescoring
  * changing one baseline's settings  -> only that row recomputes
  * changing the protocol or the input-> everything recomputes (it is a different task)
Run with --dry-run first: it prints what is cached and what will be computed.

  # one-time, comparable to the literature
  python compare.py --fivek-480p /mnt/data/FiveK --out_dir release/checkpoints_lut \
      --style_cache_dir release/style_cache_lut --use_lut --ip2p --work compare_work

  # no download: our own DNG render, reusing an existing decoded-pair cache
  python compare.py --raw-dir /mnt/data/fivek_dng \
      --expert-dir /mnt/data/fivek_experts/tiff16_c --pair-cache .pair_cache \
      --out_dir release/checkpoints_lut --style_cache_dir release/style_cache_lut \
      --use_lut --ip2p --work compare_work

Rows are grouped by what each is allowed to see, the only axis on which they compare:
  training-free   identity, Reinhard (oracle statistics, weakest bound)
  closed-set      matrix+gamma (the two-stage composition), global LUT (Zeng report 20.37 dB
                  for this setting, so it doubles as a protocol check)
  open-set        ours, ours (cached style), InstructPix2Pix / MagicBrush
  oracle bounds   fitted theta, per-channel curves, full 3D LUT
"""

import os, io, re, json, glob, math, time, random, hashlib, argparse, contextlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import camera_look_isp as C
import bench_distill as B

# ============================== published numbers ==============================
# FiveK, expert C, 480p. dE is the EUCLIDEAN distance in CIE Lab (CIE76, "dE_ab") in every
# one of these papers -- NOT CIEDE2000. They are different quantities and dE_ab runs higher,
# so this script measures both and reports them in separate columns.
# White-box rows are Kosugi's (ACM MM 2024) re-runs on this protocol, not our own.
PUBLISHED = [
    ("UPE",                  "wang2019upe",             "none (closed-set)", 21.88, 0.853, 10.80),
    ("Single fixed LUT",     "zeng2020learning3dlut",   "none (closed-set)", 20.37, 0.852, None),
    ("DPE",                  "chen2018dpe",             "none (closed-set)", 23.75, 0.908,  9.34),
    ("HDRNet",               "gharbi2017deepbilateral", "none (closed-set)", 24.66, 0.915,  8.06),
    ("DeepLPF",              "moran2020deeplpf",        "none (closed-set)", 24.73, 0.916,  7.99),
    ("CSRNet",               "he2020csrnet",            "none (closed-set)", 25.17, 0.921,  7.75),
    ("3D LUT",               "zeng2020learning3dlut",   "none (closed-set)", 25.29, 0.920,  7.55),
    ("ICELUT",               "yang2024icelut",          "none (closed-set)", 25.27, 0.918,  7.51),
    ("SepLUT",               "yang2022seplut",          "none (closed-set)", 25.47, 0.921,  7.54),
    ("AdaInt",               "yang2022adaint",          "none (closed-set)", 25.49, 0.926,  7.47),
    ("CLUT-Net",             "zhang2022clut",           "none (closed-set)", 25.55, 0.931,  7.50),
]
PUBLISHED_WHITEBOX = [
    ("Distort-and-Recover",  "park2018drl",    "none (closed-set)", 23.86, 0.903, 9.07),
    ("UIE",                  "kosugi2020uie",  "none (closed-set)", 24.74, 0.923, 8.06),
    ("RSFNet",               "ouyang2023rsfnet", "none (closed-set)", 24.86, 0.924, 7.89),
    ("Exposure",             "hu2018exposure", "none (closed-set)", 25.04, 0.920, 7.83),
    ("PG-IA-NILUT",          "kosugi2024",     "text prompt",       25.22, 0.930, 7.76),
]
PUBLISHED_OTHER = [
    ("PixTalk (ICCV 2025)",      "text instruction",        "own benchmark, RAW/sRGB pairs"),
    ("MonetGPT (TOG 2025)",      "instruction, MLLM agent", "own benchmark, PSNR 23.10 / SSIM 0.82"),
    ("RetouchLLM (2025)",        "instruction, train-free", "own benchmark"),
    ("NILUT (2024)",             "style vector",            "FiveK, reproduces 3D LUT outputs"),
]

# slug -> (label, group, conditioning)
ROWS = {
    "identity":     ("identity",                     "training-free", "—"),
    "reinhard":     ("Reinhard (Lab mean/std)",      "training-free", "oracle statistics"),
    "matgamma":     ("matrix 3x3 + gamma",           "closed-set",    "training split"),
    "globallut":    ("global LUT",                   "closed-set",    "training split"),
    "editor":       ("InstructPix2Pix",              "open-set",      "text instruction"),
    "ours":         ("ours",                         "open-set",      "text prompt"),
    "ours_cached":  ("ours (cached style)",          "open-set",      "text prompt, cached"),
    "fit_theta":    ("ours (fitted theta)",          "oracle",        "target"),
    "oracle_curve": ("oracle curves (per-channel)",  "oracle",        "target"),
    "oracle_lut":   ("oracle LUT",                   "oracle",        "target"),
}
GROUP_LABEL = {"training-free": "Training-free", "closed-set": "Closed-set, fitted on train",
               "open-set": "Open-set, language-conditioned", "oracle": "Oracle bounds (not methods)"}


def fp(obj, n=10):
    """Stable short fingerprint of a configuration dict."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


# ============================== metrics ==============================

def delta_e_ab(a, b):
    """CIE76: plain L2 in Lab. This is the 'dE' reported by Zeng et al., AdaInt, SepLUT,
    CLUT-Net and Kosugi. Reporting our CIEDE2000 against their CIE76 would compare two
    different quantities, and since dE_ab is systematically larger it would flatter us."""
    l1, l2 = C.rgb_to_lab(a.clamp(0, 1)), C.rgb_to_lab(b.clamp(0, 1))
    return float((l1 - l2).pow(2).sum(1).clamp(min=1e-12).sqrt().mean())


@torch.no_grad()
def score_all(pred, tgt, inp, lp):
    pred = pred.clamp(0, 1)
    return {"psnr": B.psnr(pred, tgt), "ssim": B.ssim(pred, tgt),
            "de00": float(C.delta_e00(pred, tgt)), "deab": delta_e_ab(pred, tgt),
            "lpips": float(lp(pred, tgt)), "drift": float(C.delta_e00(pred, inp))}


# ============================== colour helpers ==============================

def lab_to_rgb(lab):
    """Inverse of C.rgb_to_lab (D65, sRGB primaries and transfer). The forward direction is
    all camera_look_isp ships; keep the white point identical or the round trip adds an
    error that would be charged to the Reinhard baseline."""
    L, A, Bc = lab.unbind(1)
    fy = (L + 16.0) / 116.0
    fx, fz = fy + A / 500.0, fy - Bc / 200.0

    def finv(t):
        d = 6.0 / 29.0
        return torch.where(t > d, t.clamp(min=0) ** 3, 3 * d * d * (t - 4.0 / 29.0))

    x, y, z = 0.95047 * finv(fx), 1.0 * finv(fy), 1.08883 * finv(fz)
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    lin = torch.stack([r, g, b], 1).clamp(0, 1)
    return torch.where(lin <= 0.0031308, 12.92 * lin,
                       1.055 * lin.clamp(min=0.0031308) ** (1 / 2.4) - 0.055).clamp(0, 1)


@torch.no_grad()
def reinhard_transfer(x, y):
    """Reinhard et al. 2001: match per-channel mean and std in Lab. Instant, training-free,
    and an oracle -- but the weakest one here, bounding a single global affine map in Lab.
    Anything that fails to beat it is not doing useful work."""
    lx, ly = C.rgb_to_lab(x), C.rgb_to_lab(y)
    mx, sx = lx.mean((2, 3), keepdim=True), lx.std((2, 3), keepdim=True).clamp(min=1e-4)
    my, sy = ly.mean((2, 3), keepdim=True), ly.std((2, 3), keepdim=True)
    return lab_to_rgb((lx - mx) / sx * sy + my)


def subsample(x, y, n=16384, seed=0):
    """Scatter a random pixel subset into a square tensor.

    Valid ONLY for per-pixel fits: it destroys spatial structure by construction. Every
    oracle here is a colour map (theta with pointwise_only, 1D curves, 3D LUT), so a pixel
    subset carries the same information as the full image at a fraction of the cost. The fit
    runs on the subset; scoring always runs on the full image.
    """
    b, c, h, w = x.shape
    N = h * w
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(N, generator=g)[:min(n, N)]
    s = int(math.isqrt(idx.numel()))
    idx = idx[:s * s].to(x.device)
    return (x.reshape(b, c, N)[:, :, idx].reshape(b, c, s, s),
            y.reshape(b, c, N)[:, :, idx].reshape(b, c, s, s))


def apply_curves(curves, x):
    """Per-channel 1D map, linear interpolation. curves (3,K) in [0,1]."""
    K = curves.shape[-1]
    pos = x.clamp(0, 1) * (K - 1)
    i0 = pos.floor().long().clamp(0, K - 2)
    t = pos - i0.float()
    out = []
    for c in range(3):
        cv, idx = curves[c], i0[:, c]
        out.append(cv[idx] + t[:, c] * (cv[idx + 1] - cv[idx]))
    return torch.stack(out, 1).clamp(0, 1)


def fit_curves(x, y, device, K=33, iters=200, lr=0.06):
    """Fit a FREE (not monotone) per-channel map. As an upper bound a looser family is
    correct: the quantity wanted is 'what can ANY channel-separable map reach', so the gap
    to the oracle 3D LUT is exactly the cross-channel part of the look."""
    lin = torch.linspace(0, 1, K, device=device)
    curves = lin[None].repeat(3, 1).clone().requires_grad_(True)
    opt = torch.optim.Adam([curves], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    for _ in range(iters):
        opt.zero_grad()
        F.l1_loss(apply_curves(curves.clamp(0, 1), x), y).backward()
        opt.step(); sch.step()
    return curves.detach().clamp(0, 1)


def fit_lut(pairs, device, n=33, iters=400, lr=0.04, tv=0.3):
    """Fit one 3D LUT by gradient descent over all pairs at once.

    The smoothness term uses the SECOND difference. The first difference of an identity LUT
    is a nonzero constant, so penalising it fights the identity itself; curvature is zero for
    identity, so this penalises only departure from a smooth ramp. It is not cosmetic: only
    cells visited by training pixels get gradient, and unregularised neighbours drift apart,
    which shows up as banding and destroys LPIPS/SSIM on test colours that land there. Set
    tv=0 for a single-image oracle fit, where smoothing would understate the bound.
    """
    lin = torch.linspace(0, 1, n, device=device)
    ident = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij")[::-1], 0)[None]
    lut = ident.clone().requires_grad_(True)
    opt = torch.optim.Adam([lut], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    for _ in range(iters):
        opt.zero_grad()
        loss = sum(F.l1_loss(C.lut3d_apply(lut.clamp(0, 1), a), b) for a, b in pairs) / len(pairs)
        if tv > 0:
            loss = loss + tv * sum(torch.diff(lut, n=2, dim=k).pow(2).mean() for k in (2, 3, 4))
        loss.backward()
        opt.step(); sch.step()
    return lut.detach().clamp(0, 1)


def apply_matrix_gamma(x, p):
    M, lgain, lgam = p
    lin = C.srgb_to_linear(x.clamp(0, 1))
    lin = torch.einsum("ij,bjhw->bihw", M, lin)
    lin = (lin * lgain.exp()).clamp(1e-6, 1.0)
    return (lin ** (1.0 / (2.4 * lgam.exp()).clamp(1.0, 6.0))).clamp(0, 1)


def fit_matrix_gamma(pairs, device, iters=400, lr=0.02, batch=8):
    """White-box two-stage baseline: 3x3 colour matrix + per-channel gamma, one fixed setting
    for all test images. This is the honest name for 'neutral render -> fixed colour
    correction', i.e. the composition of existing components a reviewer will ask about."""
    M = torch.eye(3, device=device).clone().requires_grad_(True)
    lgain = torch.zeros(1, 3, 1, 1, device=device, requires_grad=True)
    lgam = torch.zeros(1, 3, 1, 1, device=device, requires_grad=True)
    opt = torch.optim.Adam([M, lgain, lgam], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        sub = pairs if len(pairs) <= batch else random.sample(pairs, batch)
        loss = sum(F.l1_loss(apply_matrix_gamma(a, (M, lgain, lgam)), b) for a, b in sub) / len(sub)
        loss.backward()
        opt.step()
    return (M.detach(), lgain.detach(), lgam.detach())


def fit_theta(x_lin, y, isp, cfg, device, iters=200, lr=0.06):
    """Fit theta with pointwise_only=True -- the same tract that bakes into one table.

    camera_look_isp.fit_look leaves unsharp_mask and chroma NR active, and since its loss is
    L1 + colour with no perceptual term, it drives sharpening to the maximum: that is why a
    fitted-theta row can show an excellent PSNR alongside an LPIPS worse than identity. The
    LUT methods compared against have no sharpening at all, so restricting to the per-pixel
    path is also the only apples-to-apples choice.
    """
    th = torch.zeros(1, cfg.theta_dim, device=device, requires_grad=True)
    opt = torch.optim.Adam([th], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    best, best_th = float("inf"), th.detach().clone()
    for _ in range(iters):
        opt.zero_grad()
        pred = isp(x_lin, th, None, pointwise_only=True)
        loss = F.l1_loss(pred, y) + 0.1 * C.delta_e00(pred, y)
        loss.backward()
        opt.step(); sch.step()
        if float(loss) < best:
            best, best_th = float(loss), th.detach().clone()
    return best_th


# ============================== editors ==============================

EDITOR_CKPT = {"ip2p": ("timbrooks/instruct-pix2pix", "InstructPix2Pix"),
               "magicbrush": ("vinesmsuic/magicbrush-jul7", "MagicBrush")}


def load_editor(which, device, offload=False):
    from diffusers import StableDiffusionInstructPix2PixPipeline
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        EDITOR_CKPT[which][0], torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_model_cpu_offload() if offload else pipe.to(device)
    with contextlib.suppress(Exception):
        pipe.enable_attention_slicing()
    return pipe


@torch.no_grad()
def run_editor(pipe, img, prompt, steps, tcfg, icfg, seed=0):
    """The UNet needs dimensions divisible by 8; round, then resize back, since comparing at
    a different resolution than every other row would be meaningless."""
    w, h = img.size
    w8, h8 = max(8, (w // 8) * 8), max(8, (h // 8) * 8)
    inp = img if (w8, h8) == (w, h) else img.resize((w8, h8), Image.LANCZOS)
    out = pipe(prompt, image=inp, num_inference_steps=steps, guidance_scale=tcfg,
               image_guidance_scale=icfg,
               generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    return out if out.size == img.size else out.resize(img.size, Image.LANCZOS)


def fivek_480p(root, split, n_train=4500):
    inp = os.path.join(root, "input", "JPG", "480p")
    tgt = os.path.join(root, "expertC", "JPG", "480p")
    ex = (".png", ".jpg", ".jpeg")
    si = {os.path.splitext(f)[0]: os.path.join(inp, f)
          for f in os.listdir(inp) if f.lower().endswith(ex)}
    st = {os.path.splitext(f)[0]: os.path.join(tgt, f)
          for f in os.listdir(tgt) if f.lower().endswith(ex)}
    common = sorted(set(si) & set(st))
    p = os.path.join(root, f"{split}.txt")
    if os.path.exists(p):
        names = [n for n in (os.path.splitext(os.path.basename(l.strip()))[0]
                             for l in open(p) if l.strip()) if n in si and n in st]
    else:
        idx = {s: (int(m.group(1)) if (m := re.match(r"a(\d{4})", s)) else 0) for s in common}
        names = [s for s in common if (idx[s] <= n_train) == (split == "train")]
        if not names and split == "test":
            names = common[int(len(common) * 0.9):]
        print(f"  [{split}] no {split}.txt; reconstructed, {len(names)} pairs")
    return [(n, si[n], st[n]) for n in names]



def raw_pairs(raw_dir, expert_dir):
    experts = {os.path.splitext(f)[0]: os.path.join(expert_dir, f)
               for f in os.listdir(expert_dir) if f.lower().endswith((".tif", ".tiff"))}
    raws = {}
    for e in B.RAW_EXT:
        for p in glob.glob(os.path.join(raw_dir, "*" + e)):
            raws[os.path.splitext(os.path.basename(p))[0]] = p
    names = sorted(set(experts) & set(raws))
    rng = random.Random(0); rng.shuffle(names)
    return [(nm, raws[nm], experts[nm]) for nm in names]


def decode_pair(a_path, b_path, side, mode, is_raw):
    if not is_raw:
        return Image.open(a_path).convert("RGB"), Image.open(b_path).convert("RGB")
    img = B.load_input_image(a_path)
    tgt = Image.open(b_path)
    try:
        import tifffile
        arr = tifffile.imread(b_path)
        if arr.ndim == 3:
            arr = arr[..., :3]
            arr = (arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
                   if np.issubdtype(arr.dtype, np.integer) else arr.astype(np.float32))
            tgt = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    except Exception:
        tgt = tgt.convert("RGB")

    def rs(im):
        w, h = im.size
        s = side / (min(w, h) if mode == "short" else max(w, h))
        return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)

    img, tgt = rs(img.convert("RGB")), rs(tgt.convert("RGB"))
    if img.size != tgt.size:
        tgt = tgt.resize(img.size, Image.LANCZOS)
    return img, tgt


def get_pair(nm, a_path, b_path, pair_cache, side, mode, is_raw):
    """Decoded pairs cached as PNG (lossless, so no added error). RAW decode is ~1 s/file
    and dominates a rerun otherwise."""
    if pair_cache:
        ia, ib = os.path.join(pair_cache, f"{nm}_in.png"), os.path.join(pair_cache, f"{nm}_tgt.png")
        if os.path.exists(ia) and os.path.exists(ib):
            return Image.open(ia).convert("RGB"), Image.open(ib).convert("RGB")
    a, b = decode_pair(a_path, b_path, side, mode, is_raw)
    if pair_cache:
        os.makedirs(pair_cache, exist_ok=True)
        a.save(os.path.join(pair_cache, f"{nm}_in.png"))
        b.save(os.path.join(pair_cache, f"{nm}_tgt.png"))
    return a, b


def to_t(im, device):
    return torch.from_numpy(np.array(im, np.float32) / 255.).permute(2, 0, 1)[None].to(device)


def save_pred(t, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


# ============================== optional: LUT representation ==============================

def lut_repr_report(cfg, lut_dir, images, device):
    """Our VQ-VAE vs NILUT as compact representations of N real 3D LUTs.

    NILUT cannot be evaluated on expert C -- it encodes artistic LUTs, not a retouch -- but
    the TASK is identical to our LUT head's: represent N given 3D LUTs compactly and
    reproduce their output on real images, measured as dE to the real LUT's output. That
    makes it the one directly comparable point for the representation itself.
    Published NILUT (MLP-Res 128x2, 3 styles): dE 1.65 at PSNR 42.04 on 100 MIT5K images.
    """
    ck = os.path.join(cfg.out_dir, "lut_vqvae.pt")
    paths = sorted(glob.glob(os.path.join(lut_dir, "*.npy")))
    if not (os.path.exists(ck) and paths):
        return []
    vq = C.LUTVQVAE(cfg).to(device).eval()
    vq.load_state_dict(torch.load(ck, map_location=device))
    des, ps = [], []
    for p in paths:
        real = torch.from_numpy(np.load(p).astype(np.float32))[None].to(device)
        with torch.no_grad():
            rec = vq(real)[0]
            for im in images:
                x = to_t(im, device)
                a = C.lut3d_apply(real, x).clamp(0, 1)
                b = C.lut3d_apply(rec, x).clamp(0, 1)
                des.append(float(C.delta_e00(a, b)))
                ps.append(B.psnr(b, a))
    bits = cfg.lut_tokens * math.log2(cfg.lut_codebook)
    dec = sum(q.numel() for q in vq.parameters())
    return ["", "## LUT representation: ours vs NILUT\n",
            "| representation | styles | cost per style | shared | dE | PSNR |",
            "|---|---|---|---|---|---|",
            f"| NILUT MLP-Res 128x2 (published) | 3 | 33.9K params / 3 styles | — | 1.65 | 42.04 |",
            f"| CNILUT 256x3 (published) | 5 | 199K params / 5 styles | — | 0.80-1.19 | — |",
            f"| ours (VQ-VAE) | {len(paths)} | {bits/8:.0f} B of tokens | "
            f"{dec/1e6:.1f}M decoder | {np.mean(des):.2f} | {np.mean(ps):.2f} |",
            "",
            "NILUT spends one network per 3-5 styles; here one shared decoder covers every",
            "style and each costs only its token string. The published dE is CIEDE2000 on",
            "MIT5K images in both cases, so the numbers are directly comparable."]


# ============================== main ==============================

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_argument_group("input (choose one)")
    src.add_argument("--fivek-480p", default=None,
                     help="root of the authors' preprocessed FiveK: input/JPG/480p, "
                          "expertC/JPG/480p, train.txt, test.txt. REQUIRED for the published "
                          "numbers to be comparable.")
    src.add_argument("--raw-dir", default=None)
    src.add_argument("--expert-dir", default=None, help="tiff16_c directory")
    ap.add_argument("--out_dir", default="checkpoints_lut")
    ap.add_argument("--style_cache_dir", default="style_cache_lut")
    ap.add_argument("--use_lut", action="store_true")
    ap.add_argument("--work", default="compare_work", help="prediction cache (enables resume)")
    ap.add_argument("--pair-cache", default=None, help="decoded-pair PNG cache")
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--n-fit", type=int, default=40)
    ap.add_argument("--side", type=int, default=480)
    ap.add_argument("--side-mode", choices=["short", "long"], default="short")
    ap.add_argument("--prompt", default="expert C retouch")
    ap.add_argument("--cache-key", default="expert_c", help="style_cache entry; '' to skip")
    ap.add_argument("--rows", default=None, help="comma-separated subset of " + ",".join(ROWS))
    ap.add_argument("--force", default=None, help="comma-separated rows to recompute")
    ap.add_argument("--dry-run", action="store_true", help="report cache state and exit")
    ap.add_argument("--report", default="comparison")
    ap.add_argument("--fit-pixels", type=int, default=16384, help="pixels per oracle fit")
    ap.add_argument("--oracle-iters", type=int, default=200)
    ap.add_argument("--editor", choices=["none", "ip2p", "magicbrush"], default="none")
    ap.add_argument("--ip2p", action="store_true", help="alias for --editor ip2p")
    ap.add_argument("--editor-prompt", default="make it look like a professionally retouched photograph")
    ap.add_argument("--editor-steps", type=int, default=20)
    ap.add_argument("--editor-text-cfg", type=float, default=7.5)
    ap.add_argument("--editor-image-cfg", type=float, default=1.5)
    ap.add_argument("--editor-offload", action="store_true")
    ap.add_argument("--lut-dir", default=None, help="dir of *.npy LUTs for the representation table")
    args = ap.parse_args()

    if args.ip2p and args.editor == "none":
        args.editor = "ip2p"
    if not args.fivek_480p and not (args.raw_dir and args.expert_dir):
        ap.error("pass --fivek-480p, or both --raw-dir and --expert-dir")

    C.quiet_third_party()
    cfg = C.Config()
    cfg.use_4bit = cfg.use_8bit_adam = False
    cfg.out_dir, cfg.style_cache_dir = args.out_dir, args.style_cache_dir
    if args.use_lut:
        cfg.use_lut = True
    C.sync_for(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start, timings = time.time(), {}

    # ---- pairs ---------------------------------------------------------------
    is_raw = args.fivek_480p is None
    if is_raw:
        allp = raw_pairs(args.raw_dir, args.expert_dir)
        fit_list, test_list = allp[:args.n_fit], allp[args.n_fit:args.n_fit + args.n_test]
        proto = {"mode": "rawpy", "side": args.side, "edge": args.side_mode}
        print("WARNING: input is our own DNG render, NOT the protocol's sRGB render.\n"
              "  The published block will be reported separately as non-comparable.\n"
              "  Pass --fivek-480p to fix this (SepLUT repo hosts the preprocessed set).")
    else:
        fit_list = fivek_480p(args.fivek_480p, "train")[:args.n_fit]
        test_list = fivek_480p(args.fivek_480p, "test")[:args.n_test]
        proto = {"mode": "fivek480p", "root": os.path.abspath(args.fivek_480p)}
    if not test_list:
        print("no test pairs found; check the paths"); return
    pair_fp = fp(proto)

    # ---- row fingerprints: what invalidates what -----------------------------
    oracle_cfg = {"px": args.fit_pixels, "it": args.oracle_iters}
    row_cfg = {
        "identity":     {},
        "reinhard":     {},
        "matgamma":     {"nfit": len(fit_list), "it": 400},
        "globallut":    {"nfit": len(fit_list), "it": 800, "tv": 0.3, "n": 33},
        "editor":       {"m": args.editor, "p": args.editor_prompt, "s": args.editor_steps,
                         "t": args.editor_text_cfg, "i": args.editor_image_cfg},
        "ours":         {"ckpt": os.path.abspath(cfg.out_dir), "p": args.prompt,
                         "lut": cfg.use_lut, "pw": True},
        "ours_cached":  {"cache": os.path.abspath(cfg.style_cache_dir), "k": args.cache_key,
                         "lut": cfg.use_lut, "pw": True},
        "fit_theta":    {**oracle_cfg, "pw": True},
        "oracle_curve": {**oracle_cfg, "K": 33},
        "oracle_lut":   {**oracle_cfg, "n": 33, "tv": 0.0},
    }
    want = [r for r in ROWS if (args.rows is None or r in args.rows.split(","))]
    if args.editor == "none" and "editor" in want:
        want.remove("editor")
    if not args.cache_key and "ours_cached" in want:
        want.remove("ours_cached")
    forced = set((args.force or "").split(",")) - {""}
    rfp = {r: fp({"pair": pair_fp, **row_cfg[r]}) for r in want}
    pdir = lambda r: os.path.join(args.work, "preds", r, rfp[r])
    names = [nm for nm, _, _ in test_list]

    def cached(r, nm):
        return (r not in forced) and os.path.exists(os.path.join(pdir(r), nm + ".png"))

    todo = {r: [nm for nm in names if not cached(r, nm)] for r in want}
    print(f"\nprotocol {proto['mode']}  |  {len(test_list)} test, {len(fit_list)} fit  "
          f"|  work={args.work}")
    print(f"{'row':<30}{'fingerprint':<13}{'cached':>8}{'to compute':>12}")
    print("-" * 63)
    for r in want:
        print(f"{ROWS[r][0]:<30}{rfp[r]:<13}{len(names)-len(todo[r]):>8}{len(todo[r]):>12}")
    n_new = sum(len(v) for v in todo.values())
    print("-" * 63)
    print(f"{'TOTAL':<30}{'':<13}{len(want)*len(names)-n_new:>8}{n_new:>12}")
    if args.dry_run:
        print("\n--dry-run: nothing computed."); return

    with contextlib.redirect_stdout(io.StringIO()):
        lp = C.LPIPSLoss().to(device)
    isp = C.ParametricISP(cfg).to(device)

    # ---- lazily built shared artifacts ---------------------------------------
    shared = {}

    def fit_pairs_t():
        if "fit" not in shared:
            t0 = time.time()
            out = []
            for nm, a_p, b_p in fit_list:
                try:
                    a, b = get_pair(nm, a_p, b_p, args.pair_cache, args.side, args.side_mode, is_raw)
                except Exception as e:
                    print(f"  skipping broken fit image {nm}: {e}"); continue
                out.append((to_t(a, device), to_t(b, device)))
            shared["fit"] = out
            timings["decode fit split"] = time.time() - t0
        return shared["fit"]

    def artifact(name, builder, cfg_key):
        """Closed-set fits are cached on disk too: a rerun that only changes a test row must
        not refit the baselines."""
        path = os.path.join(args.work, "artifacts", f"{name}_{fp({'pair': pair_fp, **cfg_key})}.pt")
        if name not in shared:
            if os.path.exists(path):
                shared[name] = torch.load(path, map_location=device)
            else:
                t0 = time.time()
                print(f"fitting {name} ...")
                shared[name] = builder()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(shared[name], path)
                timings[f"fit {name}"] = time.time() - t0
        return shared[name]

    def director():
        if "dir" not in shared:
            t0 = time.time()
            d = C.StyleDirector(cfg).to(device).eval()
            C.load_director(d, cfg, required=True)
            vq = None
            if cfg.use_lut:
                ck = os.path.join(cfg.out_dir, "lut_vqvae.pt")
                if os.path.exists(ck):
                    vq = C.LUTVQVAE(cfg).to(device).eval()
                    vq.load_state_dict(torch.load(ck, map_location=device))
            shared["dir"] = (d, vq)
            timings["load director"] = time.time() - t0
        return shared["dir"]

    def editor():
        if "ed" not in shared:
            t0 = time.time()
            print(f"loading {EDITOR_CKPT[args.editor][1]} ...")
            shared["ed"] = load_editor(args.editor, device, args.editor_offload)
            timings["load editor"] = time.time() - t0
        return shared["ed"]

    # ---- producers -----------------------------------------------------------
    def produce(r, nm, a, b, x, y, x_lin):
        if r == "identity":
            return x
        if r == "reinhard":
            return reinhard_transfer(x, y)
        if r == "matgamma":
            p = artifact("matgamma", lambda: fit_matrix_gamma(fit_pairs_t(), device),
                         row_cfg["matgamma"])
            with torch.no_grad():
                return apply_matrix_gamma(x, p)
        if r == "globallut":
            L = artifact("globallut",
                         lambda: fit_lut(fit_pairs_t(), device, iters=800, lr=0.05, tv=0.3),
                         row_cfg["globallut"])
            with torch.no_grad():
                return C.lut3d_apply(L, x)
        if r == "editor":
            return to_t(run_editor(editor(), a, args.editor_prompt, args.editor_steps,
                                   args.editor_text_cfg, args.editor_image_cfg), device)
        if r == "ours":
            d, vq = director()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                th, ll, _ = d([a], [args.prompt])
            lut = vq.decode_from_ids(ll.float().argmax(-1)) if vq is not None else None
            with torch.no_grad():
                return isp(x_lin, th.float(), lut, pointwise_only=True)
        if r == "ours_cached":
            if "sc" not in shared:
                shared["sc"] = C.StyleCache(cfg.style_cache_dir)
            th, _, lut = C.fetch_style_batch(shared["sc"], [args.cache_key], device, cfg)
            with torch.no_grad():
                return isp(x_lin, th, lut, pointwise_only=True)
        xs, ys = subsample(x, y, args.fit_pixels)
        if r == "fit_theta":
            xls, _ = subsample(x_lin, y, args.fit_pixels)
            th = fit_theta(xls, ys, isp, cfg, device, iters=args.oracle_iters)
            with torch.no_grad():
                return isp(x_lin, th, None, pointwise_only=True)
        if r == "oracle_curve":
            cv = fit_curves(xs, ys, device, iters=args.oracle_iters)
            with torch.no_grad():
                return apply_curves(cv, x)
        if r == "oracle_lut":
            L = fit_lut([(xs, ys)], device, iters=args.oracle_iters, tv=0.0)
            with torch.no_grad():
                return C.lut3d_apply(L, x)
        raise KeyError(r)

    # ---- produce -------------------------------------------------------------
    t_loop = time.time()
    done = 0
    for i, (nm, a_p, b_p) in enumerate(test_list, 1):
        need = [r for r in want if not cached(r, nm)]
        if not need:
            continue
        try:
            a, b = get_pair(nm, a_p, b_p, args.pair_cache, args.side, args.side_mode, is_raw)
        except Exception as e:
            print(f"  skipping broken image {nm}: {e}"); continue
        x, y = to_t(a, device), to_t(b, device)
        x_lin = C.decode_transfer(x, cfg.transfer)
        for r in need:
            save_pred(produce(r, nm, a, b, x, y, x_lin), os.path.join(pdir(r), nm + ".png"))
        done += 1
        if done % 10 == 0:
            el = time.time() - t_loop
            left = max(len(test_list) - i, 0) * el / done
            print(f"  {i}/{len(test_list)}  {el:.0f}s elapsed, ~{left:.0f}s left")
    timings["produce"] = time.time() - t_loop

    # ---- score (always from cached predictions; metrics are free to change) ---
    t0 = time.time()
    res = {r: [] for r in want}
    for nm, a_p, b_p in test_list:
        if not all(os.path.exists(os.path.join(pdir(r), nm + ".png")) for r in want):
            continue
        try:
            a, b = get_pair(nm, a_p, b_p, args.pair_cache, args.side, args.side_mode, is_raw)
        except Exception:
            continue
        x, y = to_t(a, device), to_t(b, device)
        for r in want:
            p = to_t(Image.open(os.path.join(pdir(r), nm + ".png")).convert("RGB"), device)
            res[r].append(score_all(p, y, x, lp))
    timings["score"] = time.time() - t0
    n_eval = len(res[want[0]]) if want else 0
    if n_eval == 0:
        print("nothing scored"); return

    def agg(r, k):
        return float(np.mean([d[k] for d in res[r]]))

    # ---- report --------------------------------------------------------------
    note = ("the authors' preprocessed 480p pairs" if not is_raw
            else "OUR OWN DNG render, not the protocol's sRGB input")
    L = ["# Comparison on the FiveK expert-C protocol\n",
         f"{n_eval} test images, input: {note}. Closed-set rows fitted on "
         f"{len(fit_list)} separate training images.",
         f"Our prompt: `{args.prompt}`; ISP rows use the per-pixel tract only "
         f"(`pointwise_only=True`), the one that bakes into a single table.",
         (f"{EDITOR_CKPT[args.editor][1]} instruction: `{args.editor_prompt}` "
          f"({args.editor_steps} steps, text cfg {args.editor_text_cfg}, image cfg "
          f"{args.editor_image_cfg})." if args.editor != "none" else ""),
         "", "## Measured here\n",
         "| method | sees | PSNR | SSIM | dE00 | dE_ab | LPIPS | drift |",
         "|---|---|---|---|---|---|---|---|"]
    for g in ("training-free", "closed-set", "open-set", "oracle"):
        rs = [r for r in want if ROWS[r][1] == g]
        if not rs:
            continue
        L.append(f"| *{GROUP_LABEL[g]}* | | | | | | | |")
        for r in rs:
            lbl = EDITOR_CKPT[args.editor][1] if r == "editor" else ROWS[r][0]
            L.append(f"| {lbl} | {ROWS[r][2]} | {agg(r,'psnr'):.2f} | {agg(r,'ssim'):.4f} | "
                     f"{agg(r,'de00'):.2f} | {agg(r,'deab'):.2f} | {agg(r,'lpips'):.4f} | "
                     f"{agg(r,'drift'):.2f} |")
    L += ["",
          "`dE_ab` is the Euclidean Lab distance (CIE76) that every published method below",
          "reports; `dE00` is CIEDE2000. They are different quantities -- compare each column",
          "only against itself. `drift` is how far a method moved the input regardless of",
          "direction: a large drift with no dE gain means motion away from the target."]

    if "globallut" in want:
        g = agg("globallut", "psnr")
        ok = abs(g - 20.37) < 1.0
        L += ["", f"**Protocol check.** The global-LUT row measures {g:.2f} dB against the "
                  f"20.37 dB published for this exact setting: "
                  + ("consistent, so the protocol is reproduced."
                     if ok else "**inconsistent**, so the rows above are NOT comparable to the "
                                "published block and must be reported separately.")]

    if all(r in want for r in ("fit_theta", "oracle_curve", "oracle_lut")):
        t_, c_, o_ = (agg(r, "de00") for r in ("fit_theta", "oracle_curve", "oracle_lut"))
        L += ["", "### Where the capacity goes\n", "| bound | dE00 |", "|---|---|",
              f"| our parameterisation (41 params, theta only) | {t_:.2f} |",
              f"| any channel-separable map (99 free knots) | {c_:.2f} |",
              f"| any per-pixel map (full 3D LUT) | {o_:.2f} |", "",
              f"The {c_:.2f} -> {o_:.2f} step is the part of the target that cannot be",
              "expressed by treating channels independently, i.e. the part that needs a",
              "cross-channel map: the tone curve acts on luma uniformly and the hue/saturation",
              "map acts per hue bin, so neither covers it. That is the measurement justifying",
              "the LUT head."]

    L += ["", "## Published on the SAME protocol (not re-run here)\n",
          "| method | conditioning | PSNR | SSIM | dE_ab |", "|---|---|---|---|---|"]
    for nm_, _, cd, ps, ss, de in PUBLISHED:
        L.append(f"| {nm_} | {cd} | {ps:.2f} | {ss:.3f} | {de if de is None else f'{de:.2f}'} |"
                 .replace("None", "—"))
    L += ["", "White-box family (interpretable filters), as re-run by Kosugi (ACM MM 2024):",
          "", "| method | conditioning | PSNR | SSIM | dE_ab |", "|---|---|---|---|---|"]
    for nm_, _, cd, ps, ss, de in PUBLISHED_WHITEBOX:
        L.append(f"| {nm_} | {cd} | {ps:.2f} | {ss:.3f} | {de:.2f} |")
    L += ["", "PG-IA-NILUT is the only open-set entry measured on this protocol, and it trails",
          "the best closed-set methods by 0.33 dB -- its authors attribute that gap to adding",
          "prompt control (25.46 dB without it). An independent confirmation that openness has",
          "a measurable but small cost. Note its prompt only names the filters; the target is",
          "still the single expert-C style, whereas here the prompt selects the style itself.",
          "", "## Language-conditioned work on OTHER benchmarks (context only)\n",
          "| method | conditioning | reported |", "|---|---|---|"]
    for nm_, cd, rep in PUBLISHED_OTHER:
        L.append(f"| {nm_} | {cd} | {rep} |")
    L += ["", "Measured on different data; listed to place this work among its peers, not as a",
          "ranking. PixTalk publishes no weights, and the MLLM-agent methods need paid",
          "multi-turn API calls per image, so neither can be re-run here."]

    if args.lut_dir:
        probe = [get_pair(nm, p, q, args.pair_cache, args.side, args.side_mode, is_raw)[0]
                 for nm, p, q in test_list[:4]]
        L += lut_repr_report(cfg, args.lut_dir, probe, device)

    L += ["", "## Cost\n", "| stage | seconds |", "|---|---|"]
    for k, v in timings.items():
        L.append(f"| {k} | {v:.0f} |")
    L.append(f"| total | {time.time()-t_start:.0f} |")
    L += ["", f"Predictions cached in `{args.work}/preds`. Re-running is incremental: a new",
          "metric costs only rescoring, and changing one baseline recomputes only that row.",
          "`--dry-run` reports the cache state before spending anything."]

    txt = "\n".join(s for s in L if s)
    open(f"{args.report}.md", "w").write(txt)
    out = {ROWS[r][0]: {k: agg(r, k) for k in ("psnr", "ssim", "de00", "deab", "lpips", "drift")}
           for r in want}
    out["_meta"] = {"n_test": n_eval, "n_fit": len(fit_list), "protocol": proto,
                    "prompt": args.prompt, "pointwise_only": True,
                    "editor": None if args.editor == "none" else
                              {"model": EDITOR_CKPT[args.editor][0], "prompt": args.editor_prompt,
                               "steps": args.editor_steps, "text_cfg": args.editor_text_cfg,
                               "image_cfg": args.editor_image_cfg},
                    "fingerprints": rfp, "timings": {k: float(v) for k, v in timings.items()}}
    json.dump(out, open(f"{args.report}.json", "w"), indent=2)
    print(txt)
    print(f"\nwrote {args.report}.md and {args.report}.json")


if __name__ == "__main__":
    main()