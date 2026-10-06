"""
Camera-look RAW ISP: canonical front-end + differentiable parametric ISP + LUT VQ-VAE
+ VLM style director + one-step diffusion renderer (FLUX.2 klein 4B backbone).

Single-file reference implementation. Defaults below target a SINGLE RTX 4090 (24GB):
both frozen backbones load in 4-bit NF4 (QLoRA-style), optimizer is 8-bit AdamW,
resolution/rank/batch are cut down, and each phase auto-stops at max_steps instead
of sweeping full epochs over a huge corpus. For a multi-GPU server run, set
use_4bit=False, use_8bit_adam=False, raise lora_rank/renderer_res/batch sizes back up
and drop max_steps (see comments on each field).

Requires: torch>=2.4, torchvision, transformers>=4.57, diffusers>=0.37, peft>=0.13,
          bitsandbytes, rawpy, exifread, pillow, lpips, numpy.

Launch (single GPU):
  python camera_look_isp.py --phase frontend      --manifest data/frontend.jsonl
  python camera_look_isp.py --phase director_sft  --manifest data/director_sft.jsonl
  python camera_look_isp.py --phase director_rl   --manifest data/director_rl.jsonl
  python camera_look_isp.py --phase cache_style   --manifest data/renderer_pairs.jsonl
  python camera_look_isp.py --phase renderer --stage 1 --manifest data/dcp_pairs.jsonl

cache_style runs the (LoRA-tuned) director once per unique style_key in the manifest and
saves theta/s/lut to --manifest's sibling cache dir; renderer training then loads those
instead of holding the 4B VLM in memory. Pass --use_lut to opt back into the LUT VQ-VAE
path (off by default here since a theta-only look already gets most of the effect and
skipping it removes a whole training stage + data-curation burden).

Launch (multi-GPU server):
  torchrun --nproc_per_node=8 camera_look_isp.py --phase renderer --stage 1 \
      --manifest data/dcp_pairs.jsonl
"""

import os, io, json, math, random, argparse, time, warnings, contextlib, glob
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image

# ============================== config ==============================

@dataclass
class Config:
    director_ckpt: str = "Qwen/Qwen3-VL-4B-Instruct"
    director_revision: str = None   # pin a commit sha for reproducible downloads
    renderer_revision: str = None
    text_encoder_revision: str = None
    renderer_ckpt: str = "black-forest-labs/FLUX.2-klein-4B"

    use_4bit: bool = True          # QLoRA-style NF4 quant of the frozen backbones; needed to fit 24GB
    use_8bit_adam: bool = True     # bitsandbytes AdamW8bit for LoRA params; ~4x less optimizer memory
    max_steps: int = 2000          # hard cap per phase call; None disables (server/full-corpus runs)

    lora_rank: int = 16            # 64 for a proper server run
    lora_alpha: int = 16
    theta_dim: int = 41
    transfer: str = "bt709"        # must match the curve the targets were encoded with
    n_hue_bins: int = 8
    lut_size: int = 33
    lut_tokens: int = 64
    lut_codebook: int = 512
    lut_embed_dim: int = 64
    use_lut: bool = False           # skip the LUT VQ-VAE path entirely (theta-only look); big time saver, --use_lut / server to re-enable
    use_style_cache: bool = True    # renderer reads precomputed director outputs from disk instead of running the VLM live
    style_cache_dir: str = "style_cache"
    style_dim: int = 1024
    prompt_dropout: float = 0.1   # blank-prompt samples whose target is the unstyled base
    grad_clip: float = 10.0        # fixed fallback; see grad_clip_factor (adaptive is default)
    grad_clip_factor: float = 4.0  # clip when ||g|| > factor * EMA(||g||); absolute thresholds
                                    # are meaningless here (natural scale is in the hundreds)
    style_dropout: float = 0.1     # per-sample; renderer must degrade to a pure quality ISP
    timestep_max: float = 0.3      # pseudo-timestep sampled in [0, max) so the knob is learned
    pool: str = "last"             # "last" | "mean"; see D1.7 (mean dilutes the prompt)
    normalize_difficulty: bool = True  # dE loss relative to each sample's own
                                       # base->target distance, so strong looks do
                                       # not dominate the gradient
    difficulty_floor: float = 3.0      # keeps easy samples from exploding the ratio
    head_init_std: float = 1e-3    # near-neutral theta at init WITHOUT blocking upstream grad
    head_lr_mult: float = 3.0      # heads are from scratch; LoRA only adapts a pretrained net.
                                   # Was 10.0: over thousands of steps that drove raw theta
                                   # past |3|, where every tanh/sigmoid saturates and the
                                   # gradient dies, pinning parameters at their bounds.
    resume: object = None          # path (or True for director_final.pt) to warm-start from
    lut_only: bool = False         # train ONLY the LUT head; theta output frozen
    lut_ce_weight: float = 0.1     # auxiliary token CE; primary signal is the render loss
    cache_avg: int = 6             # images averaged per style_key when caching
    use_frontend: bool = False     # evaluate through the front-end from RAW,
                                   # instead of the pre-rendered neutral image
    split: str = "train"           # train | test | all; splitting is BY SCENE
    split_frac: float = 0.15       # share of scenes held out for evaluation
    demo_max_side: int = 1600      # downscale demo inputs to this long edge
    force_distill: bool = False    # distill even without a renderer that beats the guide
    rl_lr_scale: float = 0.2       # RL runs well below SFT LR; it is a nudge, not a retrain
    rl_sigma: float = 0.15         # exploration noise on theta
    rl_anchor: float = 1.0         # weight of the supervised anchor that holds fidelity
    tame_limit: float = 0.0        # >0 clamps raw theta at INFERENCE (no retraining):
                                   # 1.0 -> sat in [0.24,1.76], exposure +/-0.76 stops
    theta_l2: float = 2e-2         # keeps RAW theta in the responsive region of tanh.
                                   # At 2e-2 the equilibrium |theta| stays under ~2.5 for
                                   # task gradients up to 0.1; 5e-3 only holds to 0.025.
                                   # Bounding alone does not prevent saturation: it only
                                   # hides it, and a saturated parameter cannot recover
                                   # because its gradient is ~0.
    grad_clip_frontend: float = 1.0
    style_ctx_tokens: int = 4
    joint_attention_dim: int = 7680
    bgrid_res: int = 16
    frontend_width: int = 48
    frontend_crop: int = 256
    frontend_batch: int = 16       # tiny full-precision net, 24GB easily fits this; raise on a server
    director_img_size: int = 384
    director_batch: int = 1        # effective batch = director_batch * director_grad_accum
    director_grad_accum: int = 16
    rl_group_size: int = 4         # 8 on a server; fewer rollouts/prompt = cheaper, noisier advantage
    rl_prompts_per_step: int = 1
    renderer_res: int = 384        # 768 on a server; diffusion trains fine smaller, guided upsample handles full-res later
    renderer_batch: int = 1
    renderer_grad_accum: int = 16
    lr_frontend: float = 2e-4
    lr_director: float = 1e-4
    lr_renderer: float = 5e-5
    epochs: int = 20               # max_steps will usually cut this short on a single GPU
    warmup_steps: int = 500
    grad_ckpt: bool = True
    out_dir: str = "checkpoints"
    log_every: int = 20
    save_every: int = 500
    seed: int = 0


def load_frontend(cfg, device, required=False):
    """Load the trained front-end, or None if it has not been trained.

    The style path can read a pre-rendered neutral image instead, which is how the training
    pairs were built; passing the front-end in makes the evaluated pipeline the full one,
    RAW included.
    """
    path = f"{cfg.out_dir}/frontend_final.pt"
    if not os.path.exists(path):
        cand = sorted(glob.glob(f"{cfg.out_dir}/frontend_*.pt"), key=os.path.getmtime)
        path = cand[-1] if cand else None
    if path is None or not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"front-end checkpoint not found in {cfg.out_dir}. Train it with --phase frontend.")
        return None
    fe = FrontEnd(cfg).to(device).eval()
    fe.load_state_dict(torch.load(path, map_location=device))
    fe.requires_grad_(False)
    print(f"[frontend] active ({os.path.basename(path)})")
    return fe


@torch.no_grad()
def frontend_linear(fe, it, device, cfg):
    """Scene-referred linear image produced from the packed Bayer crop by the front-end."""
    d = np.load(it["raw_path"])
    bayer = torch.from_numpy(d["bayer"].astype(np.float32))[None, None].to(device)
    ccm = torch.from_numpy(d["ccm"].astype(np.float32))[None].to(device)
    meta = torch.tensor([[it.get("iso", 100.0), it.get("exposure", 1 / 60),
                          it.get("fnumber", 4.0), it.get("focal_length", 35.0),
                          1.0, 5500.0]], dtype=torch.float32, device=device)
    x_lin, _ = fe(bayer.clamp(0, 1), meta, ccm)
    return x_lin.clamp(0, 1)


def scene_of(it):
    """Scene identifier of a manifest entry. Crops are named <scene>__<style>__<k>."""
    p = it.get("base_path") or it.get("jpeg_path", "")
    return os.path.basename(p).split("__")[0]


def split_manifest(items, split="all", frac=0.15, seed=0):
    """Split BY SCENE, never by pair.

    One scene produces many pairs (every style x every crop), so a pair-level split puts
    the same pixels on both sides and the test set stops being held out. Splitting on the
    scene identifier keeps every crop and every style of a scene on one side.
    """
    if split == "all":
        return items
    scenes = sorted({scene_of(it) for it in items})
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n_test = max(1, int(round(len(scenes) * frac)))
    test = set(scenes[:n_test])
    keep = test if split == "test" else set(scenes) - test
    out = [it for it in items if scene_of(it) in keep]
    if is_main():
        print(f"[split] {split}: {len(out)} pairs from {len(keep)} scenes "
              f"(of {len(items)} pairs / {len(scenes)} scenes)")
    return out


def load_manifest(path):
    """Read a manifest, resolving any relative file paths against the manifest's own
    directory. Manifests written with relative paths are portable between machines."""
    base = os.path.dirname(os.path.abspath(path))
    items = []
    for line in open(path):
        if not line.strip():
            continue
        it = json.loads(line)
        for k in ("raw_path", "jpeg_path", "base_path", "lut_path"):
            if k in it and not os.path.isabs(it[k]):
                it[k] = os.path.normpath(os.path.join(base, it[k]))
        items.append(it)
    return items


def quiet_third_party():
    """Silence progress bars and banners from transformers / huggingface / lpips so that
    command output contains metrics only. Set CAMERA_LOOK_VERBOSE=1 to keep them."""
    if os.environ.get("CAMERA_LOOK_VERBOSE"):
        return
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass
    warnings.filterwarnings("ignore")


def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def enable_grad_ckpt(model, label=""):
    """diffusers spells it enable_gradient_checkpointing(); transformers spells it
    gradient_checkpointing_enable(). PEFT forwards __getattr__ to the wrapped model, so the
    wrong name surfaces as an AttributeError only at call time."""
    for name in ("gradient_checkpointing_enable", "enable_gradient_checkpointing"):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                fn()
                return True
            except Exception as e:
                print(f"[grad_ckpt] {label}: {name} failed ({e})")
    # last resort: the flag some diffusers modules read directly
    try:
        model.gradient_checkpointing = True
        return True
    except Exception:
        pass
    print(f"[grad_ckpt] {label}: could not enable; training will use more memory")
    return False


def unwrap(model):
    return model.module if hasattr(model, "module") else model


class GradNormTracker:
    """Adaptive gradient clipping.

    A fixed absolute threshold is the wrong tool here: the natural gradient scale is set by
    dE00 (a Lab-unit distance, ~100x the rgb scale) times ||pooled|| after LayerNorm (~sqrt(d)
    ~= 50), so norms of several hundred are expected and benign. Worse, AdamW's update is
    invariant to a constant rescale of all gradients, so clipping everything by the same large
    factor accomplishes nothing except distorting the moment estimates.

    What we actually want is spike protection: clip relative to the recent typical norm.
    """

    def __init__(self, factor=4.0, warmup=25, beta=0.95):
        self.factor, self.warmup, self.beta = factor, warmup, beta
        self.ema = None
        self.n = 0
        self.n_clipped = 0

    def threshold(self):
        if self.ema is None or self.n < self.warmup:
            return None
        return self.factor * self.ema

    def update(self, gnorm):
        self.n += 1
        self.ema = gnorm if self.ema is None else self.beta * self.ema + (1 - self.beta) * gnorm


def adaptive_clip(params, tracker, fallback=None):
    """Measure, then clip only if this step is an outlier. Returns (norm, threshold_used)."""
    params = list(params)
    gnorm = float(torch.nn.utils.clip_grad_norm_(params, float("inf")))
    if not math.isfinite(gnorm):
        return gnorm, None
    thr = tracker.threshold()
    if thr is None and fallback is not None and gnorm > fallback:
        thr = fallback
    if thr is not None and gnorm > thr:
        torch.nn.utils.clip_grad_norm_(params, thr)
        tracker.n_clipped += 1
    tracker.update(gnorm)
    return gnorm, thr


def make_optimizer(params, lr, cfg: Config):
    # params may be a flat iterable of tensors OR a list of param-group dicts
    if isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
        groups = [{**g, "params": [p for p in g["params"] if p.requires_grad]} for g in params]
        groups = [g for g in groups if g["params"]]
    else:
        groups = [p for p in params if p.requires_grad]
    if cfg.use_8bit_adam:
        try:
            from bitsandbytes.optim import AdamW8bit
            return AdamW8bit(groups, lr=lr, weight_decay=1e-4)
        except ImportError:
            print("bitsandbytes not available, falling back to torch.optim.AdamW")
    return torch.optim.AdamW(groups, lr=lr, weight_decay=1e-4)


def split_director_params(director, cfg):
    """Heads train from scratch; LoRA adapts a pretrained backbone. Same LR for both is wrong."""
    head_pref = ("theta_head", "lut_head", "style_head", "pool_norm")
    heads, lora = [], []
    for n, p in unwrap(director).named_parameters():
        if not p.requires_grad:
            continue
        (heads if n.startswith(head_pref) else lora).append(p)
    return [
        {"params": lora, "lr": cfg.lr_director},
        {"params": heads, "lr": cfg.lr_director * cfg.head_lr_mult},
    ], heads, lora


# ============================== A. canonical front-end ==============================

class ParamNet(nn.Module):
    """EXIF -> embedding via non-linear equalization.

    Raw EXIF spans wildly different magnitudes (exposure 1/8000, temp 5500), and
    feeding 1/x on a dropped-to-zero field yields 1e4, which scales gradients by 1e4
    and overflows bf16. So: divide by per-field reference scales, replace dropped
    fields with the neutral value 1.0 plus a missing-flag, clamp, then LayerNorm.
    """

    # iso, exposure_s, f_number, focal_mm, cfa_code, temp_k
    REF = (100.0, 1.0 / 60.0, 4.0, 35.0, 1.0, 5500.0)

    def __init__(self, n_raw=6, dim=256, p_drop=0.15):
        super().__init__()
        self.n_raw = n_raw
        self.p_drop = p_drop
        ref = torch.tensor(self.REF[:n_raw], dtype=torch.float32)
        self.register_buffer("ref", ref)
        in_dim = n_raw * 5 + n_raw          # 5 equalized terms + missing mask
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.SiLU(),
            nn.Linear(256, dim), nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x_raw):
        x = x_raw.float() / self.ref                      # ~1.0 for typical values
        missing = torch.zeros_like(x)
        if self.training:
            drop = (torch.rand_like(x) < self.p_drop)
            x = torch.where(drop, torch.ones_like(x), x)  # neutral, not zero
            missing = drop.float()
        x = x.clamp(1e-2, 1e2)                            # bounds 1/x to [1e-2, 1e2]
        lx = torch.log(x)
        feats = torch.cat([x, 1.0 / x, torch.sqrt(x), lx, torch.sin(lx), missing], dim=-1)
        return self.mlp(self.norm(feats))


class NAFBlock(nn.Module):
    def __init__(self, c, expand=2):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, c)
        self.conv1 = nn.Conv2d(c, c * expand, 1)
        self.dwconv = nn.Conv2d(c * expand, c * expand, 3, padding=1, groups=c * expand)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c * expand, c * expand, 1))
        self.conv2 = nn.Conv2d(c * expand, c, 1)
        self.norm2 = nn.GroupNorm(1, c)
        self.ff1 = nn.Conv2d(c, c * expand, 1)
        self.ff2 = nn.Conv2d(c * expand, c, 1)
        self.gamma1 = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma2 = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.conv1(self.norm1(x))
        y = F.gelu(self.dwconv(y)) * torch.sigmoid(self.sca(y))
        y = self.conv2(y)
        x = x + y * self.gamma1
        y = self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return x + y * self.gamma2


class CanoNet(nn.Module):
    """Joint denoise + demosaic on packed Bayer -> linear RGB, conditioned on ParamNet embedding."""

    def __init__(self, width=48, n_blocks=(2, 2, 4, 2)):
        super().__init__()
        self.stem = nn.Conv2d(4, width, 3, padding=1)
        self.cond = nn.Linear(256, width)
        self.blocks = nn.ModuleList([NAFBlock(width) for _ in range(sum(n_blocks))])
        self.to_rgb = nn.Sequential(nn.Conv2d(width, 12, 3, padding=1), nn.PixelShuffle(2))

    def forward(self, bayer, m):
        b, _, h, w = bayer.shape
        packed = torch.stack([
            bayer[:, 0, 0::2, 0::2], bayer[:, 0, 0::2, 1::2],
            bayer[:, 0, 1::2, 0::2], bayer[:, 0, 1::2, 1::2],
        ], dim=1)
        x = self.stem(packed) + self.cond(m)[:, :, None, None]
        for blk in self.blocks:
            x = blk(x)
        return F.softplus(self.to_rgb(x))


class FrontEnd(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.paramnet = ParamNet()
        self.canonet = CanoNet(cfg.frontend_width)

    def forward(self, bayer, raw_meta, ccm):
        m = self.paramnet(raw_meta)
        rgb_cam = self.canonet(bayer, m)
        x_lin = torch.einsum("bij,bjhw->bihw", ccm, rgb_cam)
        return x_lin.clamp(min=0.0), m


# ============================== C1. differentiable parametric ISP ==============================

class MonotonicCurve(nn.Module):
    """A monotone piecewise-linear 1D curve from n control-point deltas (softplus + cumsum)."""

    def __init__(self, n_points=8):
        super().__init__()
        self.n_points = n_points

    def forward(self, x, deltas):
        b = deltas.shape[0]
        steps = F.softplus(deltas) + 1e-4
        cum = torch.cumsum(steps, dim=-1)
        cum = cum / cum[:, -1:].clamp(min=1e-6)
        cum = torch.cat([torch.zeros(b, 1, device=x.device, dtype=x.dtype), cum], dim=-1)
        xs = torch.linspace(0, 1, self.n_points + 1, device=x.device, dtype=x.dtype)
        xf = x.clamp(0, 1)
        idx = torch.bucketize(xf.reshape(b, -1), xs[1:-1].contiguous())
        idx = idx.clamp(0, self.n_points - 1)
        x0 = xs[idx]
        x1 = xs[idx + 1]
        y0 = torch.gather(cum, 1, idx)
        y1 = torch.gather(cum, 1, idx + 1)
        t = ((xf.reshape(b, -1) - x0) / (x1 - x0).clamp(min=1e-6)).clamp(0, 1)
        y = y0 + t * (y1 - y0)
        return y.reshape(x.shape)


def rgb_to_hsv(rgb):
    r, g, b = rgb.unbind(1)
    maxc, _ = rgb.max(1)
    minc, _ = rgb.min(1)
    v = maxc
    diff = (maxc - minc).clamp(min=1e-3)
    s = diff / maxc.clamp(min=1e-3)
    rc = (maxc - r) / diff
    gc = (maxc - g) / diff
    bc = (maxc - b) / diff
    h = torch.where(maxc == r, bc - gc,
        torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    return torch.stack([h, s, v], 1)


def hsv_to_rgb(hsv):
    h, s, v = hsv.unbind(1)
    i = (h * 6.0).floor()
    f = h * 6.0 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i.long() % 6
    r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, v)))))
    g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p, p)))))
    bch = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q)))))
    return torch.stack([r, g, bch], 1)


class HueSatMap(nn.Module):
    def __init__(self, n_bins=8):
        super().__init__()
        self.n_bins = n_bins

    def forward(self, rgb, huesat):
        hsv = rgb_to_hsv(rgb.clamp(1e-6, 1))
        h, s, v = hsv.unbind(1)
        huesat = huesat.view(-1, self.n_bins, 3)
        pos = h * self.n_bins
        i0 = pos.floor().long() % self.n_bins
        i1 = (i0 + 1) % self.n_bins
        t = (pos - pos.floor())[:, None]
        b = rgb.shape[0]
        hs0 = torch.stack([huesat[bi, i0[bi]] for bi in range(b)], 0).permute(0, 3, 1, 2)
        hs1 = torch.stack([huesat[bi, i1[bi]] for bi in range(b)], 0).permute(0, 3, 1, 2)
        hs = hs0 * (1 - t) + hs1 * t
        h = (h + 0.1 * torch.tanh(hs[:, 0])) % 1.0
        s = (s * (1 + torch.tanh(hs[:, 1]))).clamp(0, 1)
        v = (v * (1 + 0.3 * torch.tanh(hs[:, 2]))).clamp(0, 1)
        return hsv_to_rgb(torch.stack([h, s, v], 1))


def lut3d_apply(lut, rgb):
    b, c, h, w = rgb.shape
    n = lut.shape[-1]
    vol = lut.view(b, 3, n, n, n)
    grid = (rgb.clamp(0, 1) * 2 - 1).permute(0, 2, 3, 1).view(b, h * w, 1, 1, 3)
    out = F.grid_sample(vol, grid, mode="bilinear", align_corners=True)
    return out.view(b, 3, h, w)


def gaussian_blur(x, sigma):
    sigma = sigma.clamp(min=0.1)
    k = 7
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    kern = torch.exp(-0.5 * (ax[None] / sigma[:, None]) ** 2)
    kern = kern / kern.sum(-1, keepdim=True)
    kern2d = kern[:, :, None] * kern[:, None, :]
    kern2d = kern2d[:, None].expand(-1, x.shape[1], -1, -1).reshape(-1, 1, k, k)
    xg = x.reshape(1, -1, x.shape[-2], x.shape[-1])
    out = F.conv2d(xg, kern2d, padding=k // 2, groups=xg.shape[1])
    return out.reshape(x.shape)


def apply_grain(x, sigma, size):
    noise = torch.randn_like(x)
    noise = gaussian_blur(noise, size)
    return (x + noise * sigma[:, None, None, None]).clamp(0, 1)


def unsharp_mask(x, radius, amount):
    blur = gaussian_blur(x, radius)
    return (x + amount[:, None, None, None] * (x - blur)).clamp(0, 1)


THETA_SLOTS = ["exposure", "black", "tone", "huesat", "knee", "sat", "sh_r", "sh_a", "gr_s", "gr_z", "cnr"]
THETA_SIZES = [1, 1, 8, 24, 1, 1, 1, 1, 1, 1, 1]
THETA_DIM = sum(THETA_SIZES)


def split_theta(theta):
    out, i = {}, 0
    for name, n in zip(THETA_SLOTS, THETA_SIZES):
        out[name] = theta[:, i:i + n]
        i += n
    return out


def decode_transfer(y, mode="bt709"):
    """Inverse of encode_transfer. Must use the same curve as the encode: a mismatched
    pair leaves a systematic colour error that no amount of training can remove."""
    y = y.clamp(0.0, 1.0)
    if mode == "bt709":
        return torch.where(y < 4.5 * 0.018, y / 4.5,
                           ((y.clamp(min=4.5 * 0.018) + 0.099) / 1.099) ** 2.222)
    if mode == "srgb":
        return torch.where(y <= 0.04045, y / 12.92,
                           ((y.clamp(min=0.04045) + 0.055) / 1.055) ** 2.4)
    return y ** 2.2


def encode_transfer(x, mode="bt709"):
    """Encode linear values for display. Must match the curve the targets were
    written with: rawpy gamma=(2.222, 4.5) corresponds to BT.709."""
    x = x.clamp(1e-6, 1.0)
    if mode == "bt709":
        return torch.where(x < 0.018, 4.5 * x,
                           1.099 * x.clamp(min=0.018) ** (1 / 2.222) - 0.099)
    if mode == "srgb":
        return torch.where(x <= 0.0031308, 12.92 * x,
                           1.055 * x.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)
    return x ** (1 / 2.2)


# Offsets so that theta=0 is a NEUTRAL render. Without these, sigmoid(0)=0.5 means
# "half strength" for every effect: 25% blur, grain, and a +0.14 midtone shift baked
# into every output. probe_isp.py confirmed knee=0 is the best default.
KNEE_OFF, GRAIN_OFF, CNR_OFF = 3.0, 4.0, 4.0
# Physical bounds on the free parameters. Without them the heads emit raw values and the
# exposure -4.5 stops and black point -0.24 (negative black is meaningless), which renders
# black or grey for every prompt. Bounding makes degenerate solutions unreachable, so strong
# "effect" looks can stay in the training set without poisoning everything else.
EXPOSURE_STOPS = 2.0    # +/- 2 stops is a generous photographic range
BLACK_MAX = 0.08        # black point lift, never negative
# The tone curve was the LAST unbounded dimension, and bounding exposure simply moved the
# degenerate solution here: with raw deltas the monotone curve can map mid-grey 0.5 -> 0.0000,
# crushing the image to black and collapsing chroma with it. Bounding the deltas limits how
# extreme the curve can get while still allowing real contrast changes.
# With +/-1.5 the per-segment step ratio is softplus(1.5)/softplus(-1.5) ~ 8.5x, which is a
# strong S-curve or a strong crush-curve, but not a cliff.
TONE_RANGE = 1.5


class ParametricISP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.curve = MonotonicCurve(8)
        self.huesat = HueSatMap(cfg.n_hue_bins)
        self.transfer = getattr(cfg, "transfer", "bt709")

    def forward(self, x_lin, theta, lut, grain=True, pointwise_only=False):
        """pointwise_only skips the spatial ops (sharpen, grain, chroma NR). Everything
        before them is a per-pixel colour map, which means it can be BAKED into a single 3D
        LUT -- the whole look becomes one texture lookup on device (see bench_distill.py
        --phase bake)."""
        t = split_theta(theta)
        # normalize FIRST, then expose. The old order applied exposure before dividing
        # by amax(x), which cancelled it exactly: (x*2^e)/amax(x*2^e) == x/amax(x).
        # Scale floor of 1.0, not 1e-2. With 1e-2 an image whose linear max is 0.7 gets
        # divided by 0.7, i.e. force-brightened 1.43x, and the model CANNOT undo it: the
        # tone curve is normalised so it always maps 1 -> 1, so the brightest pixel always
        # floor. A floor of 1.0 makes it a no-op for in-range data and only scales down
        # genuinely over-range RAW.
        scale = x_lin.amax(dim=(1, 2, 3), keepdim=True).detach().clamp(min=1.0)
        x = x_lin / scale
        exposure = EXPOSURE_STOPS * torch.tanh(t["exposure"] / EXPOSURE_STOPS)
        black = BLACK_MAX * torch.sigmoid(t["black"] - 3.0)
        x = x * torch.exp2(exposure)[:, :, None, None]
        x = (x - black[:, :, None, None]).clamp(min=0).clamp(max=4.0)
        tone = TONE_RANGE * torch.tanh(t["tone"] / TONE_RANGE)
        x = self.curve(x.clamp(0, 1), tone)
        knee = torch.sigmoid(t["knee"] - KNEE_OFF)[:, :, None, None]
        x = x * (1 - knee) + (1 - torch.exp(-3 * x)) * knee
        x = encode_transfer(x, self.transfer)
        x = self.huesat(x, t["huesat"])
        sat = 1 + torch.tanh(t["sat"])[:, :, None, None]
        gray = x.mean(1, keepdim=True)
        x = (gray + (x - gray) * sat).clamp(0, 1)
        if lut is not None:
            x = lut3d_apply(lut, x).clamp(0, 1)
        if pointwise_only:
            return x.clamp(0, 1)
        x = unsharp_mask(x, 0.5 + 2 * torch.sigmoid(t["sh_r"]).squeeze(1), torch.tanh(t["sh_a"]).squeeze(1))
        if grain:
            x = apply_grain(x, 0.02 * torch.sigmoid(t["gr_s"] - GRAIN_OFF).squeeze(1),
                            0.5 + 2 * torch.sigmoid(t["gr_z"]).squeeze(1))
        strength = torch.sigmoid(t["cnr"] - CNR_OFF)[:, :, None, None]
        if float(strength.max()) > 1e-3:
            cb_cr_blur = gaussian_blur(x, 0.3 + 2 * torch.sigmoid(t["cnr"]).squeeze(1))
            x = x * (1 - 0.5 * strength) + cb_cr_blur * (0.5 * strength)
        return x.clamp(0, 1)


# ============================== LUT VQ-VAE ==============================

class VectorQuantizer(nn.Module):
    def __init__(self, k, d, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(k, d)
        self.codebook.weight.data.uniform_(-1 / k, 1 / k)
        self.beta = beta

    def forward(self, z):
        b, n, d = z.shape
        flat = z.reshape(-1, d)
        dist = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(1))
        idx = dist.argmin(1)
        zq = self.codebook(idx).view(b, n, d)
        loss = F.mse_loss(zq.detach(), z) + self.beta * F.mse_loss(zq, z.detach())
        zq = z + (zq - z).detach()
        return zq, idx.view(b, n), loss


class LUTVQVAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        n = cfg.lut_size
        self.enc = nn.Sequential(
            nn.Conv3d(3, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv3d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv3d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.to_tokens = nn.Linear(128, cfg.lut_tokens * cfg.lut_embed_dim)
        self.vq = VectorQuantizer(cfg.lut_codebook, cfg.lut_embed_dim)
        self.from_tokens = nn.Linear(cfg.lut_tokens * cfg.lut_embed_dim, 128 * 4 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose3d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose3d(32, 3, 4, 2, 1),
        )
        self.n = n
        self.cfg = cfg

    def encode(self, lut):
        h = self.enc(lut).flatten(1)
        z = self.to_tokens(h).view(lut.shape[0], self.cfg.lut_tokens, self.cfg.lut_embed_dim)
        zq, idx, vq_loss = self.vq(z)
        return zq, idx, vq_loss

    def decode(self, zq):
        b = zq.shape[0]
        h = self.from_tokens(zq.flatten(1)).view(b, 128, 4, 4, 4)
        out = self.dec(h)
        out = F.interpolate(out, size=(self.n, self.n, self.n), mode="trilinear", align_corners=True)
        return torch.sigmoid(out)

    def decode_from_ids(self, idx):
        zq = self.vq.codebook(idx)
        return self.decode(zq)

    def forward(self, lut):
        zq, idx, vq_loss = self.encode(lut)
        rec = self.decode(zq)
        return rec, idx, vq_loss


# ============================== B. style director (VLM) ==============================

def build_style_director(cfg: Config):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    quant_kwargs = {}
    if cfg.use_4bit:
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    else:
        quant_kwargs["dtype"] = torch.bfloat16

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.director_ckpt, revision=cfg.director_revision, **quant_kwargs)
    processor = AutoProcessor.from_pretrained(cfg.director_ckpt,
                                              revision=cfg.director_revision)
    if cfg.use_4bit:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=cfg.grad_ckpt)
    elif cfg.grad_ckpt:
        enable_grad_ckpt(base, "director")
    lora_cfg = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    base = get_peft_model(base, lora_cfg)
    hidden = base.config.text_config.hidden_size if hasattr(base.config, "text_config") else base.config.hidden_size
    return base, processor, hidden


class StyleDirector(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.vlm, self.processor, hidden = build_style_director(cfg)
        # LLM hidden states have large, layer-dependent magnitude; normalising here keeps
        # head gradients on a sane scale instead of tracking the backbone's activation size.
        self.pool_norm = nn.LayerNorm(hidden)
        self.theta_head = nn.Linear(hidden, cfg.theta_dim)
        # NEAR-neutral init, deliberately not exactly zero. theta=0 is the neutral look, so
        # zero-init is tempting -- but it makes grad wrt pooled = W^T @ grad_theta = 0, and
        # LoRA's B is zero-initialized too, so the ENTIRE VLM receives zero gradient and the
        # model can only learn theta_head.bias: a prompt-independent average look. Hence a
        # small random W: ||pooled||~sqrt(hidden) after LayerNorm, so std=1e-3 gives theta
        # std ~0.05 (a 1.04x exposure swing, visually neutral) while gradient flows upstream.
        nn.init.normal_(self.theta_head.weight, std=cfg.head_init_std)
        nn.init.zeros_(self.theta_head.bias)
        self.lut_head = nn.Linear(hidden, cfg.lut_tokens * cfg.lut_codebook)
        self.style_head = nn.Linear(hidden, cfg.style_dim)
        self.cfg = cfg

    def forward(self, images, prompts):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": p}]}]
                for im, p in zip(images, prompts)]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.vlm.device)
        out = self.vlm(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        mask = inputs["attention_mask"]
        if getattr(self.cfg, "pool", "last") == "mean":
            m = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            # LAST-TOKEN pooling. The sequence is dominated by image tokens (256-1024 of them
            # vs ~10 text tokens), so a mean pool makes the prompt only ~2-4% of the vector
            # prompts rendered to dE 0.43, i.e. indistinguishable. In a causal LM the final
            # token has attended over image AND prompt, so it is the natural summary.
            last_idx = mask.cumsum(1).argmax(1)
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx]
        pooled = self.pool_norm(pooled.float()).to(pooled.dtype)
        theta = self.theta_head(pooled)
        lut_logits = self.lut_head(pooled).view(-1, self.cfg.lut_tokens, self.cfg.lut_codebook)
        s = self.style_head(pooled)
        return theta, lut_logits, s


# ============================== director-output cache ==============================
# theta/LUT/s are cacheable per style (same idea as shipping precomputed style packs):
# run the VLM once per (camera, style) via `--phase cache_style`, then the renderer
# phase just loads these tensors from disk and never has to hold the 4B VLM in memory.

class StyleCache:
    def __init__(self, cache_dir, strict=False):
        self.cache_dir = cache_dir
        self._mem = {}
        self.hits = 0
        self.misses = 0
        self.missing_keys = set()
        self.strict = strict
        self._warned = False

    def get(self, key, cfg):
        if key not in self._mem:
            path = os.path.join(self.cache_dir, f"{key}.pt")
            if os.path.exists(path):
                d = torch.load(path, map_location="cpu")
                self._mem[key] = (d["theta"], d["s"], d.get("lut"), True)
            else:
                self._mem[key] = (torch.zeros(cfg.theta_dim), torch.zeros(cfg.style_dim), None, False)
        theta, s, lut, ok = self._mem[key]
        if ok:
            self.hits += 1
        else:
            self.misses += 1
            self.missing_keys.add(key)
            if self.strict:
                raise KeyError(
                    f"style cache miss: '{key}' not in {self.cache_dir}. "
                    f"Rebuild with --phase cache_style.")
            if not self._warned:
                print(f"[style cache] miss: '{key}' (rebuild with --phase cache_style)")
                self._warned = True
        return theta, s, lut

    def report(self):
        tot = self.hits + self.misses
        if tot == 0:
            return
        print(f"[style cache] {self.hits}/{tot} hits, {self.misses} misses "
              f"({len(self.missing_keys)} distinct keys missing)")
        if self.missing_keys:
            ex = sorted(self.missing_keys)[:5]
            print(f"  missing e.g.: {ex}")


def fetch_style_batch(cache, keys, device, cfg):
    thetas, ss, luts = [], [], []
    for k in keys:
        theta, s, lut = cache.get(k, cfg)
        thetas.append(theta)
        ss.append(s)
        luts.append(lut)
    theta = torch.stack(thetas).to(device)
    s = torch.stack(ss).to(device)
    lut = None
    if cfg.use_lut and any(l is not None for l in luts):
        # Per-sample fallback. Requiring every entry to have a LUT meant one stale cache file
        # that does nothing. Missing entries get identity instead.
        n = cfg.lut_size
        lin = torch.linspace(0, 1, n)
        ident = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij")[::-1], 0)
        lut = torch.stack([l if l is not None else ident for l in luts]).to(device)
    return theta, s, lut


def precompute_style_cache(cfg: Config, manifest, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sync_for(cfg)
    director = StyleDirector(cfg).to(device).eval()
    load_director(director, cfg, required=True)
    lut_vqvae = LUTVQVAE(cfg).to(device).eval() if cfg.use_lut else None
    if lut_vqvae is not None:
        ck = f"{cfg.out_dir}/lut_vqvae.pt"
        if os.path.exists(ck):
            lut_vqvae.load_state_dict(torch.load(ck, map_location=device))
            print(f"[cache_style] loaded {ck}")
        else:
            print(f"[cache_style] WARNING: {ck} missing; LUT tokens will be meaningless.\n"
                  f"  Run: python camera_look_isp.py --phase lut_vqvae --manifest <manifest>")
    items = split_manifest(load_manifest(manifest), cfg.split, cfg.split_frac)
    keys = [it.get("style_key", f"{it.get('camera', 'unk')}_{it.get('prompt', '')}") for it in items]
    os.makedirs(out_dir, exist_ok=True)
    seen = set()
    warned_base = False
    # Group items by style so each cached entry can be averaged over several images.
    by_key = {}
    for it, key in zip(items, keys):
        by_key.setdefault(key, []).append(it)

    for ki, (key, group) in enumerate(by_key.items(), 1):
        if "base_path" not in group[0] and not warned_base:
            print("[cache_style] WARNING: manifest has no 'base_path'; falling back to the "
                  "styled target as VLM input. The director was trained on the UNSTYLED base, "
                  "so this is out-of-distribution and theta will be wrong. Regenerate the "
                  "dataset with build_dataset.py.")
            warned_base = True
        thetas, ss, luts = [], [], []
        for it in group[:cfg.cache_avg]:
            src_path = it.get("base_path", it["jpeg_path"])
            # (director_img_size, director_img_size), while training and eval feed the native
            # crop -- a different resolution AND aspect ratio, so the VLM saw a different image
            # than it was trained on and produced theta that did not match. That mismatch was
            # the entire cached-vs-live gap.
            img = Image.open(src_path).convert("RGB")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                theta, lut_logits, s = director([img], [it.get("prompt", "")])
                if cfg.tame_limit > 0:
                    theta = tame_theta(theta, cfg.tame_limit)
                thetas.append(theta[0].float().cpu())
                ss.append(s[0].float().cpu())
                if cfg.use_lut and lut_vqvae is not None:
                    luts.append(lut_vqvae.decode_from_ids(
                        lut_logits.float().argmax(-1))[0].float().cpu())
        # One entry per style is reused across every scene, so averaging a few images gives a
        # look that generalises instead of one tied to whichever image happened to be first.
        theta_m = torch.stack(thetas).mean(0)
        s_m = torch.stack(ss).mean(0)
        lut_m = torch.stack(luts).mean(0) if luts else None
        torch.save({"theta": theta_m, "s": s_m, "lut": lut_m},
                   os.path.join(out_dir, f"{key}.pt"))
        seen.add(key)
        if ki % 10 == 0 or ki == len(by_key):
            print(f"[cache_style] {ki}/{len(by_key)} styles  (avg over "
                  f"{min(len(group), cfg.cache_avg)} images)  {key}")

    # coverage check: every style_key in the manifest must have an entry, or the renderer
    want = set(keys)
    have = {f[:-3] for f in os.listdir(out_dir) if f.endswith(".pt")}
    missing = sorted(want - have)
    print(f"\n[cache_style] {len(want)} style keys in manifest, {len(have)} cached")
    if missing:
        print(f"[cache_style] WARNING: {len(missing)} keys have no entry, e.g. {missing[:5]}")
        print("  Those samples will render with theta=0 (unstyled base).")
    else:
        print("[cache_style] coverage complete")


# ============================== C2. one-step renderer (FLUX.2 klein) ==============================

def build_renderer_backbone(cfg: Config):
    from diffusers import Flux2Transformer2DModel, AutoencoderKL
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    quant_kwargs = {}
    if cfg.use_4bit:
        from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
        quant_kwargs["quantization_config"] = DiffusersBnBConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    else:
        quant_kwargs["torch_dtype"] = torch.bfloat16

    transformer = Flux2Transformer2DModel.from_pretrained(
        cfg.renderer_ckpt, subfolder="transformer", **quant_kwargs)
    vae = AutoencoderKL.from_pretrained(cfg.renderer_ckpt, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.requires_grad_(False)
    if cfg.use_4bit:
        # prepare_model_for_kbit_training assumes the transformers API internally; on a
        # diffusers model its checkpointing call can fail the same way, so fall back.
        try:
            transformer = prepare_model_for_kbit_training(
                transformer, use_gradient_checkpointing=cfg.grad_ckpt)
        except AttributeError as e:
            print(f"[renderer] prepare_model_for_kbit_training: {e}; continuing without it")
            if cfg.grad_ckpt:
                enable_grad_ckpt(transformer, "renderer")
    lora_cfg = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                         "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
                         "proj_mlp", "proj_out"],
    )
    transformer = get_peft_model(transformer, lora_cfg)
    if not cfg.use_4bit and cfg.grad_ckpt:
        enable_grad_ckpt(transformer, "renderer")
    return transformer, vae


def _install_meta_gate_hook(transformer, meta_gate, m_holder):
    base = transformer.base_model.model if hasattr(transformer, "base_model") else transformer
    target = getattr(base, "time_text_embed", None)
    if target is None:
        return

    def hook(module, inp, out):
        return out + meta_gate(m_holder["m"])

    target.register_forward_hook(hook)


class BilateralGridHead(nn.Module):
    def __init__(self, in_dim, grid_res):
        super().__init__()
        self.grid_res = grid_res
        self.affine_limit = 0.15
        self.proj = nn.Linear(in_dim, grid_res * 12)
        # Identity init: A = I, t = 0, so the affine branch is a no-op at step 0 and the
        # model learns a deviation from "pass the guide through". Now that composite() is
        # in the training path, a random init would emit garbage from the first step.
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            bias = torch.zeros(12, grid_res)
            bias[0] = 1.0   # A[0,0]
            bias[4] = 1.0   # A[1,1]
            bias[8] = 1.0   # A[2,2]
            self.proj.bias.copy_(bias.reshape(-1))

    def forward(self, pooled):
        b = pooled.shape[0]
        g = self.proj(pooled).view(b, 12, self.grid_res, 1, 1)
        if self.affine_limit > 0:
            # residual of only |R|=0.0065 -- far more than the residual can explain, which
            # implicates this branch. It is also the least-tested path, having only started
            # receiving gradient when training was routed through composite().
            ident = torch.zeros(12, device=g.device, dtype=g.dtype)
            ident[0] = ident[4] = ident[8] = 1.0
            ident = ident.view(1, 12, 1, 1, 1)
            g = ident + self.affine_limit * torch.tanh((g - ident) / self.affine_limit)
        return g.expand(-1, -1, -1, self.grid_res, self.grid_res).contiguous()


def bilateral_slice(grid, guide):
    b, c, dz, dh, dw = grid.shape
    h, w = guide.shape[-2:]
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, h, device=guide.device),
        torch.linspace(-1, 1, w, device=guide.device), indexing="ij")
    ys = ys[None].expand(b, -1, -1)
    xs = xs[None].expand(b, -1, -1)
    zguide = guide.mean(1) * 2 - 1
    coords = torch.stack([xs, ys, zguide], -1).unsqueeze(1)
    out = F.grid_sample(grid, coords, mode="bilinear", align_corners=True)
    return out.squeeze(2)


def apply_affine(coeffs, guide):
    A = coeffs[:, :9].reshape(-1, 3, 3, *guide.shape[-2:])
    t = coeffs[:, 9:12]
    out = torch.einsum("bijhw,bjhw->bihw", A, guide) + t
    return out


def unwrap_peft_model(m):
    """Reach the real nn.Module through PEFT / DDP wrappers for signature inspection."""
    for _ in range(4):
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model
        elif hasattr(m, "module"):
            m = m.module
        else:
            break
    return m


def build_flux_ids(h, w, txt_len, device, dtype, n_axes=3):
    """RoPE position ids. Flux1 uses 3 axes (16,56,56); Flux2 uses 4 (32,32,32,32), so the
    axis count must come from the model config -- assuming 3 raises IndexError inside
    pos_embed. Row goes in the second-to-last axis, column in the last, rest zero."""
    ids = torch.zeros(h, w, n_axes, device=device, dtype=dtype)
    ids[..., -2] = torch.arange(h, device=device, dtype=dtype)[:, None]
    ids[..., -1] = torch.arange(w, device=device, dtype=dtype)[None, :]
    return ids.reshape(h * w, n_axes), torch.zeros(txt_len, n_axes, device=device, dtype=dtype)


class OneStepRenderer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.transformer, self.vae = build_renderer_backbone(cfg)
        z_ch = self.vae.config.latent_channels
        self.z_ch = z_ch
        tcfg = getattr(self.transformer, "config", None)
        self.in_ch = getattr(tcfg, "in_channels", 64) if tcfg is not None else 64
        self.out_ch = getattr(tcfg, "out_channels", None) or self.in_ch
        # Flux packs 2x2 latent patches into one token, so in_channels is latent_ch * 4.
        # Infer the factor instead of assuming: p=2 when it divides cleanly, else no packing.
        self.patch = 2 if self.in_ch % 4 == 0 and self.in_ch // 4 >= 1 else 1
        self.adapter_ch = self.in_ch // (self.patch ** 2)
        self.input_adapter = nn.Conv2d(2 * z_ch, self.adapter_ch, 1)
        nn.init.zeros_(self.input_adapter.weight)
        nn.init.zeros_(self.input_adapter.bias)
        # map the transformer output back to VAE latent channels; zero-init so the residual
        # starts at 0 and the model begins as "guide unchanged"
        self.output_adapter = nn.Conv2d(self.out_ch // (self.patch ** 2), z_ch, 1)
        nn.init.zeros_(self.output_adapter.weight)
        nn.init.zeros_(self.output_adapter.bias)
        # Read these from the checkpoint rather than hard-coding: Flux1 and Flux2 differ on
        # every one of them (joint_attention_dim 4096 vs 15360, rope axes 3 vs 4).
        self.jad = getattr(tcfg, "joint_attention_dim", cfg.joint_attention_dim) if tcfg else cfg.joint_attention_dim
        self.n_axes = len(getattr(tcfg, "axes_dims_rope", (16, 56, 56))) if tcfg else 3
        inner_dim = ((getattr(tcfg, "num_attention_heads", 24) *
                      getattr(tcfg, "attention_head_dim", 128)) if tcfg else 3072)
        if is_main():
            print(f"[renderer] in_ch {self.in_ch} latent {z_ch} patch {self.patch} "
                  f"| joint_attention_dim {self.jad} rope_axes {self.n_axes} inner_dim {inner_dim}")
        self.style_to_ctx = nn.Linear(cfg.style_dim, cfg.style_ctx_tokens * self.jad)
        self.paramnet = ParamNet()
        self.meta_gate = nn.Sequential(nn.Linear(256, 512), nn.SiLU(), nn.Linear(512, inner_dim))
        nn.init.zeros_(self.meta_gate[-1].weight)
        nn.init.zeros_(self.meta_gate[-1].bias)
        self._m_holder = {"m": None}
        _install_meta_gate_hook(self.transformer, self.meta_gate, self._m_holder)
        self.bgrid_head = BilateralGridHead(cfg.style_dim, cfg.bgrid_res)
        self._fwd_params = None
        self.cfg = cfg

    def _accepts(self, name):
        """The transformer signature differs across diffusers versions (and Flux1 vs Flux2),
        so filter kwargs at runtime rather than guessing. 'height'/'width' in particular are
        NOT parameters of Flux2Transformer2DModel.forward."""
        if self._fwd_params is None:
            import inspect
            m = unwrap_peft_model(self.transformer)
            try:
                self._fwd_params = set(inspect.signature(m.forward).parameters)
            except (TypeError, ValueError):
                self._fwd_params = set()
        return name in self._fwd_params

    def encode(self, x):
        return self.vae.encode(x * 2 - 1).latent_dist.sample() * self.vae.config.scaling_factor

    def decode(self, z):
        return ((self.vae.decode(z / self.vae.config.scaling_factor).sample) + 1) / 2

    def forward(self, guide, x_lin_down, s, meta, timestep):
        # meta is RAW exif (b,6); the embedding is computed here so ParamNet participates in
        # meta gate received no information at all and camera conditioning was dead.
        self._m_holder["m"] = self.paramnet(meta)
        zg = self.encode(guide)
        zx = self.encode(x_lin_down.clamp(0, 1))
        feat = self.input_adapter(torch.cat([zg, zx], 1))
        b, c, h, w = feat.shape
        p = self.patch
        if h % p or w % p:
            feat = F.pad(feat, (0, w % p, 0, h % p))
            b, c, h, w = feat.shape
        ph, pw = h // p, w // p
        # (b,c,h,w) -> (b, ph*pw, c*p*p) : each p x p patch becomes one token
        tok = feat.reshape(b, c, ph, p, pw, p).permute(0, 2, 4, 1, 3, 5).reshape(b, ph * pw, c * p * p)
        ctx = self.style_to_ctx(s).view(b, self.cfg.style_ctx_tokens, self.jad)

        kwargs = {"hidden_states": tok, "encoder_hidden_states": ctx, "return_dict": False}
        if self._accepts("timestep"):
            kwargs["timestep"] = timestep
        if self._accepts("img_ids") or self._accepts("txt_ids"):
            img_ids, txt_ids = build_flux_ids(ph, pw, ctx.shape[1], tok.device, tok.dtype, self.n_axes)
            if self._accepts("img_ids"):
                kwargs["img_ids"] = img_ids
            if self._accepts("txt_ids"):
                kwargs["txt_ids"] = txt_ids
        if self._accepts("guidance"):
            kwargs["guidance"] = torch.zeros(b, device=tok.device, dtype=tok.dtype)
        out = self.transformer(**kwargs)
        out = out[0] if isinstance(out, (tuple, list)) else getattr(out, "sample", out)

        oc = out.shape[-1] // (p * p)
        img = out.reshape(b, ph, pw, oc, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, oc, ph * p, pw * p)
        pred = self.output_adapter(img)
        residual = self.decode(zg + pred) - guide
        grid = self.bgrid_head(s)
        return residual, grid


# ============================== D. full-res compositor ==============================

def guided_upsample(t_down, size):
    return F.interpolate(t_down, size=size, mode="bilinear", align_corners=False)


def composite(guide_full, residual_down, grid, guide_down):
    full_h, full_w = guide_full.shape[-2:]
    residual_full = guided_upsample(residual_down, (full_h, full_w))
    coeffs = bilateral_slice(grid, guide_down)
    coeffs_full = guided_upsample(coeffs, (full_h, full_w))
    a_out = apply_affine(coeffs_full, guide_full)
    return (guide_full + residual_full) * 0.5 + a_out * 0.5


# ============================== losses ==============================

def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb):
    lin = srgb_to_linear(rgb.clamp(0, 1).float())
    r, g, b = lin.unbind(1)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    xn, yn, zn = 0.95047, 1.0, 1.08883
    def f(t):
        d = 6 / 29
        return torch.where(t > d ** 3, t.clamp(min=1e-4) ** (1 / 3), t / (3 * d * d) + 4 / 29)
    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    L = 116 * fy - 16
    A = 500 * (fx - fy)
    B = 200 * (fy - fz)
    return torch.stack([L, A, B], 1)


def _safe_hue(b, a, eps=1e-5):
    """atan2(0,0) is finite forward but NaN backward (0/0). Achromatic pixels (grey,
    black, white) hit exactly that, so nudge them onto the +a axis before atan2.
    Hue is undefined for neutrals anyway, and dHp is weighted by sqrt(C1p*C2p) -> 0
    there, so forcing zero gradient on those pixels is the correct behaviour."""
    neutral = (a * a + b * b) < eps * eps
    a = torch.where(neutral, torch.ones_like(a), a)
    b = torch.where(neutral, torch.zeros_like(b), b)
    return torch.atan2(b, a) % (2 * math.pi)


def delta_e00(rgb1, rgb2, eps=1e-6, per_sample=False):
    lab1, lab2 = rgb_to_lab(rgb1.float()), rgb_to_lab(rgb2.float())
    L1, A1, B1 = lab1.unbind(1)
    L2, A2, B2 = lab2.unbind(1)
    C1 = torch.sqrt(A1 ** 2 + B1 ** 2 + eps)
    C2 = torch.sqrt(A2 ** 2 + B2 ** 2 + eps)
    Cb = (C1 + C2) / 2
    # (Cb^7)/(Cb^7+25^7) rewritten as u/(u+1) with u=(Cb/25)^7 to avoid ~1e14 magnitudes
    u = (Cb / 25.0).clamp(min=0, max=8.0) ** 7
    G = 0.5 * (1 - torch.sqrt((u / (u + 1)).clamp(min=0, max=1) + eps))
    a1p, a2p = A1 * (1 + G), A2 * (1 + G)
    C1p = torch.sqrt(a1p ** 2 + B1 ** 2 + eps)
    C2p = torch.sqrt(a2p ** 2 + B2 ** 2 + eps)
    h1p = _safe_hue(B1, a1p)
    h2p = _safe_hue(B2, a2p)
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = torch.where(dhp > math.pi, dhp - 2 * math.pi, dhp)
    dhp = torch.where(dhp < -math.pi, dhp + 2 * math.pi, dhp)
    dHp = 2 * torch.sqrt((C1p * C2p).clamp(min=eps)) * torch.sin(dhp / 2)
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = torch.where((C1p * C2p).abs() < eps, hsum,
          torch.where((h1p - h2p).abs() <= math.pi, hsum / 2,
          torch.where(hsum < 2 * math.pi, (hsum + 2 * math.pi) / 2, (hsum - 2 * math.pi) / 2)))
    T = (1 - 0.17 * torch.cos(hbp - math.pi / 6) + 0.24 * torch.cos(2 * hbp)
         + 0.32 * torch.cos(3 * hbp + math.pi / 30) - 0.20 * torch.cos(4 * hbp - 63 * math.pi / 180))
    dtheta = (math.pi / 6) * torch.exp(-(((hbp * 180 / math.pi - 275) / 25) ** 2))
    up = (Cbp / 25.0).clamp(min=0, max=8.0) ** 7
    Rc = 2 * torch.sqrt((up / (up + 1)).clamp(min=0, max=1) + eps)
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / torch.sqrt(20 + (Lbp - 50) ** 2 + eps)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -torch.sin(2 * dtheta) * Rc
    dE2 = ((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
           + Rt * (dCp / Sc) * (dHp / Sh))
    dE = torch.sqrt(dE2.clamp(min=eps))
    return dE.flatten(1).mean(1) if per_sample else dE.mean()


def soft_histogram(x, bins=32, sigma=0.02, idx=None):
    x = x.float().reshape(x.shape[0], x.shape[1], -1)
    if idx is not None:
        x = x[:, :, idx]
    centers = torch.linspace(0, 1, bins, device=x.device)
    diff = x.unsqueeze(-1) - centers.view(1, 1, 1, bins)
    w = torch.exp(-0.5 * (diff / sigma) ** 2)
    hist = w.sum(2)
    return hist / hist.sum(-1, keepdim=True).clamp(min=1e-6)


def histogram_loss(y, guide, max_pixels=16384):
    n = y.shape[-2] * y.shape[-1]
    # one shared subset: sampling each side independently would compare different pixels
    idx = torch.randperm(n, device=y.device)[:max_pixels] if n > max_pixels else None
    return F.l1_loss(soft_histogram(y, idx=idx), soft_histogram(guide, idx=idx))


def color_consistency_loss(y, guide):
    return delta_e00(y, guide) + histogram_loss(y, guide)


class LPIPSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import lpips
            self.net = lpips.LPIPS(net="vgg")
            self.ok = True
        except Exception:
            self.ok = False

    def forward(self, x, y):
        if not self.ok:
            return F.l1_loss(x, y)
        return self.net(x * 2 - 1, y * 2 - 1).mean()


def monotonic_penalty(base_val, plus_val, minus_val):
    return F.relu(base_val - plus_val).mean() + F.relu(minus_val - base_val).mean()


# ============================== dataset ==============================

class RawJPEGPairDataset(Dataset):
    def __init__(self, manifest_path, crop=None, require_raw=True,
                 split="all", split_frac=0.15, split_seed=0):
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"manifest not found: {manifest_path}\n"
                f"  Generate it first, e.g.:\n"
                f"    python make_manifest.py /mnt/data/raws /mnt/data/jpegs {manifest_path} mycam \"my look\"")
        self.items = split_manifest(load_manifest(manifest_path),
                                    split, split_frac, split_seed)
        if len(self.items) == 0:
            raise ValueError(
                f"manifest is empty: {manifest_path}\n"
                f"  0 entries, so there is nothing to train on.\n"
                f"  If you downloaded FiveK + HaldCLUT, build pairs from them:\n"
                f"    python build_dataset.py --scenes 300 --looks 12 --crops 4\n"
                f"  If you have your own RAW+JPEG pairs, check they share basenames\n"
                f"  (IMG_0001.CR2 <-> IMG_0001.JPG) and sit directly in those folders.")
        if require_raw:
            missing = [it for it in self.items if "raw_path" not in it]
            if missing:
                raise ValueError(
                    f"{len(missing)}/{len(self.items)} manifest entries have no 'raw_path', but this\n"
                    f"  phase (frontend) needs RAW files. Either add RAW files whose basenames match\n"
                    f"  your JPEGs and regenerate the manifest, or skip the frontend phase and start\n"
                    f"  at --phase director_sft (the other three phases only need jpeg_path).")
        bad = [it["jpeg_path"] for it in self.items[:20] if not os.path.exists(it["jpeg_path"])]
        if bad:
            raise FileNotFoundError(
                f"manifest points at files that do not exist, e.g.:\n    {bad[0]}\n"
                f"  The manifest stores absolute paths; if you moved the data after generating it,\n"
                f"  regenerate the manifest.")
        self.crop = crop
        self.require_raw = require_raw
        print(f"[dataset] {len(self.items)} entries from {manifest_path} (require_raw={require_raw})")

    def __len__(self):
        return len(self.items)

    def _load_raw(self, path):
        if path.endswith(".npz"):
            d = np.load(path)
            bayer = d["bayer"].astype(np.float32)
            ccm = d["ccm"].astype(np.float32)
            return torch.from_numpy(bayer).unsqueeze(0).clamp(0, 1), torch.from_numpy(ccm)
        import rawpy
        with rawpy.imread(path) as raw:
            bayer = raw.raw_image_visible.astype(np.float32)
            black = np.array(raw.black_level_per_channel, dtype=np.float32).mean()
            white = float(raw.white_level)
            bayer = (bayer - black) / max(white - black, 1.0)
            ccm = np.array(raw.color_matrix[:3, :3], dtype=np.float32) if raw.color_matrix is not None \
                else np.eye(3, dtype=np.float32)
        return torch.from_numpy(bayer).unsqueeze(0).clamp(0, 1), torch.from_numpy(ccm)

    def __getitem__(self, i):
        it = self.items[i]
        target = Image.open(it["jpeg_path"]).convert("RGB")
        target = torch.from_numpy(np.array(target, dtype=np.float32) / 255.0).permute(2, 0, 1)
        base = None
        if "base_path" in it and os.path.exists(it["base_path"]):
            b = Image.open(it["base_path"]).convert("RGB")
            base = torch.from_numpy(np.array(b, dtype=np.float32) / 255.0).permute(2, 0, 1)
        meta = torch.tensor([it.get("iso", 100.0), it.get("exposure", 1 / 60),
                              it.get("fnumber", 4.0), it.get("focal_length", 35.0),
                              it.get("cfa_code", 0.0) + 1.0, it.get("temp_k", 5500.0)], dtype=torch.float32)
        bayer, ccm = self._load_raw(it["raw_path"]) if self.require_raw else (None, None)
        if self.crop:
            c = self.crop
            _, H, W = target.shape
            H2, W2 = (H // 2) * 2, (W // 2) * 2
            y = random.randint(0, max(0, H2 // 2 - c // 2))
            x = random.randint(0, max(0, W2 // 2 - c // 2))
            target = target[:, 2 * y:2 * y + c, 2 * x:2 * x + c]
            if base is not None:
                base = base[:, 2 * y:2 * y + c, 2 * x:2 * x + c]
            if bayer is not None:
                bayer = bayer[:, 2 * y:2 * y + c, 2 * x:2 * x + c]
        sample = {"target": target, "meta": meta,
                  "prompt": it.get("prompt", ""), "camera": it.get("camera", ""),
                  "style_key": it.get("style_key", f"{it.get('camera', 'unk')}_{it.get('prompt', '')}")}
        if base is not None:
            sample["base"] = base
        if bayer is not None:
            sample["bayer"] = bayer
            sample["ccm"] = ccm
        if "lut_path" in it:
            sample["lut"] = torch.from_numpy(np.load(it["lut_path"]).astype(np.float32))
        if "theta" in it:
            sample["theta_target"] = torch.tensor(it["theta"], dtype=torch.float32)
        return sample


def collate_fn(batch):
    """Only collate keys present in EVERY sample.

    Samples are heterogeneous by design: LUT-look pairs carry 'lut', relative-instruction
    pairs don't, and only some manifests carry 'theta_target'. Keying off batch[0] assumed
    uniformity and raised KeyError as soon as the two kinds were mixed in one batch.
    Downstream code already guards with `if "lut" in batch`, so dropping a non-universal
    key is the correct behaviour rather than an error.
    """
    common = set(batch[0])
    for b in batch[1:]:
        common &= set(b)
    out = {}
    for k in common:
        if k in ("prompt", "camera", "style_key"):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def fit_look(x_lin, target, isp, iters=300, lr=0.05, device="cuda", shared=False, verbose=False):
    """Fit theta so isp(x_lin, theta) ~= target.
    shared=True fits ONE theta for the whole batch (use this to characterize a look)."""
    b = x_lin.shape[0]
    n = 1 if shared else b
    theta = torch.zeros(n, THETA_DIM, device=device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    best, best_theta = float("inf"), theta.detach().clone()
    for it in range(iters):
        opt.zero_grad()
        th = theta.expand(b, -1) if shared else theta
        pred = isp(x_lin, th, None, grain=False)
        loss = F.l1_loss(pred, target) + 0.1 * color_consistency_loss(pred, target)
        loss.backward()
        opt.step()
        sched.step()
        v = float(loss)
        if v < best:
            best, best_theta = v, theta.detach().clone()
        if verbose and it % 50 == 0:
            print(f"      fit iter {it:4d}  loss {v:.4f}")
    return best_theta, best


# ============================== training: phase 1 frontend ==============================

@torch.no_grad()
def evaluate_frontend(cfg: Config, manifest, ckpt, n_batches=6):
    """Decompose the composite training loss and dump side-by-side images.
    A single loss number cannot tell you whether the render is actually right."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrontEnd(cfg).to(device)
    if not os.path.exists(ckpt):
        print(f"checkpoint not found: {ckpt}")
        print(f"  expected something like {cfg.out_dir}/frontend_final.pt")
        return
    sd = torch.load(ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("\n" + "=" * 68)
        print("  CHECKPOINT DOES NOT MATCH THE MODEL")
        print(f"  {len(missing)} missing, {len(unexpected)} unexpected keys.")
        print("  Missing keys are randomly initialised, so these metrics are")
        print("  meaningless - they measure a partly-untrained network.")
        if missing:
            print(f"    missing e.g.: {missing[:4]}")
        if unexpected:
            print(f"    unexpected e.g.: {unexpected[:4]}")
        print("  This usually means the checkpoint predates a model change.")
        print("  Retrain the front-end and evaluate the fresh checkpoint.")
        print("=" * 68 + "\n")
    else:
        print(f"  checkpoint loaded cleanly: {ckpt}")
    model.eval()
    isp = ParametricISP(cfg).to(device).eval()

    ds = RawJPEGPairDataset(manifest, crop=cfg.frontend_crop, require_raw=True,
                            split=cfg.split, split_frac=cfg.split_frac)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4,
                    collate_fn=collate_fn, drop_last=True)

    out_dir = os.path.join(cfg.out_dir, "eval_frontend")
    os.makedirs(out_dir, exist_ok=True)
    tot = {"l1": 0.0, "de": 0.0, "psnr": 0.0, "n": 0}

    for bi, batch in enumerate(dl):
        if bi >= n_batches:
            break
        bayer = batch["bayer"].to(device)
        ccm = batch["ccm"].to(device)
        meta = batch["meta"].to(device)
        target = batch["target"].to(device).float()
        lut = batch["lut"].to(device) if "lut" in batch else None
        theta = torch.zeros(bayer.shape[0], cfg.theta_dim, device=device)

        x_lin, _ = model(bayer, meta, ccm)
        pred = isp(x_lin.float(), theta, lut).float()

        l1 = F.l1_loss(pred, target).item()
        de = delta_e00(pred, target).item()
        mse = F.mse_loss(pred, target).item()
        psnr = -10 * math.log10(max(mse, 1e-12))
        tot["l1"] += l1; tot["de"] += de; tot["psnr"] += psnr; tot["n"] += 1
        print(f"  batch {bi}: L1 {l1:.4f}  dE00 {de:6.2f}  PSNR {psnr:5.2f} dB")

        if bi < 3:
            from PIL import Image as PILImage
            k = min(4, pred.shape[0])
            rows = []
            for i in range(k):
                p = (pred[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                t = (target[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                rows.append(np.concatenate([p, t], axis=1))
            grid = np.concatenate(rows, axis=0)
            path = os.path.join(out_dir, f"batch{bi}_pred_vs_target.png")
            PILImage.fromarray(grid).save(path)
            print(f"    wrote {path}  (left = model, right = target)")

    n = max(tot["n"], 1)
    print(f"\n=== frontend eval over {n} batches ===")
    print(f"  L1        {tot['l1']/n:.4f}   (<0.05 good, >0.15 poor)")
    print(f"  dE00      {tot['de']/n:6.2f}   (<3 good, 3-8 usable, >10 poor)")
    print(f"  PSNR      {tot['psnr']/n:5.2f} dB   (>28 good, <22 poor)")
    print(f"  composite {tot['l1']/n + 0.05*tot['de']/n:.4f}  (what training prints, roughly)")
    print(f"  images in {out_dir} - look at them, the numbers alone can mislead")
    print(f"\n  Compare against probe_isp.py's untrained floor. A trained front-end")
    print(f"  should clearly beat a naive demosaic; if it does not, the problem is")
    print(f"  the training or the checkpoint, not the ISP settings.")


def train_frontend(cfg: Config, manifest):
    local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(cfg.seed)

    model = FrontEnd(cfg).to(device)
    isp = ParametricISP(cfg).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    ds = RawJPEGPairDataset(manifest, crop=cfg.frontend_crop, require_raw=True,
                            split=cfg.split, split_frac=cfg.split_frac)
    sampler = DistributedSampler(ds) if dist.is_initialized() else None
    dl = DataLoader(ds, batch_size=cfg.frontend_batch, sampler=sampler, shuffle=sampler is None,
                     num_workers=8, collate_fn=collate_fn, drop_last=True, pin_memory=True)

    opt = make_optimizer(model.parameters(), cfg.lr_frontend, cfg)
    total_steps = cfg.max_steps if cfg.max_steps else cfg.epochs * len(dl)
    total_steps = max(int(total_steps), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr_frontend, total_steps=total_steps)
    step = 0
    skipped = 0
    done = False
    ema = None
    window = []
    for epoch in range(cfg.epochs):
        if done:
            break
        if sampler:
            sampler.set_epoch(epoch)
        for batch in dl:
            bayer = batch["bayer"].to(device)
            ccm = batch["ccm"].to(device)
            meta = batch["meta"].to(device)
            target = batch["target"].to(device)
            # targets were rendered with this LUT, so the ISP must apply it too;
            # without it the residual error is irreducible and gradients blow up
            lut = batch["lut"].to(device) if "lut" in batch else None
            theta = (batch["theta_target"].to(device) if "theta_target" in batch
                     else torch.zeros(bayer.shape[0], cfg.theta_dim, device=device))

            # bf16 only for the conv net; the ISP is cheap elementwise color math with
            # divisions, a gamma and grid_sample, and is precision-critical -> fp32
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x_lin, m = model(bayer, meta, ccm)
            with torch.autocast("cuda", enabled=False):
                pred = isp(x_lin.float(), theta, lut)
                pred_f, target_f = pred.float(), target.float()
                l1 = F.l1_loss(pred_f, target_f)
                de = delta_e00(pred_f, target_f)
                hist = histogram_loss(pred_f, target_f)
                loss = l1 + 0.05 * (de + hist)

            stepped = False
            gnorm = torch.tensor(float("nan"))
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_frontend)
                if torch.isfinite(gnorm):
                    opt.step()
                    stepped = True
            if not stepped:
                skipped += 1
                if is_main() and skipped <= 5:
                    bad = [n for n, p in unwrap(model).named_parameters()
                           if p.grad is not None and not torch.isfinite(p.grad).all()]
                    print(f"[frontend] non-finite at micro-batch {skipped} (opt step {step}): "
                          f"loss={float(loss):.4f} finite={bool(torch.isfinite(loss))}, "
                          f"|g|={float(gnorm)}, "
                          f"bad grads: {bad[:3] if bad else 'none (forward-side)'}"
                          f"{f' +{len(bad)-3} more' if len(bad) > 3 else ''}")
                opt.zero_grad(set_to_none=True)

            # scheduler advances only alongside a real optimizer step, and never past total_steps
            if stepped:
                step += 1
                lv = loss.item()
                ema = lv if ema is None else 0.98 * ema + 0.02 * lv
                window.append(lv)
                if sched.last_epoch + 1 < total_steps:
                    sched.step()
                if step % cfg.log_every == 0 and is_main():
                    lo, hi = min(window), max(window)
                    print(f"[frontend] step {step}/{total_steps} "
                          f"loss {lv:.4f} ema {ema:.4f} [{lo:.2f}-{hi:.2f}] "
                          f"| L1 {l1.item():.4f} dE {de.item():.2f} hist {hist.item():.4f} "
                          f"| |g| {float(gnorm):.2f} lr {sched.get_last_lr()[0]:.2e} skipped {skipped}")
                    window = []
                if step % cfg.save_every == 0 and is_main():
                    os.makedirs(cfg.out_dir, exist_ok=True)
                    torch.save(unwrap(model).state_dict(), f"{cfg.out_dir}/frontend_{step}.pt")

            if cfg.max_steps and step >= cfg.max_steps:
                done = True
                break

    if is_main():
        os.makedirs(cfg.out_dir, exist_ok=True)
        torch.save(unwrap(model).state_dict(), f"{cfg.out_dir}/frontend_final.pt")
        print(f"[frontend] finished at step {step} ({skipped} skipped)")


# ============================== training: phase 0 LUT tokenizer ==============================

def train_lut_vqvae(cfg: Config, manifest):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = split_manifest(load_manifest(manifest), cfg.split, cfg.split_frac)
    paths = sorted({it["lut_path"] for it in items if "lut_path" in it})
    if not paths:
        raise ValueError(
            "no 'lut_path' entries in the manifest.\n"
            "  Regenerate with build_dataset.py (it writes one .npy per look).")
    luts = torch.stack([torch.from_numpy(np.load(p).astype(np.float32)) for p in paths])
    print(f"[lut-vqvae] {len(luts)} base LUTs from {manifest}")

    model = LUTVQVAE(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    steps = cfg.max_steps or 3000
    bs = min(8, len(luts))
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(luts), (bs,))
        batch = luts[idx].to(device)
        if random.random() < 0.3:
            # blend toward IDENTITY: most real looks are mild, and a codebook trained only on
            # the raw pack covers strong transforms far better than weak ones
            n = cfg.lut_size
            lin = torch.linspace(0, 1, n, device=device)
            ident = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij")[::-1], 0)
            a = torch.rand(bs, 1, 1, 1, 1, device=device)
            batch = (batch * a + ident[None] * (1 - a)).clamp(0, 1)
        elif random.random() < 0.5:  # random compositions widen coverage beyond the base looks
            idx2 = torch.randint(0, len(luts), (bs,))
            a = torch.rand(bs, 1, 1, 1, 1, device=device)
            batch = (batch * a + luts[idx2].to(device) * (1 - a)).clamp(0, 1)
        rec, ids, vq_loss = model(batch)
        loss = F.l1_loss(rec, batch) + vq_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                usage = len(torch.unique(ids))
                # the cube is only a means, what matters is the colour error it produces.
                probe = torch.rand(1, 3, 64, 64, device=device)
                a = lut3d_apply(batch[:1], probe)
                b = lut3d_apply(rec[:1], probe)
                rec_de = float(delta_e00(a.clamp(0, 1), b.clamp(0, 1)))
            flag = "" if rec_de < 2.0 else "   (target <2.0)"
            print(f"[lut-vqvae] step {step}/{steps} loss {loss.item():.5f} "
                  f"recon dE {rec_de:.2f}{flag} codes_used {usage}/{cfg.lut_codebook}")
    os.makedirs(cfg.out_dir, exist_ok=True)
    out = f"{cfg.out_dir}/lut_vqvae.pt"
    torch.save(model.state_dict(), out)
    print(f"[lut-vqvae] saved {out}")


# ============================== training: phase 2 style director ==============================

def train_director_sft(cfg: Config, manifest):
    local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    # The LoRA rank must match the checkpoint BEFORE StyleDirector is constructed --
    # syncing afterwards is too late, the adapters are already the wrong shape.
    if cfg.resume:
        _rp = cfg.resume if isinstance(cfg.resume, str) else f"{cfg.out_dir}/director_final.pt"
        if os.path.exists(_rp):
            sync_cfg_to_ckpt(cfg, _rp)
    director = StyleDirector(cfg).to(device)
    isp = ParametricISP(cfg).to(device)
    lut_vqvae = None
    if cfg.use_lut:
        lut_vqvae = LUTVQVAE(cfg).to(device).eval()
        ck = f"{cfg.out_dir}/lut_vqvae.pt"
        if not os.path.exists(ck):
            raise FileNotFoundError(
                f"--use_lut needs a pretrained tokenizer: {ck} not found.\n"
                f"  python camera_look_isp.py --phase lut_vqvae --manifest <manifest>")
        lut_vqvae.load_state_dict(torch.load(ck, map_location=device))
        lut_vqvae.requires_grad_(False)   # frozen decoder; only the head learns
        print(f"[director-sft] LUT head ON, tokenizer from {ck}")
    if lut_vqvae is not None:
        ck = f"{cfg.out_dir}/lut_vqvae.pt"
        if os.path.exists(ck):
            lut_vqvae.load_state_dict(torch.load(ck, map_location=device))
            lut_vqvae.eval()
        else:
            raise FileNotFoundError(
                f"--use_lut needs a pretrained tokenizer, but {ck} is missing.\n"
                f"  Run first: python camera_look_isp.py --phase lut_vqvae --manifest <manifest>")
    if dist.is_initialized():
        director = DDP(director, device_ids=[local_rank], find_unused_parameters=True)

    ds = RawJPEGPairDataset(manifest, require_raw=False,
                            split=cfg.split, split_frac=cfg.split_frac)
    sampler = DistributedSampler(ds) if dist.is_initialized() else None
    dl = DataLoader(ds, batch_size=cfg.director_batch, sampler=sampler, shuffle=sampler is None,
                     num_workers=8, collate_fn=collate_fn, drop_last=True)

    trainable = [p for p in director.parameters() if p.requires_grad]
    if cfg.lut_only and not cfg.use_lut:
        raise ValueError("--lut_only requires --use_lut: with the LUT path off there "
                         "is nothing left to train once theta is frozen.")
    if cfg.lut_only:
        # Hard guarantee that the working theta behaviour is preserved: freeze everything
        # except the LUT head. pooled is unchanged (LoRA + pool_norm frozen) and theta_head is
        # frozen, so theta is bit-for-bit what the resumed checkpoint produced. Only the
        # residual colour twist is learned. Fast, and it cannot regress what already works.
        for n_, p_ in unwrap(director).named_parameters():
            p_.requires_grad_(n_.startswith("lut_head"))
        if is_main():
            n_tr = sum(p_.numel() for p_ in director.parameters() if p_.requires_grad)
            print(f"[director-sft] --lut_only: {n_tr/1e6:.1f}M trainable (lut_head only); "
                  f"theta output is frozen and cannot regress")
    if cfg.resume:
        # Start from the working theta-only director: the LUT head then only has to learn the
        # residual colour twist that theta cannot express, which is a much shorter run than
        # retraining everything.
        load_director(director, cfg, cfg.resume if isinstance(cfg.resume, str) else None,
                      required=True)
    groups, heads, lora = split_director_params(director, cfg)
    opt = make_optimizer(groups, cfg.lr_director, cfg)
    if is_main():
        nh = sum(p.numel() for p in heads)
        nl = sum(p.numel() for p in lora)
        print(f"[director-sft] {nl/1e6:.1f}M LoRA @ {cfg.lr_director:.1e} | "
              f"{nh/1e6:.1f}M heads @ {cfg.lr_director*cfg.head_lr_mult:.1e}")
    total_steps = cfg.max_steps or max(1, (len(dl) // max(cfg.director_grad_accum, 1)) * cfg.epochs)
    warmup = min(cfg.warmup_steps, max(1, total_steps // 10))

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        prog = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    if is_main():
        print(f"[director-sft] {total_steps} steps, {warmup} warmup, "
              f"effective batch {cfg.director_batch * cfg.director_grad_accum}")
    accum = cfg.director_grad_accum
    step = 0
    warned = False
    ema = None
    window = []
    run_loss, n_micro, n_drop = 0.0, 0, 0
    n_clip, gmax, n_since, clip_prev = 0, 0.0, 0, 0
    gl = gh = 0.0
    clip_tracker = GradNormTracker(factor=cfg.grad_clip_factor)
    parts = {"l1": 0.0, "dE": 0.0, "hist": 0.0, "dnull": 0.0,
             "dE_edit": 0.0, "n_edit": 0, "dE_look": 0.0, "n_look": 0}
    opt.zero_grad()
    for epoch in range(cfg.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        for micro, batch in enumerate(dl):
            # The VLM must see the UNSTYLED base: if it sees the styled target it can copy the
            # answer from its input and ignore the prompt, which is exactly what we are training.
            vlm_src = batch["base"] if "base" in batch else batch["target"]
            if "base" not in batch and not warned and is_main():
                print("[director-sft] WARNING: no 'base_path' in manifest, feeding the styled target "
                      "to the VLM. It can read the look off the image and ignore the prompt. "
                      "Regenerate the dataset with build_dataset.py for real text-conditioned training.")
            imgs = [Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    for t in vlm_src]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prompts = list(batch["prompt"])
                # Prompt dropout: on these samples the correct answer is "no style change".
                # Without this the model can ignore the text and guess the look from the image.
                drop = [random.random() < cfg.prompt_dropout for _ in prompts]
                for i, d in enumerate(drop):
                    if d:
                        prompts[i] = ""
                theta, lut_logits, s = director(imgs, prompts)
                loss = torch.zeros((), device=device, dtype=theta.dtype)
                terms = []
                if "theta_target" in batch:
                    loss = loss + F.mse_loss(theta, batch["theta_target"].to(device))
                    terms.append("theta")
                if cfg.use_lut and "lut" in batch:
                    with torch.no_grad():
                        _, lut_ids, _ = lut_vqvae.encode(batch["lut"].to(device))
                    # AUXILIARY only. The LUT head's primary supervision is the render loss
                    # below, which fires on every batch. This term anchors tokens to the
                    # ground-truth LUT where one exists, and is weighted down because
                    # cross-entropy over 512 codes starts near ln(512)=6.2 against a render
                    # loss of ~1: unweighted it would dominate on look batches and vanish on
                    # the others, making the objective alternate between two different losses.
                    loss = loss + cfg.lut_ce_weight * F.cross_entropy(
                        lut_logits.reshape(-1, cfg.lut_codebook), lut_ids.reshape(-1))
                    terms.append("lut_ce")
                if "base" in batch:
                    tgt = batch["target"].to(device)
                    src = decode_transfer(batch["base"].to(device), cfg.transfer)
                    # where the prompt was dropped, the correct output is the unstyled base
                    if any(drop):
                        m = torch.tensor(drop, device=device).view(-1, 1, 1, 1)
                        tgt = torch.where(m, batch["base"].to(device), tgt)
                    lut_pred = None
                    if cfg.use_lut and lut_vqvae is not None:
                        # Straight-through: forward uses the ARGMAX codes (exactly what
                        # cache_style will emit at inference), backward flows through the
                        # softmax. Supervising the LUT head through the render loss means it
                        # needs no ground-truth LUT tokens, so it also trains on the edit and
                        # expert pairs that carry no LUT at all.
                        probs = F.softmax(lut_logits.float(), dim=-1)
                        cb = lut_vqvae.vq.codebook.weight
                        z_soft = probs @ cb
                        z_hard = cb[probs.argmax(-1)]
                        z = z_soft + (z_hard - z_soft).detach()
                        lut_pred = lut_vqvae.decode(z)
                    rendered = isp(src, theta.float(), lut_pred)
                    l1 = F.l1_loss(rendered, tgt)
                    hist = histogram_loss(rendered, tgt)
                    de_vec = delta_e00(rendered, tgt, per_sample=True)
                    de = de_vec.mean()
                    with torch.no_grad():
                        base_t = batch["base"].to(device)
                        dnull_vec = delta_e00(base_t, tgt, per_sample=True)
                    if cfg.normalize_difficulty:
                        # Relative error, not absolute. dE loss scales with how far the target
                        # is from the base, so a b&w or day-for-night look contributes ~10x the
                        # director learned degenerate theta. Dividing by each sample's own
                        # base->target distance makes every look contribute comparably, so
                        # strong "effect" looks can stay in the set for coverage.
                        de_term = (de_vec / (dnull_vec.detach() + cfg.difficulty_floor)).mean()
                    else:
                        de_term = de / 10.0
                    loss = loss + l1 + 2.0 * de_term + 0.2 * hist
                    # Penalise RAW theta, not the applied value. Bounds stop the OUTPUT going
                    # out of range but do nothing to stop the INPUT growing: past |raw|~3 the
                    # tanh gradient is under 1% and the parameter is frozen at its bound with
                    # no way back. This term keeps theta in the region where it can still learn.
                    loss = loss + cfg.theta_l2 * theta.float().pow(2).mean()
                    parts["l1"] += float(l1)
                    parts["dE"] += float(de)
                    parts["hist"] += float(hist)
                    with torch.no_grad():
                        dnull = float(dnull_vec.mean())
                        parts["dnull"] += dnull
                        is_edit = [k.startswith("edit_") for k in batch["style_key"]]
                        if any(is_edit):
                            parts["dE_edit"] += float(de)
                            parts["n_edit"] += 1
                        else:
                            parts["dE_look"] += float(de)
                            parts["n_look"] += 1
                    terms.append("render")
            if not loss.requires_grad:
                raise RuntimeError(
                    "director loss has no gradient: none of the supervision terms fired.\n"
                    "  Needs at least one of:\n"
                    "    - 'base_path' in the manifest (rendering loss)  <- regenerate with build_dataset.py\n"
                    "    - 'theta' in the manifest (direct regression)\n"
                    "    - --use_lut with a PRETRAINED tokenizer (python camera_look_isp.py --phase lut_vqvae ...)\n"
                    f"  Current manifest keys: {sorted(batch.keys())}")
            if not warned and is_main():
                print(f"[director-sft] loss terms active: {', '.join(terms)}")
                warned = True
            (loss / accum).backward()
            run_loss += float(loss)
            n_micro += 1
            n_drop += sum(drop)
            if (micro + 1) % accum == 0:
                gnorm, thr = adaptive_clip(trainable, clip_tracker, fallback=None)
                n_clip = clip_tracker.n_clipped
                gmax = max(gmax, float(gnorm))
                # must be read BEFORE zero_grad()
                if step % cfg.log_every == cfg.log_every - 1:
                    gl = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in lora if p.grad is not None)) if lora else 0.0
                    gh = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in heads if p.grad is not None)) if heads else 0.0
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                n_since += 1
                avg = run_loss / max(n_micro, 1)
                ema = avg if ema is None else 0.9 * ema + 0.1 * avg
                window.append(avg)
                if step % cfg.log_every == 0 and is_main():
                    k = max(n_micro, 1)
                    lo, hi = min(window), max(window)
                    rate = 100.0 * (n_clip - clip_prev) / max(n_since, 1)
                    tshow = f"{thr:.0f}" if thr else "off"
                    dead = "  <-- VLM NOT LEARNING" if gl < 1e-8 else ""
                    dn = parts["dnull"] / k if parts["dnull"] else 0.0
                    de_now = parts["dE"] / k
                    frac = (1 - de_now / dn) * 100 if dn > 0 else 0.0
                    el = (f" look {parts['dE_look']/max(parts['n_look'],1):.1f}"
                          if parts["n_look"] else "")
                    ee = (f" edit {parts['dE_edit']/max(parts['n_edit'],1):.1f}"
                          if parts["n_edit"] else "")
                    satf = theta_saturation(theta.detach().float())
                    worst = max(satf.values()) if satf else 0.0
                    rawmag = float(theta.detach().float().abs().mean())
                    sat_s = f" sat{{{max(satf, key=satf.get)} {100*worst:.0f}%}}" if satf else ""
                    if worst > 0.30:
                        sat_s += "  !! SATURATING -- theta is pinned; raise theta_l2 or lower head_lr_mult"
                    print(f"[director-sft] epoch {epoch} step {step} "
                          f"loss {avg:.4f} ema {ema:.4f} [{lo:.2f}-{hi:.2f}] "
                          f"| dE {de_now:.2f} of null {dn:.2f} ({frac:+.0f}%){el}{ee} "
                          f"| L1 {parts['l1']/k:.4f} "
                          f"| g_lora {gl:.2e} g_head {gh:.2e} clip {rate:.0f}% "
                          f"lr {sched.get_last_lr()[0]:.2e} |th| {rawmag:.2f}{sat_s}{dead}")
                    window = []
                    clip_prev, gmax, n_since = n_clip, 0.0, 0
                run_loss, n_micro, n_drop = 0.0, 0, 0
                parts = {"l1": 0.0, "dE": 0.0, "hist": 0.0, "dnull": 0.0,
             "dE_edit": 0.0, "n_edit": 0, "dE_look": 0.0, "n_look": 0}
                if step % cfg.save_every == 0 and is_main():
                    save_director(director, cfg, str(step))
                if cfg.max_steps and step >= cfg.max_steps:
                    if is_main():
                        save_director(director, cfg, "final")
                    return
    if is_main():
        save_director(director, cfg, "final")


def infer_lora_rank(path):
    """Read the LoRA rank out of a checkpoint.

    The rank is a training-time preset (--gpu48 uses 32, the default is 16), so any tool
    that rebuilds the model has to agree with whatever trained it. Guessing produces a wall
    of shape-mismatch errors; the checkpoint already knows the answer.
    """
    try:
        sd = torch.load(path, map_location="cpu")
    except Exception:
        return None
    for k, v in sd.items():
        if "lora_A" in k and hasattr(v, "shape") and len(v.shape) == 2:
            return int(v.shape[0])
    return None


def sync_for(cfg, explicit=None, verbose=False):
    """Align cfg with the checkpoint that is about to be loaded, before the model is built."""
    path = explicit if isinstance(explicit, str) else f"{cfg.out_dir}/director_final.pt"
    if os.path.exists(path):
        sync_cfg_to_ckpt(cfg, path, verbose=verbose)
    return cfg


def sync_cfg_to_ckpt(cfg, path, verbose=True):
    """Align cfg with a checkpoint before building the model."""
    r = infer_lora_rank(path)
    if r and r != cfg.lora_rank:
        if verbose:
            print(f"[cfg] checkpoint {os.path.basename(path)} was trained with lora_rank={r} "
                  f"(cfg had {cfg.lora_rank}); using {r}")
        cfg.lora_rank = r
        cfg.lora_alpha = r
    return cfg


def director_trainable_state(director):
    """LoRA params + the three heads. save_pretrained() alone loses the heads,
    and the theta head is what actually produces the look."""
    sd = unwrap(director).state_dict()
    keep = {}
    for k, v in sd.items():
        if "lora" in k.lower() or k.startswith(("theta_head", "lut_head", "style_head", "pool_norm")):
            keep[k] = v.detach().cpu()
    return keep


def save_director(director, cfg, tag):
    os.makedirs(cfg.out_dir, exist_ok=True)
    path = f"{cfg.out_dir}/director_{tag}.pt"
    # Never clobber a previous final without a copy: a fine-tune that turns out worse must
    # not destroy the checkpoint it was warm-started from.
    if tag == "final" and os.path.exists(path):
        prev = f"{cfg.out_dir}/director_final.prev.pt"
        try:
            import shutil as _sh
            _sh.copy2(path, prev)
            print(f"[save] backed up previous final -> {prev}")
        except Exception as e:
            print(f"[save] WARNING: could not back up {path}: {e}")
    state = director_trainable_state(director)
    torch.save(state, path)
    n_head = sum(1 for k in state if k.startswith(("theta_head", "lut_head", "style_head", "pool_norm")))
    print(f"[save] {path}  ({len(state)} tensors, {n_head} head)")
    return path


def load_director(director, cfg, path=None, required=True):
    path = path or f"{cfg.out_dir}/director_final.pt"
    if not os.path.exists(path):
        msg = (f"director checkpoint not found: {path}\n"
               f"  Train it first:  python camera_look_isp.py --phase director_sft --manifest <manifest>\n"
               f"  Without it the theta head is RANDOMLY INITIALIZED and every prompt renders the same.")
        if required:
            raise FileNotFoundError(msg)
        print(f"[load] WARNING: {msg}")
        return False
    state = torch.load(path, map_location="cpu")
    ck_r = infer_lora_rank(path)
    if ck_r and ck_r != cfg.lora_rank:
        raise RuntimeError(
            f"LoRA rank mismatch: {os.path.basename(path)} has rank {ck_r}, this model was "
            f"built with rank {cfg.lora_rank}.\n"
            f"  The rank is a training preset (--gpu48 uses 32, default 16). Call\n"
            f"  sync_cfg_to_ckpt(cfg, path) BEFORE constructing StyleDirector, or pass the\n"
            f"  same preset flag you trained with.")
    missing, unexpected = unwrap(director).load_state_dict(state, strict=False)
    loaded = len(state)
    heads = [k for k in state if k.startswith(("theta_head", "lut_head", "style_head", "pool_norm"))]
    if unexpected:
        print(f"[load] WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    if not heads:
        raise RuntimeError(
            f"{path} contains no head weights. It was written by an old version that only saved\n"
            f"  the VLM LoRA. Retrain director_sft so the theta head is saved too.")
    print(f"[load] {path}: {loaded} tensors ({len(heads)} head tensors)")
    return True


def decode_theta(theta):
    """Raw theta -> the values the ISP actually uses. The heads emit unbounded reals; every
    dimension is squashed into a physical range before use, so printing raw theta is
    misleading (a raw black of -7.5 is really 0.00003)."""
    t = split_theta(theta)
    tone = TONE_RANGE * torch.tanh(t["tone"] / TONE_RANGE)
    steps = F.softplus(tone) + 1e-4
    cum = torch.cumsum(steps, -1)
    cum = cum / cum[:, -1:].clamp(min=1e-6)
    mid = cum[:, cum.shape[1] // 2 - 1]           # where mid-grey lands
    return {
        "exposure": EXPOSURE_STOPS * torch.tanh(t["exposure"][:, 0] / EXPOSURE_STOPS),
        "black": BLACK_MAX * torch.sigmoid(t["black"][:, 0] - 3.0),
        "sat": 1 + torch.tanh(t["sat"][:, 0]),
        "knee": torch.sigmoid(t["knee"][:, 0] - KNEE_OFF),
        "tone_mid": mid,
        "grain": 0.02 * torch.sigmoid(t["gr_s"][:, 0] - GRAIN_OFF),
        "sharpen": torch.tanh(t["sh_a"][:, 0]),
    }


def theta_saturation(theta, tol=0.02):
    """Fraction of theta dimensions sitting at the edge of their squashing function.
    A knob pinned at its bound means the optimiser wanted to go further, which is the
    signature of a degenerate solution being pursued -- catch it here, not in the images."""
    t = split_theta(theta)
    flags = {}
    flags["exposure"] = float((torch.tanh(t["exposure"] / EXPOSURE_STOPS).abs() > 1 - tol).float().mean())
    flags["tone"] = float((torch.tanh(t["tone"] / TONE_RANGE).abs() > 1 - tol).float().mean())
    flags["sat"] = float((torch.tanh(t["sat"]).abs() > 1 - tol).float().mean())
    flags["huesat"] = float((torch.tanh(t["huesat"]).abs() > 1 - tol).float().mean())
    return flags


def tame_theta(theta, limit=1.0):
    """Clamp raw theta at inference, keeping every parameter in a mild band.
    limit=1.0 gives saturation in [0.24, 1.76] and exposure within +/-0.76 stops."""
    return theta.clamp(-limit, limit)


def eval_prompts(cfg: Config, manifest, prompts=None, ckpt=None):
    """Render one image under several natural-language prompts and report how far apart
    the results are. If prompts don't move the output, text conditioning isn't working."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sync_for(cfg, ckpt)
    director = StyleDirector(cfg).to(device).eval()
    isp = ParametricISP(cfg).to(device)
    load_director(director, cfg, ckpt, required=True)

    items = split_manifest(load_manifest(manifest), cfg.split, cfg.split_frac)
    it = items[0]
    src_path = it.get("base_path", it["jpeg_path"])
    img = Image.open(src_path).convert("RGB")
    t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    x_lin = decode_transfer(t, cfg.transfer)

    # from the manifest too: if training prompts work and generic ones collapse, the problem
    # is generalisation across phrasing, not the model or the loss.
    rng = random.Random(0)
    train_prompts, seen_keys = [], set()
    for jt in rng.sample(items, min(len(items), 400)):
        k = jt.get("style_key", "")
        if k in seen_keys:
            continue
        seen_keys.add(k)
        if jt.get("prompt"):
            train_prompts.append(jt["prompt"])
        if len(train_prompts) >= 4:
            break

    generic = prompts or [
        "",
        "make this look like it was shot on a Fujifilm camera",
        "make this warmer",
        "make this cooler",
        "high contrast black and white",
    ]
    prompts = generic + train_prompts
    labels = ["generic"] * len(generic) + ["TRAIN"] * len(train_prompts)

    lut_vq = None
    if cfg.use_lut:
        ckv = f"{cfg.out_dir}/lut_vqvae.pt"
        if os.path.exists(ckv):
            lut_vq = LUTVQVAE(cfg).to(device).eval()
            lut_vq.load_state_dict(torch.load(ckv, map_location=device))
            lut_vq.requires_grad_(False)
            print(f"[eval] LUT path ACTIVE ({ckv})")
        else:
            print(f"[eval] --use_lut set but {ckv} missing; rendering theta-only")
    else:
        print("[eval] LUT path OFF (pass --use_lut to include it)")

    def render(theta, seed=0, lut=None):
        # the ISP adds stochastic grain; fix the seed so dE reflects the LOOK,
        # not the noise realisation, otherwise identical thetas still differ.
        torch.manual_seed(seed)
        return isp(x_lin, theta.float(), lut)

    outs, thetas = [], []
    os.makedirs("eval_out", exist_ok=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for p in prompts:
            theta, lut_logits_e, _ = director([img], [p])
            if cfg.tame_limit > 0:
                theta = tame_theta(theta, cfg.tame_limit)
            # decode the LUT head exactly as cache_style does, so the images shown here match
            # what the deployed path renders
            lut_e = (lut_vq.decode_from_ids(lut_logits_e.float().argmax(-1))
                     if lut_vq is not None else None)
            r = render(theta, lut=lut_e)
            outs.append(r)
            thetas.append(theta.float())
            tag = (p[:40].replace(" ", "_").replace("/", "_") or "EMPTY")
            Image.fromarray((r[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                            ).save(f"eval_out/{tag}.jpg", quality=95)

        # noise floor: same prompt, two different grain seeds. Any dE at or below this
        # is measurement noise, not a real difference between prompts.
        th0, _, _ = director([img], [prompts[1]])
        floor = float(delta_e00(render(th0, 0), render(th0, 1)))
        # and the true zero: identical theta, identical seed
        exact = float(delta_e00(render(th0, 0), render(th0, 0)))

    # Decoded (applied) values, not raw theta: every dimension is squashed before use, so
    # raw numbers are not interpretable (raw black -7.5 applies as ~0.00003).
    print(f"\n{'src':<8}{'prompt':<30}{'exp':>6}{'black':>8}{'sat':>6}{'tone@mid':>9}{'mean':>7}{'chroma':>8}")
    print("-" * 82)
    for lab_, p, th, r in zip(labels, prompts, thetas, outs):
        d = decode_theta(th)
        mean = float(r.mean())
        chroma = float((r - r.mean(1, keepdim=True)).abs().mean())
        flag = ""
        if mean < 0.06:
            flag = "  <-- BLACK"
        elif chroma < 0.01:
            flag = "  <-- GREY"
        print(f"{lab_:<8}{(p[:28] or 'EMPTY'):<30}{float(d['exposure'][0]):6.2f}"
              f"{float(d['black'][0]):8.4f}{float(d['sat'][0]):6.2f}"
              f"{float(d['tone_mid'][0]):9.3f}{mean:7.3f}{chroma:8.4f}{flag}")
    print("  exp: stops (bounded +/-2) | sat: 1.0 unchanged, 0 greyscale")
    print("  tone@mid: where mid-grey lands after the curve. 0.5 neutral, <0.15 crushes to black")
    satf = theta_saturation(torch.cat(thetas, 0))
    pinned = {k: f"{100*v:.0f}%" for k, v in satf.items() if v > 0.15}
    if pinned:
        print(f"  warning: theta pinned at its bounds: {pinned}")
    else:
        print(f"  theta saturation: {{{', '.join(f'{k} {100*v:.0f}%' for k,v in satf.items())}}} (healthy)")
    tr = [m for l_, m in zip(labels, [float(r.mean()) for r in outs]) if l_ == "TRAIN"]
    ge = [m for l_, m in zip(labels, [float(r.mean()) for r in outs])
          if l_ == "generic" and m > 0]
    if tr and ge:
        print(f"\n  mean brightness: TRAIN prompts {np.mean(tr):.3f} | generic {np.mean(ge):.3f}")


    Image.fromarray((t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    ).save("eval_out/00_INPUT.jpg", quality=95)

    print(f"\nwrote {len(outs)+1} renders to eval_out/")
    print(f"\nnoise floor (same prompt, different grain seed): dE {floor:.2f}")
    print(f"exact repeat (same prompt, same seed):           dE {exact:.2f}")
    print(f"\npairwise dE00 between prompts (compare against the {floor:.2f} floor):")
    n = len(prompts)
    real = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(delta_e00(outs[i], outs[j]))
            dt = float((thetas[i] - thetas[j]).abs().mean())
            if d < max(floor, 0.5):
                flag = "  <-- at/below noise floor"
            else:
                flag = ""
                real += 1
            print(f"  {prompts[i][:26]!r:30} vs {prompts[j][:26]!r:30} "
                  f"dE {d:6.2f} dtheta {dt:.4f}{flag}")

    spread = float(torch.stack(thetas).std(0).mean())
    pairs = n * (n - 1) // 2
    print(f"\ntheta spread across prompts: {spread:.4f}")
    print(f"pairs above noise floor: {real}/{pairs}")
    if spread < 1e-3:
        print("  warning: theta identical for every prompt")
    elif real < pairs // 2:
        print("  warning: most prompt pairs are within the noise floor")


# Axis statistics: a scalar that a given instruction should move in a known direction.
# These are the quantities the graded-instruction protocol measures, so optimising them
# directly is optimising the thing we report.
AXIS_STATS = {
    "warm":     (lambda x: float((x[:, 0] - x[:, 2]).mean()), "warmer", "cooler"),
    "bright":   (lambda x: float(x.mean()), "brighter", "darker"),
    "contrast": (lambda x: float(x.std()), "more contrasty", "flatter"),
    "satur":    (lambda x: float((x - x.mean(1, keepdim=True)).abs().mean()),
                 "more saturated", "more muted"),
}
LEVEL_WORDS = ["slightly ", "", "much "]


def monotonicity_reward(stats_pos, stats_neg, stat_neutral):
    """Reward ordered, correctly-signed response to graded instructions.

    This is what RL adds over SFT here. Supervised training sees one (prompt, target) pair
    at a time and has no way to express "'much warmer' must exceed 'slightly warmer'" --
    that is a constraint ACROSS prompts, and it is not differentiable from any single
    sample. The reward below is computed from three renders at once and is exactly the
    controllability metric reported in the benchmark.
    """
    r = 0.0
    # ordering within each direction
    for seq, sign in ((stats_pos, 1.0), (stats_neg, -1.0)):
        d = [sign * (v - stat_neutral) for v in seq]
        for i in range(len(d) - 1):
            r += 1.0 if d[i + 1] >= d[i] - 1e-6 else -1.0     # monotone in magnitude
        r += 1.0 if d[0] > 0 else -1.0                        # correct direction at all
    # opposites must actually oppose
    if (stats_pos[-1] - stat_neutral) * (stats_neg[-1] - stat_neutral) < 0:
        r += 2.0
    else:
        r -= 2.0
    return r


def train_director_rl(cfg: Config, manifest):
    """Short RL pass for CONTROLLABILITY, not fidelity.

    Rewarding -dE to the target would be pointless: that objective is differentiable and
    already optimised directly by SFT. RL earns its place only on rewards that are not
    differentiable from a single sample -- here, monotonic and correctly-signed response
    across graded instructions.

    A supervised anchor term is mixed in so fidelity cannot drift while chasing the reward,
    which also removes the need to hold a frozen reference policy in memory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _rp = cfg.resume if isinstance(cfg.resume, str) else f"{cfg.out_dir}/director_final.pt"
    if os.path.exists(_rp):
        sync_cfg_to_ckpt(cfg, _rp)      # before construction; see train_director_sft
    director = StyleDirector(cfg).to(device)
    # Read from --resume so the source checkpoint and the output directory can differ:
    # RL writes to its own --out_dir and must not need the input to be copied there first.
    load_director(director, cfg, cfg.resume if isinstance(cfg.resume, str) else None,
                  required=True)
    isp = ParametricISP(cfg).to(device)

    # Render WITH the LUT when the deployed system uses one. Otherwise the anchor term
    # compares a theta-only render against the target and pushes theta to do the LUT's job,
    # degrading exactly the theta/LUT split that the LUT stage established.
    lut_vq = None
    if cfg.use_lut:
        ckv = f"{cfg.out_dir}/lut_vqvae.pt"
        if not os.path.exists(ckv) and isinstance(cfg.resume, str):
            ckv = os.path.join(os.path.dirname(cfg.resume), "lut_vqvae.pt")
        if os.path.exists(ckv):
            lut_vq = LUTVQVAE(cfg).to(device).eval()
            lut_vq.load_state_dict(torch.load(ckv, map_location=device))
            lut_vq.requires_grad_(False)
            print(f"[director-rl] LUT active ({ckv})")
        else:
            print("[director-rl] WARNING: --use_lut but no tokenizer found; rendering theta-only. "
                  "The anchor term will push theta to compensate for the absent LUT.")

    items = split_manifest(load_manifest(manifest), cfg.split, cfg.split_frac)
    items = [it for it in items if "base_path" in it]
    if not items:
        raise ValueError("RL needs 'base_path' in the manifest; rebuild with build_dataset.py")
    rng = random.Random(0)

    trainable = [p for p in director.parameters() if p.requires_grad]
    opt = make_optimizer(trainable, cfg.lr_director * cfg.rl_lr_scale, cfg)
    steps = cfg.max_steps or 400
    sigma = cfg.rl_sigma
    print(f"[director-rl] {steps} steps, lr {cfg.lr_director * cfg.rl_lr_scale:.1e}, "
          f"sigma {sigma}, anchor {cfg.rl_anchor}")

    ema_r, ema_a = None, None
    for step in range(1, steps + 1):
        it = rng.choice(items)
        img = Image.open(it["base_path"]).convert("RGB")
        tgt = torch.from_numpy(np.array(Image.open(it["jpeg_path"]).convert("RGB"),
                                        dtype=np.float32) / 255.0
                               ).permute(2, 0, 1)[None].to(device)
        t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0
                             ).permute(2, 0, 1)[None].to(device)
        x_lin = decode_transfer(t, cfg.transfer)
        with torch.no_grad():
            neutral = isp(x_lin, torch.zeros(1, cfg.theta_dim, device=device), None, grain=False)

        axis = rng.choice(list(AXIS_STATS))
        stat_fn, pos_w, neg_w = AXIS_STATS[axis]
        stat_neutral = stat_fn(neutral)

        prompts, thetas = [], []
        for word in (pos_w, neg_w):
            for lv in LEVEL_WORDS:
                prompts.append(f"make this {lv}{word}")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            theta_all, lut_logits_all, _ = director([img] * len(prompts), prompts)
        theta_all = theta_all.float()
        lut_all = None
        if lut_vq is not None:
            with torch.no_grad():
                lut_all = lut_vq.decode_from_ids(lut_logits_all.float().argmax(-1))

        # REINFORCE with group-normalised advantage (GRPO-style). Exploration noise is added
        # to the deterministic head output; the sampled action is detached before the
        # log-prob so the gradient is the policy-gradient term, not a reparameterised one.
        G = cfg.rl_group_size
        rewards, logps = [], []
        for _ in range(G):
            noisy = (theta_all + sigma * torch.randn_like(theta_all)).detach()
            with torch.no_grad():
                stats = [stat_fn(isp(x_lin, noisy[i:i + 1],
                                     None if lut_all is None else lut_all[i:i + 1],
                                     grain=False))
                         for i in range(noisy.shape[0])]
            n_lv = len(LEVEL_WORDS)
            rewards.append(monotonicity_reward(stats[:n_lv], stats[n_lv:], stat_neutral))
            logps.append(-((noisy - theta_all) ** 2).sum() / (2 * sigma ** 2))
        R = torch.tensor(rewards, device=device, dtype=torch.float32)
        adv = (R - R.mean()) / (R.std() + 1e-4)
        pg = -(adv * torch.stack(logps)).mean() / theta_all.numel()

        # supervised anchor on this image's real prompt: keeps fidelity from drifting
        with torch.autocast("cuda", dtype=torch.bfloat16):
            th_a, lut_logits_a, _ = director([img], [it.get("prompt", "")])
        lut_a = (lut_vq.decode_from_ids(lut_logits_a.float().argmax(-1))
                 if lut_vq is not None else None)
        rendered = isp(x_lin, th_a.float(), lut_a)
        anchor = F.l1_loss(rendered, tgt) + 0.1 * delta_e00(rendered, tgt)

        loss = pg + cfg.rl_anchor * anchor
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        ema_r = float(R.mean()) if ema_r is None else 0.95 * ema_r + 0.05 * float(R.mean())
        ema_a = float(anchor) if ema_a is None else 0.95 * ema_a + 0.05 * float(anchor)
        if step % 20 == 0:
            print(f"[director-rl] step {step}/{steps} axis {axis:8} "
                  f"reward {float(R.mean()):+.2f} ema {ema_r:+.2f} | anchor {ema_a:.4f} "
                  f"(max +8.0, dead model 0.0, inverted -4.0)")
        if step % 200 == 0:
            save_director(director, cfg, f"rl_{step}")
    save_director(director, cfg, "final")

RENDERER_HEADS = (
    "input_adapter",
    "output_adapter",
    "style_to_ctx",
    "paramnet",
    "meta_gate",
    "bgrid_head",
)

def save_renderer(renderer, cfg, tag):
    """Saves the DiT LoRA together with the trainable adapters, which
    transformer.save_pretrained() alone would not include."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    sd = unwrap(renderer).state_dict()
    keep = {k: v.detach().cpu() for k, v in sd.items()
            if "lora" in k.lower() or k.startswith(RENDERER_HEADS)}
    path = f"{cfg.out_dir}/renderer_{tag}.pt"
    torch.save(keep, path)
    n_head = sum(1 for k in keep if k.startswith(RENDERER_HEADS))
    print(f"[save] {path}  ({len(keep)} tensors, {n_head} adapter)")
    return path


def load_renderer(renderer, cfg, path=None, required=True):
    path = path or f"{cfg.out_dir}/renderer_final.pt"
    if not os.path.exists(path):
        msg = (f"renderer checkpoint not found: {path}\n"
               f"  Train it first: --phase renderer --stage 1\n"
               f"  Without it the adapters are randomly initialized.")
        if required:
            raise FileNotFoundError(msg)
        print(f"[load] WARNING: {msg}")
        return False
    state = torch.load(path, map_location="cpu")
    heads = [k for k in state if k.startswith(RENDERER_HEADS)]
    if not heads:
        raise RuntimeError(f"{path} has no adapter weights; retrain (old partial checkpoint).")
    unwrap(renderer).load_state_dict(state, strict=False)
    print(f"[load] {path}: {len(state)} tensors ({len(heads)} adapter)")
    return True


def psnr(a, b):
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1))
    return float(10 * torch.log10(1.0 / mse.clamp(min=1e-10)))


def probe_guide(cfg: Config, manifest, n=24, ckpt=None):
    """Why is the guide worse than the untouched base? Four renders, same target.

      neutral   isp(base, theta=0)      -> should equal dE(base,target). If not, the ISP
                                           itself is not identity at theta=0 (transfer /
                                           normalisation bug, see D3.10 / D3.11).
      cached    isp(base, theta_cache)  -> what eval_renderer and the renderer actually use.
      live      isp(base, theta_live)   -> director run on THIS image and prompt, right now.
      fitted    isp(base, theta*)       -> theta optimised directly by gradient descent:
                                           the capacity ceiling of the parametrisation.

    Reading it:
      neutral != base           -> ISP bug, fix before anything else
      live good, cached bad     -> the cache is stale or theta is not reusable across images
      live bad, fitted good     -> the director is the problem (training/eval mismatch)
      fitted also bad           -> theta cannot express these looks; enable the LUT head
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    isp = ParametricISP(cfg).to(device)
    sync_for(cfg, ckpt)
    director = StyleDirector(cfg).to(device).eval()
    load_director(director, cfg, ckpt, required=True)
    cache = StyleCache(cfg.style_cache_dir) if cfg.use_style_cache else None
    lut_vq = None
    if cfg.use_lut:
        ckv = f"{cfg.out_dir}/lut_vqvae.pt"
        if os.path.exists(ckv):
            lut_vq = LUTVQVAE(cfg).to(device).eval()
            lut_vq.load_state_dict(torch.load(ckv, map_location=device))
            lut_vq.requires_grad_(False)
            print(f"[probe] LUT path ACTIVE (tokenizer {ckv})")
        else:
            print(f"[probe] --use_lut set but {ckv} missing; LUT path INACTIVE")
    else:
        print("[probe] LUT path OFF (pass --use_lut to include it; theta-only otherwise)")

    fe = load_frontend(cfg, device) if cfg.use_frontend else None
    items = split_manifest(load_manifest(manifest), cfg.split, cfg.split_frac)
    if fe is not None:
        items = [it for it in items if "raw_path" in it and os.path.exists(it["raw_path"])]
    random.Random(0).shuffle(items)
    items = items[:n]

    print(f"\n{'sample':<26} {'base':>7} {'neutral':>8} {'cached':>8} {'live':>7} {'fitted':>7}")
    print("-" * 68)
    tot = {"base": 0.0, "neutral": 0.0, "cached": 0.0, "live": 0.0, "fitted": 0.0}
    lut_dev = []
    for it in items:
        base = Image.open(it.get("base_path", it["jpeg_path"])).convert("RGB")
        tgt = Image.open(it["jpeg_path"]).convert("RGB")
        to_t = lambda im: torch.from_numpy(np.array(im, dtype=np.float32) / 255.0
                                           ).permute(2, 0, 1)[None].to(device)
        base_t, tgt_t = to_t(base), to_t(tgt)
        x_lin = (frontend_linear(fe, it, device, cfg) if fe is not None
                 else decode_transfer(base_t, cfg.transfer))
        key = it.get("style_key", "")

        d_base = float(delta_e00(base_t, tgt_t))
        # grain=False everywhere: the ISP's stochastic grain adds dE noise that would
        with torch.no_grad():
            d_neutral = float(delta_e00(
                isp(x_lin, torch.zeros(1, cfg.theta_dim, device=device), None, grain=False), tgt_t))
            if cache is not None:
                th_c, _, lut_c = fetch_style_batch(cache, [key], device, cfg)
                d_cached = float(delta_e00(isp(x_lin, th_c, lut_c, grain=False), tgt_t))
            else:
                d_cached = float("nan")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                th_l, lut_logits_l, _ = director([base], [it.get("prompt", "")])
            # Decode the LUT head the same way cache_style does (argmax codes), otherwise the
            # the LUT head -- which is exactly how a no-op LUT run would appear to succeed.
            lut_l = None
            if cfg.use_lut and lut_vq is not None:
                with torch.no_grad():
                    lut_l = lut_vq.decode_from_ids(lut_logits_l.float().argmax(-1))
            d_live = float(delta_e00(isp(x_lin, th_l.float(), lut_l, grain=False), tgt_t))
            if lut_l is not None:
                with torch.no_grad():
                    probe_rgb = torch.rand(1, 3, 64, 64, device=device)
                    lut_dev.append(float(delta_e00(
                        lut3d_apply(lut_l, probe_rgb).clamp(0, 1), probe_rgb.clamp(0, 1))))
        th_f, _ = fit_look(x_lin, tgt_t, isp, iters=250, lr=0.05, device=device)
        with torch.no_grad():
            d_fit = float(delta_e00(isp(x_lin, th_f, None, grain=False), tgt_t))

        for k, v in zip(tot, [d_base, d_neutral, d_cached, d_live, d_fit]):
            tot[k] += v
        print(f"{key[:26]:<26} {d_base:7.2f} {d_neutral:8.2f} {d_cached:8.2f} "
              f"{d_live:7.2f} {d_fit:7.2f}")

    m = {k: v / max(len(items), 1) for k, v in tot.items()}
    print("-" * 68)
    print(f"{'MEAN':<26} {m['base']:7.2f} {m['neutral']:8.2f} {m['cached']:8.2f} "
          f"{m['live']:7.2f} {m['fitted']:7.2f}")
    if cfg.use_lut and lut_vq is not None and lut_dev:
        mean_dev = float(np.mean(lut_dev))
        print(f"\nLUT head: predicted cube deviates from identity by {mean_dev:.2f} dE on average")
        if mean_dev < 0.5:
            print("  warning: predicted LUT is near identity (no-op)")
    if abs(m["neutral"] - m["base"]) > 1.0:
        print(f"\nwarning: ISP not identity at theta=0 "
              f"({m['neutral']:.2f} vs base {m['base']:.2f})")
    if m["cached"] > m["live"] * 1.3:
        print(f"\nwarning: cached theta ({m['cached']:.2f}) much worse than live "
              f"({m['live']:.2f}); rebuild the style cache")


def eval_renderer(cfg: Config, manifest, n_images=40, ckpt=None):
    """Does the renderer actually beat the deterministic guide?

    The guide (parametric ISP alone) is a complete result on its own, so the only question
    that matters is whether adding the diffusion residual moves it CLOSER to the target.
    Reported per image and in aggregate, plus a colour-drift audit (the renderer is supposed
    to add local character, not change colour) and a residual-energy hallucination check.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = OneStepRenderer(cfg).to(device).eval()
    load_renderer(renderer, cfg, ckpt, required=True)
    isp = ParametricISP(cfg).to(device)
    with contextlib.redirect_stdout(io.StringIO()):
        lp = LPIPSLoss().to(device)
    cache = StyleCache(cfg.style_cache_dir) if cfg.use_style_cache else None

    fe = load_frontend(cfg, device) if cfg.use_frontend else None
    items = split_manifest(load_manifest(manifest), cfg.split, cfg.split_frac)
    if fe is not None:
        items = [it for it in items if "raw_path" in it and os.path.exists(it["raw_path"])]
    # по одной паре на сцену: подряд идущие записи относятся к одному кадру,
    # и оценка на них описывает один сюжет, а не выборку
    by_scene = {}
    for it in items:
        by_scene.setdefault(scene_of(it), []).append(it)
    rng_s = random.Random(0)
    items = [rng_s.choice(v) for _, v in sorted(by_scene.items())][:n_images]
    os.makedirs("eval_renderer", exist_ok=True)
    agg = {"g_psnr": 0.0, "p_psnr": 0.0, "g_lpips": 0.0, "p_lpips": 0.0,
           "g_de": 0.0, "p_de": 0.0, "drift": 0.0, "res": 0.0, "base_de": 0.0}
    wins = 0

    print(f"\n{'image':<22} {'guide dE':>9} {'pred dE':>9} {'guide LPIPS':>12} "
          f"{'pred LPIPS':>11} {'drift':>7} {'|R|':>7}")
    print("-" * 82)
    for i, it in enumerate(items):
        base = Image.open(it.get("base_path", it["jpeg_path"])).convert("RGB")
        tgt = Image.open(it["jpeg_path"]).convert("RGB")
        t = lambda im: torch.from_numpy(np.array(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        base_t, tgt_t = t(base), t(tgt)
        x_lin = (frontend_linear(fe, it, device, cfg) if fe is not None
                 else decode_transfer(base_t, cfg.transfer))
        meta = torch.tensor([[it.get("iso", 100.0), it.get("exposure", 1 / 60),
                              it.get("fnumber", 4.0), it.get("focal_length", 35.0),
                              1.0, 5500.0]], dtype=torch.float32, device=device)
        key = it.get("style_key", "")
        if cache is not None:
            theta, s, lut = fetch_style_batch(cache, [key], device, cfg)
        else:
            theta = torch.zeros(1, cfg.theta_dim, device=device)
            s = torch.zeros(1, cfg.style_dim, device=device)
            lut = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            guide = isp(x_lin, theta, lut).float()
            ts = torch.full((1,), 0.1, device=device)
            residual, grid = renderer(guide, x_lin, s, meta, ts)
            pred = composite(guide, residual.float(), grid.float(), guide).clamp(0, 1)

        g_de, p_de = float(delta_e00(guide, tgt_t)), float(delta_e00(pred, tgt_t))
        g_lp, p_lp = float(lp(guide, tgt_t)), float(lp(pred, tgt_t))
        drift = float(delta_e00(pred, guide))
        res = float(residual.float().abs().mean())
        better = p_lp < g_lp and p_de <= g_de * 1.05
        wins += int(better)
        agg["g_psnr"] += psnr(guide, tgt_t); agg["p_psnr"] += psnr(pred, tgt_t)
        agg["g_lpips"] += g_lp; agg["p_lpips"] += p_lp
        agg["g_de"] += g_de; agg["p_de"] += p_de
        agg["drift"] += drift; agg["res"] += res
        agg["base_de"] += float(delta_e00(base_t, tgt_t))
        name = os.path.basename(it["jpeg_path"])[:20]
        print(f"{name:<22} {g_de:9.2f} {p_de:9.2f} {g_lp:12.4f} {p_lp:11.4f} "
              f"{drift:7.2f} {res:7.4f} {'OK' if better else ''}")

        def save(x, tag):
            Image.fromarray((x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255
                             ).astype(np.uint8)).save(f"eval_renderer/{i:02d}_{tag}.jpg", quality=95)
        save(base_t, "0input"); save(guide, "1guide"); save(pred, "2pred"); save(tgt_t, "3target")
        # residual amplified 10x around mid grey so it is actually visible
        save((residual.float() * 10 + 0.5), "4residual_x10")

    n = max(len(items), 1)
    print("-" * 82)
    print(f"{'MEAN':<22} {agg['g_de']/n:9.2f} {agg['p_de']/n:9.2f} "
          f"{agg['g_lpips']/n:12.4f} {agg['p_lpips']/n:11.4f} "
          f"{agg['drift']/n:7.2f} {agg['res']/n:7.4f}")
    print(f"\nPSNR vs target:  guide {agg['g_psnr']/n:.2f} dB -> renderer {agg['p_psnr']/n:.2f} dB")
    print(f"renderer beat guide on {wins}/{n} images")
    if cache is not None:
        cache.report()

    # Decisive check: is the guide actually styled, or is it just the base?
    base_de = agg["base_de"] / n
    guide_de = agg["g_de"] / n
    print(f"\nSTYLING CHECK  dE(base -> target) {base_de:.2f}   dE(guide -> target) {guide_de:.2f}")
    if guide_de > base_de * 0.8:
        print("  warning: guide is not closer to the target than the untouched input; "
              "rebuild the style cache")
    else:
        print(f"  guide closes {100*(1-guide_de/max(base_de,1e-6)):.0f}% of the base->target gap")
    print()
    d_lp = agg["g_lpips"] / n - agg["p_lpips"] / n
    if wins == 0 or d_lp <= 0:
        print("verdict: renderer does not improve on the guide")
    else:
        print(f"verdict: renderer improves LPIPS by {d_lp:.4f}")
    if agg["drift"] / n > 2.0:
        print(f"warning: colour drift {agg['drift']/n:.2f} dE from the guide")
    if agg["res"] / n > 0.05:
        print(f"warning: residual energy {agg['res']/n:.4f} is high")

    # fidelity/realism knob: does the pseudo-timestep do anything?
    print("\ntimestep sweep (fidelity knob), image 0:")
    it = items[0]
    base_t = torch.from_numpy(np.array(Image.open(it.get("base_path", it["jpeg_path"])
                                                  ).convert("RGB"), dtype=np.float32) / 255.0
                              ).permute(2, 0, 1)[None].to(device)
    x_lin = decode_transfer(base_t, cfg.transfer)
    theta, s, lut = (fetch_style_batch(cache, [it.get("style_key", "")], device, cfg)
                     if cache is not None else
                     (torch.zeros(1, cfg.theta_dim, device=device),
                      torch.zeros(1, cfg.style_dim, device=device), None))
    meta = torch.zeros(1, 6, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        guide = isp(x_lin, theta, lut).float()
        for ts_v in [0.0, 0.1, 0.2, 0.3]:
            r, g = renderer(guide, x_lin, s, meta,
                            torch.full((1,), ts_v, device=device))
            p = composite(guide, r.float(), g.float(), guide).clamp(0, 1)
            print(f"  t={ts_v:.2f}  dE from guide {float(delta_e00(p, guide)):6.3f}  "
                  f"|R| {float(r.float().abs().mean()):.4f}")
    print("  (all rows identical => the knob is inert)")
    print(f"\nimages in eval_renderer/  (0input 1guide 2pred 3target 4residual_x10)")


def train_renderer(cfg: Config, manifest, stage: int):
    local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    renderer = OneStepRenderer(cfg).to(device)
    isp = ParametricISP(cfg).to(device)
    with contextlib.redirect_stdout(io.StringIO()):
        lpips_loss = LPIPSLoss().to(device)
    if dist.is_initialized():
        renderer = DDP(renderer, device_ids=[local_rank], find_unused_parameters=True)

    ds = RawJPEGPairDataset(manifest, crop=cfg.renderer_res, require_raw=False,
                            split=cfg.split, split_frac=cfg.split_frac)
    sampler = DistributedSampler(ds) if dist.is_initialized() else None
    dl = DataLoader(ds, batch_size=cfg.renderer_batch, sampler=sampler, shuffle=sampler is None,
                     num_workers=8, collate_fn=collate_fn, drop_last=True)

    trainable = [p for p in renderer.parameters() if p.requires_grad]
    opt = make_optimizer(trainable, cfg.lr_renderer, cfg)
    accum = cfg.renderer_grad_accum
    style_cache = StyleCache(cfg.style_cache_dir) if cfg.use_style_cache else None
    clip_tracker = GradNormTracker(factor=cfg.grad_clip_factor)
    step = 0
    ema = None
    run_loss, n_micro = 0.0, 0
    acc = {"l1": 0.0, "lpips": 0.0, "col": 0.0, "res": 0.0, "dnull": 0.0, "dpred": 0.0}
    if stage == 1 and is_main():
        print("[renderer] stage 1: synthetic pairs, color-consistency weighted high "
              "(learn the conditioning interface and to NOT change color)")
    w_col = 0.4 if stage == 1 else 0.2
    w_tex = 0.25 if stage == 1 else 0.5
    opt.zero_grad()
    for epoch in range(cfg.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        for micro, batch in enumerate(dl):
            target = batch["target"].to(device)
            x_lin = decode_transfer(batch["base"].to(device) if "base" in batch else target, cfg.transfer)
            meta = batch["meta"].to(device)
            b = target.shape[0]
            if style_cache is not None:
                theta, s, lut = fetch_style_batch(style_cache, batch["style_key"], device, cfg)
            else:
                theta = torch.zeros(b, cfg.theta_dim, device=device)
                s = torch.zeros(b, cfg.style_dim, device=device)
                lut = None
            with torch.no_grad():
                guide = isp(x_lin, theta, lut)
            # per-sample, not per-batch: one draw for the whole micro-batch correlates the
            # dropout with everything else in it and wastes most of the signal
            keep = (torch.rand(b, 1, device=device) >= cfg.style_dropout).to(s.dtype)
            s = s * keep
            # vary the pseudo-timestep: training at a single value means the inference-time
            # fidelity/realism knob has never been exercised and does nothing
            timestep = torch.rand(b, device=device) * cfg.timestep_max

            with torch.autocast("cuda", dtype=torch.bfloat16):
                residual, grid = renderer(guide, x_lin, s, meta, timestep)
            # pred = guide + residual while composite() blends in the bilateral grid, so the
            # grid received zero gradient (dead module) and inference ran a formula the model
            # was never trained under.
            pred = composite(guide.float(), residual.float(), grid.float(), guide.float()).clamp(0, 1)
            target_f, guide_f = target.float(), guide.float()
            l1 = F.l1_loss(pred, target_f)
            tex = lpips_loss(pred, target_f)
            col = color_consistency_loss(pred, guide_f)
            res = residual.float().pow(2).mean()
            loss = l1 + w_tex * tex + w_col * col + 0.01 * res
            if not torch.isfinite(loss):
                if is_main():
                    print(f"[renderer-stage{stage}] non-finite loss, skipping micro-batch")
                opt.zero_grad(set_to_none=True)
                continue
            with torch.no_grad():
                # The decisive pair: the renderer is only useful if pred beats guide against
                # the SAME target. Training loss alone cannot show this, because the colour
                # term anchors pred to the guide -- a renderer that does nothing scores well
                # on that term while adding no value.
                acc["dnull"] += float(delta_e00(guide_f, target_f))
                acc["dpred"] += float(delta_e00(pred, target_f))
            acc["l1"] += float(l1); acc["lpips"] += float(tex)
            acc["col"] += float(col); acc["res"] += float(res)
            (loss / accum).backward()
            run_loss += float(loss); n_micro += 1
            if (micro + 1) % accum == 0:
                gnorm, thr = adaptive_clip(trainable, clip_tracker, fallback=None)
                if math.isfinite(gnorm):
                    opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                avg = run_loss / max(n_micro, 1)
                ema = avg if ema is None else 0.9 * ema + 0.1 * avg
                if step % cfg.log_every == 0 and is_main():
                    k = max(n_micro, 1)
                    print(f"[renderer-stage{stage}] epoch {epoch} step {step} "
                          f"loss {avg:.4f} ema {ema:.4f} "
                          f"| L1 {acc['l1']/k:.4f} lpips {acc['lpips']/k:.4f} "
                          f"col {acc['col']/k:.3f} |R|^2 {acc['res']/k:.5f} "
                          f"| dE guide {acc['dnull']/k:.2f} -> pred {acc['dpred']/k:.2f} "
                          f"({acc['dnull']/k - acc['dpred']/k:+.2f}) "
                          f"| |g| {gnorm:.0f} clip {100*clip_tracker.n_clipped/max(step,1):.0f}%")
                run_loss, n_micro = 0.0, 0
                acc = {kk: 0.0 for kk in acc}
                if step % cfg.save_every == 0 and is_main():
                    save_renderer(renderer, cfg, f"stage{stage}_{step}")
                if cfg.max_steps and step >= cfg.max_steps:
                    if is_main():
                        save_renderer(renderer, cfg, "final")
                    return
    if is_main():
        save_renderer(renderer, cfg, "final")
        if style_cache is not None:
            style_cache.report()


# ============================== main ==============================

def main():
    quiet_third_party()
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                     choices=["lut_vqvae", "frontend", "eval_frontend", "director_sft",
                              "director_rl", "eval_prompts", "cache_style", "renderer",
                              "eval_renderer", "probe_guide"])
    ap.add_argument("--ckpt", default=None, help="eval_frontend: checkpoint to load")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None, help="0 disables the cap (full-corpus/server run)")
    ap.add_argument("--use_frontend", action="store_true",
                     help="evaluate the full pipeline from RAW through the trained "
                          "front-end, instead of the pre-rendered neutral image")
    ap.add_argument("--split", choices=["train", "test", "all"], default=None,
                     help="which scene split to use; training defaults to train, "
                          "evaluation phases to test")
    ap.add_argument("--lut_only", action="store_true",
                     help="train only the LUT head; freezes LoRA/pool_norm/theta_head so "
                          "a working theta director cannot regress")
    ap.add_argument("--out_dir", default=None,
                     help="checkpoint directory. Use a SEPARATE one for experimental "
                          "runs so a working checkpoint cannot be overwritten.")
    ap.add_argument("--resume", nargs="?", const=True, default=None,
                     help="warm-start director_sft from a checkpoint "
                          "(bare --resume uses checkpoints/director_final.pt)")
    ap.add_argument("--tame", type=float, default=None,
                     help="clamp raw theta at inference, e.g. 1.0. Makes an existing "
                          "checkpoint produce photographic output without retraining.")
    ap.add_argument("--safe", action="store_true",
                     help="conservative preset: stronger theta_l2, head LR close to LoRA LR, "
                          "shorter schedule. Trades some expressiveness for a run that "
                          "cannot reach the saturation trap.")
    ap.add_argument("--theta_l2", type=float, default=None)
    ap.add_argument("--head_lr_mult", type=float, default=None)
    ap.add_argument("--pool", choices=["last", "mean"], default=None,
                     help="director pooling (default: last token)")
    ap.add_argument("--lr_director", type=float, default=None,
                     help="director LR (default 1e-4); lower only if clip%% stays ~100 after warmup")
    ap.add_argument("--grad_clip", type=float, default=None,
                     help="max grad norm for director/renderer (default 10; log shows clip%%)")
    ap.add_argument("--use_lut", action="store_true", help="enable the LUT VQ-VAE path (off by default)")
    ap.add_argument("--style_cache_dir", type=str, default=None)
    ap.add_argument("--gpu48", action="store_true",
                     help="single card with ~48GB to spare (e.g. RTX 4090 48GB): plain bf16 instead of "
                          "4-bit/8-bit (less dequant overhead, you have the memory), moderate rank/res/batch")
    ap.add_argument("--detect-anomaly", action="store_true",
                     help="slow; makes autograd raise at the exact op that produces a backward NaN")
    ap.add_argument("--server", action="store_true",
                     help="disable 4-bit/8-bit shortcuts and use full rank/res/batch for a multi-GPU run")
    args = ap.parse_args()

    cfg = Config()
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("[debug] autograd anomaly detection ON (slow)")
    if args.gpu48:
        cfg.use_4bit, cfg.use_8bit_adam = False, False
        cfg.lora_rank = cfg.lora_alpha = 32
        cfg.renderer_res = 512
        cfg.renderer_batch, cfg.renderer_grad_accum = 2, 8
        cfg.director_batch, cfg.director_grad_accum = 2, 8
        cfg.frontend_batch = 32
    if args.server:
        cfg.use_4bit, cfg.use_8bit_adam, cfg.max_steps = False, False, None
        cfg.lora_rank = cfg.lora_alpha = 64
        cfg.frontend_batch, cfg.director_batch, cfg.renderer_batch = 64, 8, 4
        cfg.director_grad_accum, cfg.renderer_grad_accum = 4, 8
        cfg.rl_group_size, cfg.rl_prompts_per_step = 8, 2
        cfg.renderer_res = 768
    if args.use_lut:
        cfg.use_lut = True
    if args.grad_clip is not None:
        cfg.grad_clip = args.grad_clip
    if args.lr_director is not None:
        cfg.lr_director = args.lr_director
    if args.pool is not None:
        cfg.pool = args.pool
    if args.safe:
        cfg.theta_l2, cfg.head_lr_mult = 2e-2, 1.5
        cfg.lr_director = min(cfg.lr_director, 5e-5)
        cfg.max_steps = cfg.max_steps or 3000
    if args.use_frontend:
        cfg.use_frontend = True
    if args.split:
        cfg.split = args.split
    elif args.phase in ("probe_guide", "eval_prompts", "eval_renderer",
                        "eval_frontend"):
        cfg.split = "test"      # оценка по умолчанию на отложенных сценах
    if args.lut_only:
        cfg.lut_only = True
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.resume is not None:
        cfg.resume = args.resume
    if args.tame is not None:
        cfg.tame_limit = args.tame
    if args.theta_l2 is not None:
        cfg.theta_l2 = args.theta_l2
    if args.head_lr_mult is not None:
        cfg.head_lr_mult = args.head_lr_mult
    if args.style_cache_dir:
        cfg.style_cache_dir = args.style_cache_dir
    if args.epochs:
        cfg.epochs = args.epochs
    if args.max_steps is not None:
        cfg.max_steps = None if args.max_steps == 0 else args.max_steps

    if args.phase == "lut_vqvae":
        train_lut_vqvae(cfg, args.manifest)
    elif args.phase == "frontend":
        train_frontend(cfg, args.manifest)
    elif args.phase == "eval_frontend":
        evaluate_frontend(cfg, args.manifest,
                          args.ckpt or os.path.join(cfg.out_dir, "frontend_final.pt"))
    elif args.phase == "director_sft":
        train_director_sft(cfg, args.manifest)
    elif args.phase == "director_rl":
        train_director_rl(cfg, args.manifest)
    elif args.phase == "eval_prompts":
        eval_prompts(cfg, args.manifest, ckpt=args.ckpt)
    elif args.phase == "cache_style":
        precompute_style_cache(cfg, args.manifest, cfg.style_cache_dir)
    elif args.phase == "renderer":
        train_renderer(cfg, args.manifest, args.stage)
    elif args.phase == "eval_renderer":
        eval_renderer(cfg, args.manifest, ckpt=args.ckpt)
    elif args.phase == "probe_guide":
        probe_guide(cfg, args.manifest, ckpt=args.ckpt)


if __name__ == "__main__":
    main()
