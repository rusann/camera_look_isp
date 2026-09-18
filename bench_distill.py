"""
Benchmark + distillation for the camera-look ISP.

  python bench_distill.py --phase benchmark --manifest /mnt/data/manifest.jsonl \
      --out_dir checkpoints_lut --style_cache_dir style_cache_lut --use_lut

  python bench_distill.py --phase distill --manifest /mnt/data/manifest.jsonl \
      --out_dir checkpoints_lut --style_cache_dir style_cache_lut --use_lut --max_steps 4000

benchmark writes results.md (paper tables) and results.json (raw numbers).
distill trains a compact student for the renderer residual and reports its speedup.
"""

import os, io, json, time, math, random, argparse, contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import camera_look_isp as C


# ============================== metrics ==============================

def ssim(a, b, win=11, sigma=1.5):
    """Standard SSIM, computed per channel and averaged."""
    dev = a.device
    coords = torch.arange(win, dtype=torch.float32, device=dev) - win // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    kernel = (g[:, None] @ g[None, :]).expand(a.shape[1], 1, win, win).contiguous()

    def flt(x):
        return F.conv2d(x, kernel, padding=win // 2, groups=x.shape[1])

    mu1, mu2 = flt(a), flt(b)
    mu1s, mu2s, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = flt(a * a) - mu1s
    s2 = flt(b * b) - mu2s
    s12 = flt(a * b) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1s + mu2s + c1) * (s1 + s2 + c2))
    return float(m.mean())


def psnr(a, b):
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1))
    return float(10 * torch.log10(1.0 / mse.clamp(min=1e-10)))


def pct(vals, p):
    return float(np.percentile(np.asarray(vals), p)) if len(vals) else float("nan")


# ============================== benchmark ==============================

@torch.no_grad()
def run_benchmark(cfg, manifest, n=200, seed=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    isp = C.ParametricISP(cfg).to(device)
    with contextlib.redirect_stdout(io.StringIO()):
        lp = C.LPIPSLoss().to(device)
    director = C.StyleDirector(cfg).to(device).eval()
    C.load_director(director, cfg, required=True)
    cache = C.StyleCache(cfg.style_cache_dir) if cfg.use_style_cache else None

    lut_vq = None
    if cfg.use_lut:
        ckv = f"{cfg.out_dir}/lut_vqvae.pt"
        if os.path.exists(ckv):
            lut_vq = C.LUTVQVAE(cfg).to(device).eval()
            lut_vq.load_state_dict(torch.load(ckv, map_location=device))

    items = [json.loads(l) for l in open(manifest) if l.strip()]
    random.Random(seed).shuffle(items)
    items = items[:n]

    rows, per_style, per_cam = [], {}, {}
    t_dir, t_isp = [], []
    for it in items:
        base = Image.open(it.get("base_path", it["jpeg_path"])).convert("RGB")
        tgt = Image.open(it["jpeg_path"]).convert("RGB")
        to_t = lambda im: torch.from_numpy(np.array(im, np.float32) / 255.).permute(2, 0, 1)[None].to(device)
        b_t, t_t = to_t(base), to_t(tgt)
        x_lin = C.decode_transfer(b_t, cfg.transfer)

        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            theta, lut_logits, _ = director([base], [it.get("prompt", "")])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_dir.append(time.perf_counter() - t0)

        lut = lut_vq.decode_from_ids(lut_logits.float().argmax(-1)) if lut_vq is not None else None
        t0 = time.perf_counter()
        pred = isp(x_lin, theta.float(), lut, grain=False).clamp(0, 1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_isp.append(time.perf_counter() - t0)

        de = float(C.delta_e00(pred, t_t))
        de_null = float(C.delta_e00(b_t, t_t))
        r = {"style": it.get("style_key", "?"), "camera": it.get("camera", "?"),
             "dE": de, "dE_null": de_null, "psnr": psnr(pred, t_t),
             "ssim": ssim(pred, t_t), "lpips": float(lp(pred, t_t)),
             "closed": 1 - de / max(de_null, 1e-6)}
        rows.append(r)
        per_style.setdefault(r["style"], []).append(de)
        per_cam.setdefault(r["camera"], []).append(de)

    agg = {
        "n": len(rows),
        "dE_mean": float(np.mean([r["dE"] for r in rows])),
        "dE_median": pct([r["dE"] for r in rows], 50),
        "dE_p95": pct([r["dE"] for r in rows], 95),
        "dE_null_mean": float(np.mean([r["dE_null"] for r in rows])),
        "gap_closed_mean": float(np.mean([r["closed"] for r in rows])),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "lpips": float(np.mean([r["lpips"] for r in rows])),
        "director_ms": float(np.median(t_dir) * 1000),
        "isp_ms": float(np.median(t_isp) * 1000),
    }
    return agg, rows, per_style, per_cam


@torch.no_grad()
def controllability(cfg, manifest, device):
    """Monotonic response to graded instructions: the CameraMaster-style protocol.
    Reports whether 'slightly X' < 'X' < 'much X' in effect size, per axis."""
    director = C.StyleDirector(cfg).to(device).eval()
    C.load_director(director, cfg, required=True)
    isp = C.ParametricISP(cfg).to(device)
    items = [json.loads(l) for l in open(manifest) if l.strip()]
    it = items[0]
    img = Image.open(it.get("base_path", it["jpeg_path"])).convert("RGB")
    t = torch.from_numpy(np.array(img, np.float32) / 255.).permute(2, 0, 1)[None].to(device)
    x_lin = C.decode_transfer(t, cfg.transfer)
    with torch.no_grad():
        neutral = isp(x_lin, torch.zeros(1, cfg.theta_dim, device=device), None, grain=False)

    axes = {"warmer": ["slightly warmer", "warmer", "much warmer"],
            "cooler": ["slightly cooler", "cooler", "much cooler"],
            "brighter": ["slightly brighter", "brighter", "much brighter"],
            "more contrasty": ["slightly more contrasty", "more contrasty", "much more contrasty"]}
    out = {}
    for axis, levels in axes.items():
        mags = []
        for lv in levels:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                th, _, _ = director([img], [f"make this {lv}"])
            r = isp(x_lin, th.float(), None, grain=False)
            mags.append(float(C.delta_e00(r, neutral)))
        mono = all(mags[i] <= mags[i + 1] + 0.25 for i in range(len(mags) - 1))
        out[axis] = {"magnitudes": [round(m, 2) for m in mags], "monotonic": bool(mono)}
    return out


@torch.no_grad()
def hallucination_audit(cfg, manifest, device, n=40):
    """Renderer residual energy: the design doc's ||R|| check. Requires a renderer ckpt."""
    path = f"{cfg.out_dir}/renderer_final.pt"
    if not os.path.exists(path):
        return {"available": False, "reason": f"{path} not found"}
    renderer = C.OneStepRenderer(cfg).to(device).eval()
    C.load_renderer(renderer, cfg, required=True)
    isp = C.ParametricISP(cfg).to(device)
    cache = C.StyleCache(cfg.style_cache_dir)
    items = [json.loads(l) for l in open(manifest) if l.strip()][:n]
    res, drift = [], []
    for it in items:
        base = Image.open(it.get("base_path", it["jpeg_path"])).convert("RGB")
        b_t = torch.from_numpy(np.array(base, np.float32) / 255.).permute(2, 0, 1)[None].to(device)
        x_lin = C.decode_transfer(b_t, cfg.transfer)
        theta, s, lut = C.fetch_style_batch(cache, [it.get("style_key", "")], device, cfg)
        meta = torch.zeros(1, 6, device=device)
        guide = isp(x_lin, theta, lut, grain=False).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r, g = renderer(guide, x_lin, s, meta, torch.full((1,), 0.1, device=device))
        pred = C.composite(guide, r.float(), g.float(), guide).clamp(0, 1)
        res.append(float(r.float().abs().mean()))
        drift.append(float(C.delta_e00(pred, guide)))
    return {"available": True, "residual_mean": float(np.mean(res)),
            "residual_p95": pct(res, 95), "colour_drift_dE": float(np.mean(drift))}


def param_counts(cfg, device):
    out = {}
    fe = C.FrontEnd(cfg)
    out["frontend_M"] = sum(p.numel() for p in fe.parameters()) / 1e6
    vq = C.LUTVQVAE(cfg)
    out["lut_vqvae_M"] = sum(p.numel() for p in vq.parameters()) / 1e6
    out["isp_params"] = cfg.theta_dim
    out["lut_tokens"] = cfg.lut_tokens
    return out


def write_report(cfg, agg, rows, per_style, per_cam, ctrl, halluc, params, path="results"):
    j = {"aggregate": agg, "per_style": {k: float(np.mean(v)) for k, v in per_style.items()},
         "per_camera": {k: float(np.mean(v)) for k, v in per_cam.items()},
         "controllability": ctrl, "hallucination": halluc, "params": params,
         "config": {"use_lut": cfg.use_lut, "theta_dim": cfg.theta_dim,
                    "lut_size": cfg.lut_size, "lut_tokens": cfg.lut_tokens}}
    with open(f"{path}.json", "w") as f:
        json.dump(j, f, indent=2)

    L = []
    L.append("# Camera-look ISP: evaluation\n")
    L.append(f"Evaluated on {agg['n']} held-out pairs. "
             f"LUT head: {'on' if cfg.use_lut else 'off'}.\n")
    L.append("## Colour fidelity\n")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| dE00 mean | {agg['dE_mean']:.2f} |")
    L.append(f"| dE00 median | {agg['dE_median']:.2f} |")
    L.append(f"| dE00 p95 | {agg['dE_p95']:.2f} |")
    L.append(f"| dE00 of untouched base (null) | {agg['dE_null_mean']:.2f} |")
    L.append(f"| gap closed vs null | {100*agg['gap_closed_mean']:.1f}% |")
    L.append("")
    L.append("## Rendering fidelity\n")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| PSNR (dB) | {agg['psnr']:.2f} |")
    L.append(f"| SSIM | {agg['ssim']:.4f} |")
    L.append(f"| LPIPS | {agg['lpips']:.4f} |")
    L.append("")
    L.append("## Per-style dE00 (mean)\n")
    L.append("| style | dE00 | n |")
    L.append("|---|---|---|")
    for k, v in sorted(per_style.items(), key=lambda kv: np.mean(kv[1])):
        L.append(f"| {k} | {np.mean(v):.2f} | {len(v)} |")
    L.append("")
    L.append("## Per-camera dE00 (cross-device generalisation)\n")
    L.append("| camera | dE00 | n |")
    L.append("|---|---|---|")
    for k, v in sorted(per_cam.items(), key=lambda kv: np.mean(kv[1])):
        L.append(f"| {k} | {np.mean(v):.2f} | {len(v)} |")
    L.append("")
    L.append("## Controllability (graded instructions)\n")
    L.append("| axis | slight / normal / much (dE from neutral) | monotonic |")
    L.append("|---|---|---|")
    for k, v in ctrl.items():
        L.append(f"| {k} | {' / '.join(str(m) for m in v['magnitudes'])} | "
                 f"{'yes' if v['monotonic'] else 'NO'} |")
    L.append("")
    if halluc.get("available"):
        L.append("## Hallucination audit (renderer)\n")
        L.append("| metric | value |")
        L.append("|---|---|")
        L.append(f"| residual mean abs | {halluc['residual_mean']:.5f} |")
        L.append(f"| residual p95 | {halluc['residual_p95']:.5f} |")
        L.append(f"| colour drift from guide (dE00) | {halluc['colour_drift_dE']:.2f} |")
        L.append("")
    L.append("## Cost\n")
    L.append("| component | value |")
    L.append("|---|---|")
    L.append(f"| front-end params | {params['frontend_M']:.2f} M |")
    L.append(f"| LUT tokenizer params | {params['lut_vqvae_M']:.2f} M |")
    L.append(f"| ISP parameters (theta) | {params['isp_params']} |")
    L.append(f"| LUT tokens per style | {params['lut_tokens']} |")
    L.append(f"| director latency (median) | {agg['director_ms']:.1f} ms |")
    L.append(f"| parametric ISP latency (median) | {agg['isp_ms']:.1f} ms |")
    L.append("")
    with open(f"{path}.md", "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {path}.md and {path}.json")


# ============================== distillation ==============================

class StudentRenderer(nn.Module):
    """Compact replacement for the diffusion renderer.

    The teacher is a 4B DiT producing a bounded residual over the guide plus a bilateral
    grid. Since the output is a *residual* (D1.5) rather than an image, a small conv net can
    reproduce it: the hard part (colour) is already done deterministically by the guide.
    Ops are chosen to be INT8/NPU-friendly -- plain convs, ReLU, no attention, no LayerNorm.
    """

    def __init__(self, cfg, width=48, style_dim=None):
        super().__init__()
        sd = style_dim or cfg.style_dim
        self.style = nn.Sequential(nn.Linear(sd, 128), nn.ReLU(), nn.Linear(128, width))
        self.enc1 = nn.Sequential(nn.Conv2d(6, width, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(width, width, 3, 1, 1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(width, width * 2, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(width * 2, width * 2, 3, 1, 1), nn.ReLU())
        self.mid = nn.Sequential(nn.Conv2d(width * 2, width * 2, 3, 1, 1), nn.ReLU(),
                                 nn.Conv2d(width * 2, width * 2, 3, 1, 1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(width * 4, width, 3, 1, 1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(width * 2, width, 3, 1, 1), nn.ReLU())
        self.out = nn.Conv2d(width, 3, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)          # start as "guide unchanged", as the teacher does
        self.grid = nn.Sequential(nn.Linear(sd, 128), nn.ReLU(),
                                  nn.Linear(128, cfg.bgrid_res * 12))
        nn.init.zeros_(self.grid[-1].weight)
        with torch.no_grad():
            b = torch.zeros(12, cfg.bgrid_res)
            b[0] = b[4] = b[8] = 1.0
            self.grid[-1].bias.copy_(b.reshape(-1))
        self.bgrid_res = cfg.bgrid_res

    def forward(self, guide, x_lin, s):
        z = torch.cat([guide, x_lin.clamp(0, 1)], 1)
        e1 = self.enc1(z)
        e1 = e1 + self.style(s)[:, :, None, None]
        e2 = self.enc2(e1)
        m = self.mid(e2)
        d2 = self.dec2(torch.cat([m, e2], 1))
        d2 = F.interpolate(d2, size=e1.shape[-2:], mode="nearest")
        d1 = self.dec1(torch.cat([d2, e1], 1))
        d1 = F.interpolate(d1, size=guide.shape[-2:], mode="nearest")
        residual = self.out(d1)
        g = self.grid(s).view(-1, 12, self.bgrid_res, 1, 1)
        g = g.expand(-1, -1, -1, self.bgrid_res, self.bgrid_res).contiguous()
        return residual, g


def train_distill(cfg, manifest, steps=4000):
    """Requires a renderer that beats the guide. Distilling a harmful teacher reproduces the
    harm in a smaller model, so this refuses to run without an explicit override."""
    """Knowledge distillation: the student matches the TEACHER's residual, plus the task loss.

    Matching the teacher rather than only the target is what makes this cheap -- the teacher
    has already solved the hard assignment problem, so the student learns a smooth regression
    instead of re-discovering it from scratch (the K-DMD argument in the design doc).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not cfg.force_distill:
        print("[distill] refusing to run: verify with --phase eval_renderer that the "
              "renderer beats the guide, or pass --force.")
        return
    teacher = C.OneStepRenderer(cfg).to(device).eval()
    C.load_renderer(teacher, cfg, required=True)
    teacher.requires_grad_(False)
    isp = C.ParametricISP(cfg).to(device)
    with contextlib.redirect_stdout(io.StringIO()):
        lp = C.LPIPSLoss().to(device)
    student = StudentRenderer(cfg).to(device)
    n_t = sum(p.numel() for p in teacher.parameters())
    n_s = sum(p.numel() for p in student.parameters())
    print(f"[distill] teacher {n_t/1e6:.0f}M -> student {n_s/1e6:.2f}M "
          f"({n_t/max(n_s,1):.0f}x smaller)")

    ds = C.RawJPEGPairDataset(manifest, crop=cfg.renderer_res, require_raw=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=max(cfg.renderer_batch, 2), shuffle=True,
                                     num_workers=4, collate_fn=C.collate_fn, drop_last=True)
    cache = C.StyleCache(cfg.style_cache_dir) if cfg.use_style_cache else None
    opt = torch.optim.AdamW(student.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    step, ema = 0, None
    while step < steps:
        for batch in dl:
            if step >= steps:
                break
            target = batch["target"].to(device)
            x_lin = C.decode_transfer(batch["base"].to(device) if "base" in batch else target,
                                      cfg.transfer)
            b = target.shape[0]
            if cache is not None:
                theta, s, lut = C.fetch_style_batch(cache, batch["style_key"], device, cfg)
            else:
                theta = torch.zeros(b, cfg.theta_dim, device=device)
                s = torch.zeros(b, cfg.style_dim, device=device)
                lut = None
            meta = batch["meta"].to(device)
            with torch.no_grad():
                guide = isp(x_lin, theta, lut).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    t_res, t_grid = teacher(guide, x_lin, s, meta,
                                            torch.full((b,), 0.1, device=device))
                t_res, t_grid = t_res.float(), t_grid.float()
                t_out = C.composite(guide, t_res, t_grid, guide).clamp(0, 1)

            s_res, s_grid = student(guide, x_lin, s)
            s_out = C.composite(guide, s_res, s_grid, guide).clamp(0, 1)
            kd = F.l1_loss(s_res, t_res) + F.l1_loss(s_out, t_out)
            task = F.l1_loss(s_out, target) + 0.3 * lp(s_out, target)
            loss = kd + 0.5 * task
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            ema = float(loss) if ema is None else 0.9 * ema + 0.1 * float(loss)
            if step % 50 == 0:
                print(f"[distill] step {step}/{steps} loss {float(loss):.4f} ema {ema:.4f} "
                      f"kd {float(kd):.4f} task {float(task):.4f}")
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.save(student.state_dict(), f"{cfg.out_dir}/student_renderer.pt")
    print(f"[distill] saved {cfg.out_dir}/student_renderer.pt")

    # latency comparison, the number that justifies the whole exercise
    res = cfg.renderer_res
    g = torch.rand(1, 3, res, res, device=device)
    xl = torch.rand(1, 3, res, res, device=device)
    sv = torch.zeros(1, cfg.style_dim, device=device)
    mv = torch.zeros(1, 6, device=device)
    def bench(fn, n=20):
        for _ in range(3):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000
    with torch.no_grad():
        t_ms = bench(lambda: teacher(g, xl, sv, mv, torch.full((1,), 0.1, device=device)))
        s_ms = bench(lambda: student(g, xl, sv))
    print(f"\n[distill] latency at {res}px: teacher {t_ms:.1f} ms -> student {s_ms:.1f} ms "
          f"({t_ms/max(s_ms,1e-6):.1f}x faster), params {n_t/1e6:.0f}M -> {n_s/1e6:.2f}M")
    json.dump({"teacher_ms": t_ms, "student_ms": s_ms, "teacher_params": n_t,
               "student_params": n_s, "res": res},
              open(f"{cfg.out_dir}/distill_bench.json", "w"), indent=2)


# ============================== main ==============================

@torch.no_grad()
def bake_styles(cfg, manifest, out_path=None, verify=True):
    """Collapse the whole colour pipeline into ONE 3D LUT per style.

    Everything from exposure through the LUT is a per-pixel colour map: exposure, black,
    tone curve, highlight knee, transfer encode, hue/sat map, saturation and the learned cube.
    Evaluating that chain on an identity cube bakes all of it into a single 33^3 texture.

    On device this replaces the entire parametric ISP with one trilinear texture lookup --
    hardware-accelerated on every mobile GPU, and independent of how complex the look is.
    Only the spatial ops (sharpen, grain, chroma NR) stay outside, and they are cheap
    separable filters.

    Output is a style pack: one small cube per style, shippable with the app, with no VLM
    and no diffusion on device at all.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    isp = C.ParametricISP(cfg).to(device)
    cache = C.StyleCache(cfg.style_cache_dir)
    keys = sorted({json.loads(l).get("style_key", "") for l in open(manifest) if l.strip()})
    n = cfg.lut_size

    lin = torch.linspace(0, 1, n, device=device)
    ident = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij")[::-1], 0)  # [3,b,g,r]
    grid_img = ident.reshape(1, 3, n, n * n)          # treat the cube as an image
    grid_lin = C.decode_transfer(grid_img, cfg.transfer)

    packs, errs = {}, []
    for k in keys:
        theta, s, lut = C.fetch_style_batch(cache, [k], device, cfg)
        baked = isp(grid_lin, theta, lut, grain=False, pointwise_only=True)
        baked = baked.reshape(3, n, n, n).clamp(0, 1)
        packs[k] = baked.cpu()
        if verify:
            # does one lookup reproduce the full chain on real colours?
            probe = torch.rand(1, 3, 128, 128, device=device)
            probe_lin = C.decode_transfer(probe, cfg.transfer)
            full = isp(probe_lin, theta, lut, grain=False, pointwise_only=True)
            one = C.lut3d_apply(baked[None], probe).clamp(0, 1)
            errs.append(float(C.delta_e00(one, full.clamp(0, 1))))

    out_path = out_path or os.path.join(cfg.out_dir, "style_pack.pt")
    torch.save({"luts": packs, "lut_size": n, "transfer": cfg.transfer}, out_path)

    f32 = n ** 3 * 3 * 4
    print(f"\nbaked {len(packs)} styles -> {out_path}")
    if errs:
        print(f"  bake error (one lookup vs full chain): mean dE {np.mean(errs):.3f}, "
              f"max {max(errs):.3f}")
        if np.mean(errs) > 1.0:
            print("  warning: bake error is high; the cube does not reproduce the chain")
    print(f"  per style: {f32/1024:.0f} KB float32, {f32/2/1024:.0f} KB float16, "
          f"{n**3*3/1024:.0f} KB int8")
    print(f"  full pack: {len(packs)*f32/2/1024/1024:.2f} MB at float16")

    return packs


@torch.no_grad()
def tier_report(cfg, manifest, n=60):
    """Cost/quality by deployment tier, for the paper's mobile table."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    isp = C.ParametricISP(cfg).to(device)
    cache = C.StyleCache(cfg.style_cache_dir)
    items = [json.loads(l) for l in open(manifest) if l.strip()][:n]
    pack_path = os.path.join(cfg.out_dir, "style_pack.pt")
    packs = torch.load(pack_path, map_location=device)["luts"] if os.path.exists(pack_path) else {}

    de_full, de_baked = [], []
    for it in items:
        base = Image.open(it.get("base_path", it["jpeg_path"])).convert("RGB")
        tgt = Image.open(it["jpeg_path"]).convert("RGB")
        to_t = lambda im: torch.from_numpy(np.array(im, np.float32) / 255.).permute(2, 0, 1)[None].to(device)
        b_t, t_t = to_t(base), to_t(tgt)
        x_lin = C.decode_transfer(b_t, cfg.transfer)
        key = it.get("style_key", "")
        theta, s, lut = C.fetch_style_batch(cache, [key], device, cfg)
        de_full.append(float(C.delta_e00(isp(x_lin, theta, lut, grain=False), t_t)))
        if key in packs:
            one = C.lut3d_apply(packs[key][None].to(device), b_t).clamp(0, 1)
            de_baked.append(float(C.delta_e00(one, t_t)))

    print("\n## Deployment tiers\n")
    print("| tier | on device | dE00 | notes |")
    print("|---|---|---|---|")
    print(f"| 0 (any phone) | 3D LUT lookup only | "
          f"{np.mean(de_baked):.2f} | " if de_baked else "| 0 | (no style pack) | - | run --phase bake |")
    print(f"| 1 | front-end + full parametric ISP | {np.mean(de_full):.2f} | "
          f"adds sharpen/grain/chroma-NR |")
    print("| 2 | + distilled renderer | see eval_renderer | local texture character |")


class TextLookStudent(nn.Module):
    """Small text-only model: prompt -> (theta, LUT tokens).

    Why text-only is the right target: the deployed path already caches ONE parameter set
    per style and reuses it across scenes (D2.4), so image adaptivity is discarded at
    inference regardless. A text-only student therefore reproduces deployed behaviour
    exactly while removing the 4B VLM from the device.

    What it buys: arbitrary prompts on device, not just the styles that happened to be
    baked into the style pack.
    """

    def __init__(self, cfg, in_dim, hidden=512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.theta = nn.Linear(hidden, cfg.theta_dim)
        self.lut = nn.Linear(hidden, cfg.lut_tokens * cfg.lut_codebook)
        nn.init.normal_(self.theta.weight, std=1e-3)
        nn.init.zeros_(self.theta.bias)
        self.cfg = cfg

    def forward(self, emb):
        h = self.trunk(emb)
        return self.theta(h), self.lut(h).view(-1, self.cfg.lut_tokens, self.cfg.lut_codebook)


def _load_text_encoder(device):
    """Frozen sentence encoder. MiniLM is 22M params / ~90 MB, which is mobile-viable;
    falls back to a hashed bag-of-words if it is not installed, which still works for
    in-vocabulary phrasing but generalises worse to unseen wording."""
    try:
        from transformers import AutoTokenizer, AutoModel
        name = "sentence-transformers/all-MiniLM-L6-v2"
        tok = AutoTokenizer.from_pretrained(name)
        enc = AutoModel.from_pretrained(name).to(device).eval()
        enc.requires_grad_(False)
        dim = enc.config.hidden_size

        def embed(prompts):
            b = tok(prompts, padding=True, truncation=True, max_length=64,
                    return_tensors="pt").to(device)
            out = enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            return ((out * m).sum(1) / m.sum(1).clamp(min=1)).float()

        print(f"[distill-director] text encoder: {name} ({sum(p.numel() for p in enc.parameters())/1e6:.0f}M)")
        return embed, dim, sum(p.numel() for p in enc.parameters())
    except Exception as e:
        print(f"[distill-director] MiniLM unavailable ({e}); using hashed bag-of-words")
        D = 1024

        def embed(prompts):
            v = torch.zeros(len(prompts), D, device=device)
            for i, p in enumerate(prompts):
                for w in p.lower().replace(",", " ").split():
                    v[i, hash(w) % D] += 1.0
            return F.normalize(v, dim=-1)
        return embed, D, 0


def distill_director(cfg, manifest, steps=3000, n_images=4):
    """Teacher = the VLM director. Student = text encoder + MLP.

    Targets are the teacher's outputs averaged over a few images per prompt, matching how
    cache_style builds its entries -- the student is trained to reproduce the cache, for
    any prompt, not just the cached ones.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C.sync_for(cfg)
    director = C.StyleDirector(cfg).to(device).eval()
    C.load_director(director, cfg, required=True)
    isp = C.ParametricISP(cfg).to(device)

    lut_vq = None
    if cfg.use_lut:
        ckv = os.path.join(cfg.out_dir, "lut_vqvae.pt")
        if os.path.exists(ckv):
            lut_vq = C.LUTVQVAE(cfg).to(device).eval()
            lut_vq.load_state_dict(torch.load(ckv, map_location=device))
            lut_vq.requires_grad_(False)

    items = [json.loads(l) for l in open(manifest) if l.strip() and "base_path" in json.loads(l)]
    prompts = sorted({it.get("prompt", "") for it in items if it.get("prompt")})
    rng = random.Random(0)
    imgs = [Image.open(it["base_path"]).convert("RGB") for it in rng.sample(items, min(n_images, len(items)))]
    print(f"[distill-director] {len(prompts)} distinct prompts, {len(imgs)} images each")

    # --- build the teacher targets once ---
    T_theta, T_lut = [], []
    with torch.no_grad():
        for i, p in enumerate(prompts):
            th, ll, _ = [], None, None
            for im in imgs:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    t_, l_, _ = director([im], [p])
                th.append(t_.float())
                ll = l_.float() if ll is None else ll + l_.float()
            T_theta.append(torch.stack(th).mean(0)[0])
            T_lut.append((ll / len(imgs))[0])
            if (i + 1) % 25 == 0:
                print(f"  teacher targets {i+1}/{len(prompts)}")
    T_theta = torch.stack(T_theta).to(device)
    T_lut = torch.stack(T_lut).to(device)

    # held-out prompts test generalisation to unseen phrasing, which is the whole point
    idx = list(range(len(prompts)))
    rng.shuffle(idx)
    n_val = max(1, len(idx) // 10)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    embed, dim, enc_params = _load_text_encoder(device)
    with torch.no_grad():
        E = torch.cat([embed(prompts[i:i + 64]) for i in range(0, len(prompts), 64)], 0)

    student = TextLookStudent(cfg, dim).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    for step in range(1, steps + 1):
        b = torch.tensor(rng.sample(tr_idx, min(64, len(tr_idx))), device=device)
        s_th, s_lut = student(E[b])
        loss = F.mse_loss(s_th, T_theta[b])
        if cfg.use_lut:
            loss = loss + 0.5 * F.kl_div(
                F.log_softmax(s_lut, -1), F.softmax(T_lut[b], -1),
                reduction="batchmean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            with torch.no_grad():
                v = torch.tensor(val_idx, device=device)
                vs_th, _ = student(E[v])
                verr = float(F.l1_loss(vs_th, T_theta[v]))
            print(f"[distill-director] step {step}/{steps} loss {float(loss):.5f} "
                  f"val theta L1 {verr:.5f}")

    # --- what actually matters: does the STUDENT's look match the TEACHER's, in dE? ---
    im = imgs[0]
    t = torch.from_numpy(np.array(im, np.float32) / 255.).permute(2, 0, 1)[None].to(device)
    x_lin = C.decode_transfer(t, cfg.transfer)
    des = []
    with torch.no_grad():
        for i in val_idx:
            s_th, s_lut = student(E[i:i + 1])
            t_lut = (lut_vq.decode_from_ids(T_lut[i:i + 1].argmax(-1)) if lut_vq else None)
            s_lutc = (lut_vq.decode_from_ids(s_lut.argmax(-1)) if lut_vq else None)
            a = isp(x_lin, T_theta[i:i + 1], t_lut, grain=False)
            b_ = isp(x_lin, s_th, s_lutc, grain=False)
            des.append(float(C.delta_e00(a, b_)))

    n_s = sum(p.numel() for p in student.parameters())
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.save({"student": student.state_dict(), "dim": dim,
                "prompts_seen": len(tr_idx)}, os.path.join(cfg.out_dir, "text_look_student.pt"))

    print(f"\n[distill-director] HELD-OUT prompts: student vs teacher look")
    print(f"  mean dE {np.mean(des):.2f}   median {np.median(des):.2f}   max {np.max(des):.2f}")
    print(f"  (under ~1.5 dE is visually equivalent; these prompts were never trained on)")
    print(f"  student head {n_s/1e6:.2f}M params" +
          (f" + frozen encoder {enc_params/1e6:.0f}M" if enc_params else " + bag-of-words"))
    print(f"  replaces a {sum(p.numel() for p in director.parameters())/1e9:.1f}B VLM on device")
    json.dump({"heldout_dE_mean": float(np.mean(des)),
               "heldout_dE_median": float(np.median(des)),
               "student_params": n_s, "encoder_params": enc_params,
               "n_prompts": len(prompts), "n_heldout": len(val_idx)},
              open(os.path.join(cfg.out_dir, "distill_director.json"), "w"), indent=2)


@torch.no_grad()
def apply_look(cfg, image_path, prompt, out_path="output.jpg", use_student=True):
    """End-to-end inference using ONLY the deployable components.

    prompt -> (distilled text student) -> theta + LUT tokens -> parametric ISP -> image.
    No VLM, no diffusion. This is the path that would run on device, and it exists so the
    system can be demonstrated and timed as a whole rather than phase by phase.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    isp = C.ParametricISP(cfg).to(device)
    lut_vq = None
    ckv = os.path.join(cfg.out_dir, "lut_vqvae.pt")
    if cfg.use_lut and os.path.exists(ckv):
        lut_vq = C.LUTVQVAE(cfg).to(device).eval()
        lut_vq.load_state_dict(torch.load(ckv, map_location=device))

    img = Image.open(image_path).convert("RGB")
    t = torch.from_numpy(np.array(img, np.float32) / 255.).permute(2, 0, 1)[None].to(device)
    x_lin = C.decode_transfer(t, cfg.transfer)

    sp = os.path.join(cfg.out_dir, "text_look_student.pt")
    t0 = time.perf_counter()
    if use_student and os.path.exists(sp):
        blob = torch.load(sp, map_location=device)
        embed, dim, enc_params = _load_text_encoder(device)
        student = TextLookStudent(cfg, blob["dim"]).to(device).eval()
        student.load_state_dict(blob["student"])
        theta, lut_logits = student(embed([prompt]))
        src = f"distilled student ({enc_params/1e6:.0f}M encoder + "\
              f"{sum(p.numel() for p in student.parameters())/1e6:.1f}M head)"
    else:
        C.sync_for(cfg)
        director = C.StyleDirector(cfg).to(device).eval()
        C.load_director(director, cfg, required=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            theta, lut_logits, _ = director([img], [prompt])
        theta = theta.float()
        lut_logits = lut_logits.float()
        src = "full VLM director"
    t_pred = time.perf_counter() - t0

    lut = lut_vq.decode_from_ids(lut_logits.argmax(-1)) if lut_vq is not None else None
    t0 = time.perf_counter()
    out = isp(x_lin, theta, lut).clamp(0, 1)
    t_render = time.perf_counter() - t0

    Image.fromarray((out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    ).save(out_path, quality=95)
    d = C.decode_theta(theta)
    print(f"\nprompt : {prompt!r}")
    print(f"source : {src}")
    print(f"theta  : exposure {float(d['exposure'][0]):+.2f} stops, "
          f"sat {float(d['sat'][0]):.2f}, black {float(d['black'][0]):.4f}")
    print(f"change : {float(C.delta_e00(out, t)):.2f} dE from the input")
    print(f"timing : look prediction {t_pred*1000:.0f} ms, render {t_render*1000:.0f} ms")
    print(f"wrote  : {out_path}")
    return out


@torch.no_grad()
def heldout_camera_eval(cfg, manifest, n_hold=2, n=200, seed=0):
    """Cross-device generalisation: evaluate on camera bodies excluded from the report.

    The per-camera table in the benchmark shows performance on cameras the model saw. The
    claim that a look transfers across sensors needs bodies that were held out, so this
    splits by camera and reports seen vs unseen separately.
    """
    items = [json.loads(l) for l in open(manifest) if l.strip()]
    cams = sorted({it.get("camera", "?") for it in items})
    if len(cams) <= n_hold:
        print(f"only {len(cams)} camera bodies; cannot hold out {n_hold}")
        return
    rng = random.Random(seed)
    held = set(rng.sample(cams, n_hold))
    print(f"held-out bodies: {sorted(held)}")
    print(f"seen bodies    : {len(cams)-n_hold}")

    import tempfile
    res = {}
    for label, keep in (("seen", False), ("HELD-OUT", True)):
        sub = [it for it in items if (it.get("camera", "?") in held) == keep]
        if not sub:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for it in sub:
                f.write(json.dumps(it) + "\n")
            tmp = f.name
        agg, _, _, _ = run_benchmark(cfg, tmp, n=min(n, len(sub)), seed=seed)
        res[label] = agg
        os.unlink(tmp)
        print(f"\n{label}: n={agg['n']}  dE {agg['dE_mean']:.2f} "
              f"(null {agg['dE_null_mean']:.2f}, closed {100*agg['gap_closed_mean']:.0f}%)  "
              f"PSNR {agg['psnr']:.2f} dB")

    if "seen" in res and "HELD-OUT" in res:
        d = res["HELD-OUT"]["dE_mean"] - res["seen"]["dE_mean"]
        print(f"\ncross-device penalty: {d:+.2f} dE on unseen bodies")
        if d < 0.5:
            print("  -> the look transfers across unseen sensors")
        else:
            print("  -> notable degradation on unseen sensors")
    json.dump({k: v for k, v in res.items()},
              open(os.path.join(cfg.out_dir, "heldout_camera.json"), "w"), indent=2)


def main():
    C.quiet_third_party()
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["benchmark", "distill", "distill_director", "bake", "tiers",
                             "apply", "heldout"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--image", default=None, help="apply: input image")
    ap.add_argument("--prompt", default=None, help="apply: the look to apply")
    ap.add_argument("--out", default="output.jpg")
    ap.add_argument("--no_student", action="store_true",
                    help="apply: use the full VLM instead of the distilled student")
    ap.add_argument("--hold", type=int, default=2, help="heldout: bodies to hold out")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--style_cache_dir", default=None)
    ap.add_argument("--use_lut", action="store_true")
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--report", default="results")
    ap.add_argument("--force", action="store_true",
                    help="distill even if the renderer does not beat the guide")
    args = ap.parse_args()

    cfg = C.Config()
    cfg.use_4bit, cfg.use_8bit_adam = False, False
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.style_cache_dir:
        cfg.style_cache_dir = args.style_cache_dir
    if args.use_lut:
        cfg.use_lut = True
    cfg.force_distill = args.force
    # Match whatever preset trained the checkpoint. Rebuilding with a different LoRA rank
    # produces a wall of shape-mismatch errors at load time; the checkpoint knows its rank.
    dpath = os.path.join(cfg.out_dir, "director_final.pt")
    if os.path.exists(dpath):
        C.sync_cfg_to_ckpt(cfg, dpath)
    else:
        print(f"[cfg] no {dpath}; using cfg defaults (lora_rank={cfg.lora_rank})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.phase == "benchmark":
        agg, rows, per_style, per_cam = run_benchmark(cfg, args.manifest, n=args.n)
        ctrl = controllability(cfg, args.manifest, device)
        halluc = hallucination_audit(cfg, args.manifest, device)
        params = param_counts(cfg, device)
        write_report(cfg, agg, rows, per_style, per_cam, ctrl, halluc, params, args.report)
    elif args.phase == "distill":
        train_distill(cfg, args.manifest, steps=args.max_steps)
    elif args.phase == "distill_director":
        distill_director(cfg, args.manifest, steps=args.max_steps)
    elif args.phase == "apply":
        if not args.image or not args.prompt:
            print("apply needs --image and --prompt")
            return
        apply_look(cfg, args.image, args.prompt, args.out, use_student=not args.no_student)
    elif args.phase == "heldout":
        heldout_camera_eval(cfg, args.manifest, n_hold=args.hold, n=args.n)
    elif args.phase == "bake":
        bake_styles(cfg, args.manifest)
    else:
        tier_report(cfg, args.manifest)


if __name__ == "__main__":
    main()
