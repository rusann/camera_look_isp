"""
Find the right ISP defaults WITHOUT training anything.

The front-end's job is to produce scene-linear RGB; the ISP then renders it. If the
ISP's default tone/normalization doesn't match how the targets were made, the net
spends all its capacity fighting that mismatch (and the loss plateaus high).

This substitutes a naive bilinear demosaic for the trained CanoNet, sweeps the ISP
settings, and reports which combination best matches the targets. Runs in ~1 minute.

  python probe_isp.py --manifest /mnt/data/manifest.jsonl --n 48

Put the winning settings into Config / ParametricISP, then retrain the front-end.
"""

import os, json, math, argparse
import numpy as np
import torch
import torch.nn.functional as F

from camera_look_isp import (Config, ParametricISP, delta_e00, MonotonicCurve,
                             HueSatMap, lut3d_apply, rgb_to_hsv, hsv_to_rgb)


def naive_demosaic(bayer):
    """Bilinear-ish demosaic: average the 2x2 Bayer quad -> half-res RGB, then upsample.
    Stands in for a well-trained CanoNet so we can isolate the ISP's tone behaviour."""
    b, _, h, w = bayer.shape
    r = bayer[:, :, 0::2, 0::2]
    g1 = bayer[:, :, 0::2, 1::2]
    g2 = bayer[:, :, 1::2, 0::2]
    bl = bayer[:, :, 1::2, 1::2]
    rgb = torch.cat([r, (g1 + g2) / 2, bl], dim=1)
    return F.interpolate(rgb, size=(h, w), mode="bilinear", align_corners=False)


def normalize(x, mode):
    b = x.shape[0]
    flat = x.reshape(b, -1)
    if flat.shape[1] > 100000:
        idx = torch.randperm(flat.shape[1], device=x.device)[:100000]
        flat = flat[:, idx]
    if mode == "max":
        s = flat.amax(1)
    elif mode == "p99":
        s = torch.quantile(flat, 0.99, dim=1)
    elif mode == "p95":
        s = torch.quantile(flat, 0.95, dim=1)
    elif mode == "p999":
        s = torch.quantile(flat, 0.999, dim=1)
    else:
        raise ValueError(mode)
    return x / s.clamp(min=1e-2).view(b, 1, 1, 1)


def encode(x, mode):
    x = x.clamp(1e-6, 1.0)
    if mode == "g22":
        return x ** (1 / 2.2)
    if mode == "bt709":  # what rawpy gamma=(2.222,4.5) applies
        return torch.where(x < 0.018, 4.5 * x,
                           1.099 * x.clamp(min=0.018) ** (1 / 2.222) - 0.099)
    if mode == "srgb":
        return torch.where(x <= 0.0031308, 12.92 * x,
                           1.055 * x.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)
    raise ValueError(mode)


def render(x_lin, norm_mode, gamma_mode, knee, lut):
    x = normalize(x_lin, norm_mode).clamp(0, 1)
    if knee > 0:
        x = x * (1 - knee) + (1 - torch.exp(-3 * x)) * knee
    x = encode(x, gamma_mode)
    if lut is not None:
        x = lut3d_apply(lut, x.clamp(0, 1)).clamp(0, 1)
    return x.clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=48, help="how many crops to test")
    ap.add_argument("--crop", type=int, default=256)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = [json.loads(l) for l in open(args.manifest) if l.strip()][:args.n]
    if not items:
        print("empty manifest")
        return

    from PIL import Image
    bayers, targets, ccms, luts = [], [], [], []
    for it in items:
        if "raw_path" not in it or not it["raw_path"].endswith(".npz"):
            continue
        d = np.load(it["raw_path"])
        bay = torch.from_numpy(d["bayer"].astype(np.float32))[None]
        ccm = torch.from_numpy(d["ccm"].astype(np.float32))
        tgt = np.asarray(Image.open(it["jpeg_path"]).convert("RGB"), dtype=np.float32) / 255.0
        tgt = torch.from_numpy(tgt).permute(2, 0, 1)
        c = min(args.crop, bay.shape[-2], bay.shape[-1], tgt.shape[-2], tgt.shape[-1])
        c -= c % 2
        bayers.append(bay[:, :c, :c])
        targets.append(tgt[:, :c, :c])
        ccms.append(ccm)
        if "lut_path" in it and os.path.exists(it["lut_path"]):
            luts.append(torch.from_numpy(np.load(it["lut_path"]).astype(np.float32)))
        else:
            luts.append(None)

    if not bayers:
        print("no .npz crops in manifest - rerun build_dataset.py")
        return

    bayer = torch.stack(bayers).to(dev)
    target = torch.stack(targets).to(dev)
    ccm = torch.stack(ccms).to(dev)
    lut = torch.stack([l for l in luts]).to(dev) if all(l is not None for l in luts) else None
    print(f"probing on {bayer.shape[0]} crops of {bayer.shape[-1]}px, LUT={'yes' if lut is not None else 'no'}\n")

    rgb_cam = naive_demosaic(bayer)
    x_lin = torch.einsum("bij,bjhw->bihw", ccm, rgb_cam).clamp(min=0)

    rows = []
    for norm_mode in ["max", "p999", "p99", "p95"]:
        for gamma_mode in ["g22", "bt709", "srgb"]:
            for knee in [0.0, 0.1, 0.25, 0.5]:
                with torch.no_grad():
                    pred = render(x_lin, norm_mode, gamma_mode, knee, lut)
                    l1 = F.l1_loss(pred, target).item()
                    de = delta_e00(pred, target).item()
                    mse = F.mse_loss(pred, target).item()
                    psnr = -10 * math.log10(max(mse, 1e-12))
                rows.append((norm_mode, gamma_mode, knee, l1, de, psnr,
                             pred.mean().item(), target.mean().item()))

    rows.sort(key=lambda r: r[4])  # by dE00
    print(f"{'norm':>6} {'gamma':>6} {'knee':>5} {'L1':>8} {'dE00':>7} {'PSNR':>7} {'pred_mu':>8} {'tgt_mu':>7}")
    print("-" * 62)
    for r in rows[:12]:
        print(f"{r[0]:>6} {r[1]:>6} {r[2]:>5.2f} {r[3]:>8.4f} {r[4]:>7.2f} {r[5]:>7.2f} {r[6]:>8.3f} {r[7]:>7.3f}")

    print("\nworst 3 (for contrast):")
    for r in rows[-3:]:
        print(f"{r[0]:>6} {r[1]:>6} {r[2]:>5.2f} {r[3]:>8.4f} {r[4]:>7.2f} {r[5]:>7.2f} {r[6]:>8.3f} {r[7]:>7.3f}")

    b = rows[0]
    print(f"\nBEST: norm={b[0]} gamma={b[1]} knee={b[2]}  -> dE00 {b[4]:.2f}, PSNR {b[5]:.2f} dB")
    print("This is with a NAIVE demosaic and no training, so it is a floor:")
    print("a trained front-end should beat it. If your trained model is worse than")
    print("this row, the ISP defaults are the problem, not the network.")
    print(f"\nbrightness check: pred mean {b[6]:.3f} vs target mean {b[7]:.3f} "
          f"({'pred too dark' if b[6] < b[7] - 0.02 else 'pred too bright' if b[6] > b[7] + 0.02 else 'matched'})")


if __name__ == "__main__":
    main()
