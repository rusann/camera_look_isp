#!/usr/bin/env python3
"""Build the FiveK expert-C 480p layout that compare.py expects.

Produces exactly:

    <root>/input/JPG/480p/<stem>.png     8-bit sRGB inputs
    <root>/expertC/JPG/480p/<stem>.png   8-bit sRGB targets
    <root>/train.txt  <root>/test.txt    the official 4500/500 split when available

Two modes:

  --from-raw   Zero download. Renders inputs from local DNGs and converts the expert
               TIFFs from ProPhoto RGB to sRGB. The input render will NOT be identical
               to Zeng's Lightroom export, so published numbers stay non-comparable --
               compare.py's global-LUT check against 20.37 dB is what tells you so.

  --download   Fetches Zeng's preprocessed 480p set (the actual protocol input) and
               normalises whatever directory layout the archive happens to have.

  python setup_fivek.py /mnt/data/FiveK --from-raw \
      --raw-dir /mnt/data/fivek_dng --expert-dir /mnt/data/fivek_experts/tiff16_c --n 700
  python setup_fivek.py /mnt/data/FiveK --download
"""
import os, re, sys, glob, json, shutil, zipfile, argparse, subprocess, urllib.request
import numpy as np
from PIL import Image

GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1Y1Rv3uGiJkP6CIrNTSKxPn1p-WFAc48a"
ANN_URLS = [
    "https://raw.githubusercontent.com/ImCharlesY/AdaInt/{b}/adaint/annfiles/FiveK/{f}",
    "https://raw.githubusercontent.com/ImCharlesY/SepLUT/{b}/seplut/annfiles/FiveK/{f}",
]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
RAW_EXTS = (".dng", ".DNG", ".cr2", ".CR2", ".nef", ".NEF", ".arw", ".ARW")
OUT_IN = ("input", "JPG", "480p")
OUT_GT = ("expertC", "JPG", "480p")


# ============================== colour ==============================
# ProPhoto RGB (ROMM) -> sRGB. The FiveK expert TIFFs are 16-bit ProPhoto with a D50 white
# point; reading their values as if they were sRGB silently shifts every colour, which
# corrupts dE and PSNR for a reason unrelated to any method being compared. Matrices are
# multiplied numerically here rather than transcribed, so there is nothing to mistype.
PROPHOTO_TO_XYZ_D50 = np.array([[0.7976749, 0.1351917, 0.0313534],
                                [0.2880402, 0.7118741, 0.0000857],
                                [0.0000000, 0.0000000, 0.8252100]])
XYZ_D50_TO_SRGB_LIN = np.array([[3.1338561, -1.6168667, -0.4906146],
                                [-0.9787684, 1.9161415, 0.0334540],
                                [0.0719453, -0.2289914, 1.4052427]])
PROPHOTO_TO_SRGB_LIN = XYZ_D50_TO_SRGB_LIN @ PROPHOTO_TO_XYZ_D50


def prophoto_to_srgb(arr):
    """arr: float32 HxWx3 in [0,1], ProPhoto-encoded. Returns sRGB-encoded [0,1]."""
    # ROMM decoding: linear segment below 16/512, gamma 1.8 above
    lin = np.where(arr < 16.0 / 512.0, arr / 16.0, np.power(np.clip(arr, 0, None), 1.8))
    out = lin.reshape(-1, 3) @ PROPHOTO_TO_SRGB_LIN.T
    out = np.clip(out.reshape(arr.shape), 0.0, 1.0)
    return np.where(out <= 0.0031308, 12.92 * out,
                    1.055 * np.power(out, 1 / 2.4) - 0.055).clip(0, 1)


def read_target(path, assume):
    """Expert TIFF -> 8-bit sRGB PIL image."""
    try:
        import tifffile
        arr = tifffile.imread(path)
    except Exception:
        return Image.open(path).convert("RGB")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    arr = arr[..., :3]
    arr = (arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
           if np.issubdtype(arr.dtype, np.integer) else arr.astype(np.float32))
    if assume == "prophoto":
        arr = prophoto_to_srgb(arr)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8))


def render_raw(path):
    """Neutral sRGB render of a RAW file. Reuses bench_distill's loader when importable so
    the input distribution matches the rest of the project exactly."""
    try:
        import bench_distill as B
        return B.load_input_image(path).convert("RGB")
    except Exception:
        pass
    import rawpy
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True,
                              output_bps=8, gamma=(2.222, 4.5))
    return Image.fromarray(rgb).convert("RGB")


def resize_short(im, side):
    w, h = im.size
    s = side / min(w, h)
    return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)


# ============================== split files ==============================

def fetch_split(root):
    ok = {}
    for f in ("train.txt", "test.txt"):
        dst = os.path.join(root, f)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            ok[f] = True
            continue
        for tpl in ANN_URLS:
            for b in ("main", "master"):
                try:
                    with urllib.request.urlopen(tpl.format(b=b, f=f), timeout=20) as r:
                        data = r.read().decode()
                    if data.strip():
                        open(dst, "w").write(data)
                        print(f"  {f}: fetched {len(data.splitlines())} entries")
                        ok[f] = True
                        break
                except Exception:
                    continue
            if ok.get(f):
                break
        if not ok.get(f):
            print(f"  {f}: could not fetch")
    return all(ok.get(f) for f in ("train.txt", "test.txt"))


def read_split(root, f):
    p = os.path.join(root, f)
    if not os.path.exists(p):
        return None
    return [os.path.splitext(os.path.basename(l.strip()))[0] for l in open(p) if l.strip()]


def write_split(root, stems, official_ok):
    """Keep our test set a SUBSET of the official one. Splitting by proportion instead would
    leak official-train images into the test set, which is the one thing that would make the
    comparison invalid rather than merely narrower."""
    have = set(stems)
    tr, te = read_split(root, "train.txt"), read_split(root, "test.txt")
    if official_ok and tr and te:
        tr2 = [s for s in tr if s in have]
        te2 = [s for s in te if s in have]
        if te2:
            open(os.path.join(root, "train.txt"), "w").write("\n".join(tr2) + "\n")
            open(os.path.join(root, "test.txt"), "w").write("\n".join(te2) + "\n")
            print(f"  official split, intersected with what is present: "
                  f"{len(tr2)} train / {len(te2)} test")
            return
        print("  official test entries are all absent locally; reconstructing")
    idx = {s: (int(m.group(1)) if (m := re.match(r"a(\d{4})", s)) else None) for s in stems}
    if all(v is not None for v in idx.values()):
        tr2 = [s for s in stems if idx[s] <= 4500]
        te2 = [s for s in stems if idx[s] > 4500]
        how = "FiveK index, cut at a4500 (same boundary as the official split)"
    else:
        cut = int(len(stems) * 0.9)
        tr2, te2 = stems[:cut], stems[cut:]
        how = "sorted order (stems carry no aNNNN index)"
    if not te2:
        print(f"\n  ERROR: every one of the {len(stems)} images falls in the official TRAIN\n"
              f"  range, so none can be held out. Download a slice that includes indices\n"
              f"  above a4500, or pass --split-frac to carve a test set out anyway.")
        sys.exit(1)
    open(os.path.join(root, "train.txt"), "w").write("\n".join(tr2) + "\n")
    open(os.path.join(root, "test.txt"), "w").write("\n".join(te2) + "\n")
    print(f"  reconstructed split by {how}: {len(tr2)} train / {len(te2)} test")


# ============================== download mode ==============================

def have_gdown():
    try:
        import gdown  # noqa
        return True
    except ImportError:
        return False


def do_download(root, dl_dir):
    if not have_gdown():
        print("gdown is required:  pip install gdown")
        sys.exit(1)
    os.makedirs(dl_dir, exist_ok=True)
    print(f"downloading Zeng's 480p FiveK into {dl_dir} ...")
    print("  (if this stalls on Google's quota, open the folder in a browser and place the\n"
          "   archives in that directory by hand, then rerun -- already-present files are kept)")
    r = subprocess.run(["gdown", "--folder", "--remaining-ok", "-O", dl_dir, GDRIVE_FOLDER])
    if r.returncode != 0:
        print(f"\ngdown failed. Manual route:\n  {GDRIVE_FOLDER}\n"
              f"  Download the FiveK 480p archive, put it in {dl_dir}, rerun this script.")
    for z in glob.glob(os.path.join(dl_dir, "**", "*.zip"), recursive=True):
        print(f"  unzipping {os.path.basename(z)}")
        with zipfile.ZipFile(z) as f:
            f.extractall(dl_dir)
    return dl_dir


def find_dirs(tree):
    """Locate the input and expert-C directories anywhere under a tree. The archives differ
    in nesting between mirrors, so matching on path semantics beats hard-coding a layout."""
    cands = []
    for d, _, files in os.walk(tree):
        n = sum(1 for f in files if f.lower().endswith(IMG_EXTS))
        if n >= 10:
            cands.append((d, n, d.lower().replace("\\", "/")))
    inp = gt = None
    for d, n, low in sorted(cands, key=lambda t: -t[1]):
        if "16bit" in low or "xyz" in low:
            continue
        if gt is None and re.search(r"expert[_\-]?c|expertc", low):
            gt = (d, n)
        elif inp is None and "input" in low:
            inp = (d, n)
    if inp is None:
        for d, n, low in sorted(cands, key=lambda t: -t[1]):
            if "16bit" not in low and "xyz" not in low and (gt is None or d != gt[0]):
                inp = (d, n)
                break
    return inp, gt


# ============================== main ==============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--from-raw", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--expert-dir", default=None)
    ap.add_argument("--dl-dir", default=None, help="where archives land (default <root>/_dl)")
    ap.add_argument("--side", type=int, default=480)
    ap.add_argument("--n", type=int, default=0, help="cap the number of pairs (0 = all)")
    ap.add_argument("--target-space", choices=["prophoto", "srgb"], default="prophoto",
                    help="colour space of the expert TIFFs. FiveK ships ProPhoto RGB; "
                         "reading them as sRGB shifts every colour.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    os.makedirs(root, exist_ok=True)
    d_in = os.path.join(root, *OUT_IN)
    d_gt = os.path.join(root, *OUT_GT)
    os.makedirs(d_in, exist_ok=True)
    os.makedirs(d_gt, exist_ok=True)

    if not args.from_raw and not args.download:
        ap.error("pass --from-raw (no download) or --download")

    print("fetching the official split files ...")
    official = fetch_split(root)

    stems = []

    if args.download:
        tree = do_download(root, args.dl_dir or os.path.join(root, "_dl"))
        inp, gt = find_dirs(tree)
        if not inp or not gt:
            print(f"\ncould not locate input / expertC directories under {tree}.")
            print("  found these image directories:")
            for d, _, files in os.walk(tree):
                n = sum(1 for f in files if f.lower().endswith(IMG_EXTS))
                if n >= 10:
                    print(f"    {n:6d}  {d}")
            sys.exit(1)
        print(f"  input  {inp[0]}  ({inp[1]} images)")
        print(f"  expert {gt[0]}  ({gt[1]} images)")
        si = {os.path.splitext(f)[0]: os.path.join(inp[0], f) for f in os.listdir(inp[0])
              if f.lower().endswith(IMG_EXTS)}
        sg = {os.path.splitext(f)[0]: os.path.join(gt[0], f) for f in os.listdir(gt[0])
              if f.lower().endswith(IMG_EXTS)}
        common = sorted(set(si) & set(sg))
        if args.n:
            te = set(read_split(root, "test.txt") or [])
            common = ([s for s in common if s in te][:args.n // 2]
                      + [s for s in common if s not in te][:args.n - args.n // 2]) or common[:args.n]
            common = sorted(set(common))
        print(f"  {len(common)} pairs -> normalising into the expected layout")
        for i, s in enumerate(common, 1):
            for src, dst in ((si[s], os.path.join(d_in, s + os.path.splitext(si[s])[1])),
                             (sg[s], os.path.join(d_gt, s + os.path.splitext(sg[s])[1]))):
                if args.force or not os.path.exists(dst):
                    try:
                        os.link(src, dst)          # hard link: no second copy on disk
                    except OSError:
                        shutil.copy2(src, dst)
            stems.append(s)
            if i % 500 == 0:
                print(f"    {i}/{len(common)}")

    if args.from_raw:
        if not (args.raw_dir and args.expert_dir):
            ap.error("--from-raw needs --raw-dir and --expert-dir")
        raws = {}
        for e in RAW_EXTS:
            for p in glob.glob(os.path.join(args.raw_dir, "*" + e)):
                raws[os.path.splitext(os.path.basename(p))[0]] = p
        exps = {os.path.splitext(f)[0]: os.path.join(args.expert_dir, f)
                for f in os.listdir(args.expert_dir) if f.lower().endswith(IMG_EXTS)}
        common = sorted(set(raws) & set(exps))
        if not common:
            print(f"no stem matches between {args.raw_dir} ({len(raws)} raws) and "
                  f"{args.expert_dir} ({len(exps)} targets).")
            print(f"  raw stems e.g.:    {sorted(raws)[:3]}")
            print(f"  target stems e.g.: {sorted(exps)[:3]}")
            sys.exit(1)
        if args.n:
            te = set(read_split(root, "test.txt") or [])
            pick = [s for s in common if s in te][:args.n // 2]
            pick += [s for s in common if s not in te][:args.n - len(pick)]
            common = sorted(set(pick)) or common[:args.n]
        print(f"rendering {len(common)} pairs at short edge {args.side} "
              f"(targets assumed {args.target_space}) ...")
        for i, s in enumerate(common, 1):
            pa, pb = os.path.join(d_in, s + ".png"), os.path.join(d_gt, s + ".png")
            if not args.force and os.path.exists(pa) and os.path.exists(pb):
                stems.append(s)
                continue
            try:
                a = resize_short(render_raw(raws[s]), args.side)
                b = resize_short(read_target(exps[s], args.target_space), args.side)
                if a.size != b.size:
                    b = b.resize(a.size, Image.LANCZOS)
                a.save(pa)                      # PNG: lossless, no double JPEG error
                b.save(pb)
                stems.append(s)
            except Exception as e:
                print(f"  skip {s}: {e}")
            if i % 50 == 0:
                print(f"  {i}/{len(common)}")

    stems = sorted(set(stems))
    if not stems:
        print("nothing produced"); sys.exit(1)
    print(f"\n{len(stems)} pairs in place")
    write_split(root, stems, official)

    # ---- verify exactly what compare.py will see -------------------------
    ni = len([f for f in os.listdir(d_in) if f.lower().endswith(IMG_EXTS)])
    ng = len([f for f in os.listdir(d_gt) if f.lower().endswith(IMG_EXTS)])
    tr, te = read_split(root, "train.txt"), read_split(root, "test.txt")
    have = {os.path.splitext(f)[0] for f in os.listdir(d_in)}
    print(f"\n{root}")
    print(f"  input/JPG/480p      {ni} images")
    print(f"  expertC/JPG/480p    {ng} images")
    print(f"  train.txt           {len(tr)} entries, {sum(s in have for s in tr)} usable")
    print(f"  test.txt            {len(te)} entries, {sum(s in have for s in te)} usable")
    if args.from_raw:
        print("\n  NOTE: inputs are our own RAW render, not Zeng's Lightroom export, so the\n"
              "  published block stays non-comparable. compare.py checks this for you by\n"
              "  measuring the global-LUT row against the 20.37 dB published for it.")
    print("\nnext:")
    print(f"  python compare.py --fivek-480p {root} --out_dir release/checkpoints_lut \\")
    print(f"      --style_cache_dir release/style_cache_lut --use_lut --ip2p \\")
    print(f"      --work compare_work --dry-run")


if __name__ == "__main__":
    main()