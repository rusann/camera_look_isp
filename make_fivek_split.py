#!/usr/bin/env python3
"""Create train.txt / test.txt for the FiveK 480p protocol.

Zeng's image archive ships ONLY images; the annotation files that define the official
4500/500 split live in the AdaInt / SepLUT repositories (seplut/annfiles/FiveK/), which is
why they are absent after unzipping. This script tries to fetch them, and otherwise
reconstructs a split from the FiveK index -- loudly, because a reconstructed split is NOT
the published one and the protocol check in compare.py is then the only thing that tells
you whether the numbers are comparable.

  python make_fivek_split.py /mnt/data/FiveK
  python make_fivek_split.py /mnt/data/FiveK --offline     # skip the download attempt
"""
import os, re, sys, argparse, urllib.request

INPUT_CANDS = ("input/JPG/480p", "input/JPG/original", "input/JPG",
               "input/PNG/480p", "input/PNG/480p_16bits_XYZ_WB", "input")
TARGET_CANDS = ("expertC/JPG/480p", "expertC/JPG/original", "expertC/JPG",
                "expertC/PNG/480p", "expertC", "expert_C", "expert_c")
EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
N_TRAIN = 4500          # the published split: 4500 train / 500 test

ANN_URLS = [
    "https://raw.githubusercontent.com/ImCharlesY/AdaInt/main/adaint/annfiles/FiveK/{f}",
    "https://raw.githubusercontent.com/ImCharlesY/AdaInt/master/adaint/annfiles/FiveK/{f}",
    "https://raw.githubusercontent.com/ImCharlesY/SepLUT/main/seplut/annfiles/FiveK/{f}",
    "https://raw.githubusercontent.com/ImCharlesY/SepLUT/master/seplut/annfiles/FiveK/{f}",
]


def resolve(root, cands, what):
    for c in cands:
        p = os.path.join(root, *c.split("/"))
        if os.path.isdir(p) and any(f.lower().endswith(EXTS) for f in os.listdir(p)):
            return p
    print(f"ERROR: no {what} directory with images under {root}.")
    print(f"  tried: {', '.join(cands)}")
    print("  Expected layout after unzipping Zeng's 480p archive:")
    print("    FiveK/input/JPG/480p/    FiveK/expertC/JPG/480p/")
    sys.exit(1)


def stems(d):
    out = {}
    for f in os.listdir(d):
        s, e = os.path.splitext(f)
        if e.lower() in EXTS:
            out[s] = os.path.join(d, f)
    return out


def fetch(root, fname):
    for tpl in ANN_URLS:
        url = tpl.format(f=fname)
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = r.read().decode()
            if data.strip():
                open(os.path.join(root, fname), "w").write(data)
                print(f"  fetched {fname} ({len(data.splitlines())} lines) from {url}")
                return True
        except Exception:
            continue
    return False


def fivek_index(stem):
    """FiveK stems look like a0001-jmac_DSC1459. The leading index is what the official
    split is defined on, so splitting on it keeps our test set a SUBSET of the official
    one even when only part of the dataset was downloaded. Splitting by proportion
    instead would mix official-train images into our test set."""
    m = re.match(r"a(\d{4})", stem)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--offline", action="store_true", help="do not try to download annfiles")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--force", action="store_true", help="overwrite existing txt files")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    inp = resolve(root, INPUT_CANDS, "input")
    tgt = resolve(root, TARGET_CANDS, "expert-C")
    si, st = stems(inp), stems(tgt)
    common = sorted(set(si) & set(st))
    print(f"input   {inp}  ({len(si)} images)")
    print(f"target  {tgt}  ({len(st)} images)")
    print(f"paired  {len(common)}")
    if not common:
        print("ERROR: no stem matches between the two directories. Check the layout.")
        sys.exit(1)

    have = {f: os.path.exists(os.path.join(root, f)) and not args.force
            for f in ("train.txt", "test.txt")}
    if all(have.values()):
        print("train.txt and test.txt already exist (use --force to regenerate)")
        return

    if not args.offline:
        print("trying to fetch the official annotation files ...")
        got = all(fetch(root, f) for f in ("train.txt", "test.txt") if not have[f])
        if got:
            verify(root, common)
            return
        print("  could not fetch; falling back to a reconstructed split")

    # ---- reconstruct -----------------------------------------------------
    idx = {s: fivek_index(s) for s in common}
    if all(v is not None for v in idx.values()):
        train = [s for s in common if idx[s] <= args.n_train]
        test = [s for s in common if idx[s] > args.n_train]
        how = f"by FiveK index (<= a{args.n_train:04d} train, > test)"
    else:
        cut = min(args.n_train, int(len(common) * 0.9))
        train, test = common[:cut], common[cut:]
        how = "by sorted order (stems carry no aNNNN index)"

    if not test:
        print(f"ERROR: the reconstructed test split is empty. All {len(common)} paired "
              f"images fall in the official TRAIN range (index <= {args.n_train}), so none "
              f"of them can serve as held-out test data for this protocol.\n"
              f"  Download the rest of the dataset, or run compare.py with --raw-dir "
              f"instead and report the measured block separately.")
        sys.exit(1)

    open(os.path.join(root, "train.txt"), "w").write("\n".join(train) + "\n")
    open(os.path.join(root, "test.txt"), "w").write("\n".join(test) + "\n")
    print(f"wrote train.txt ({len(train)}) and test.txt ({len(test)}), split {how}")
    if len(test) != 500 or len(train) != args.n_train:
        print("\n  WARNING: this is NOT the published split (4500/500). The test set is a")
        print("  subset of the official one, so the comparison is directionally valid but")
        print("  the sample differs. compare.py's global-LUT check against 20.37 dB is")
        print("  what tells you whether the protocol still lines up.")
    verify(root, common)


def verify(root, common):
    cs = set(common)
    for f in ("train.txt", "test.txt"):
        p = os.path.join(root, f)
        if not os.path.exists(p):
            continue
        names = [os.path.splitext(os.path.basename(l.strip()))[0]
                 for l in open(p) if l.strip()]
        missing = [n for n in names if n not in cs]
        print(f"{f}: {len(names)} entries, {len(names)-len(missing)} usable"
              + (f", {len(missing)} missing files e.g. {missing[:3]}" if missing else ""))


if __name__ == "__main__":
    main()