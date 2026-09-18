"""
Server-side downloader for the camera-look ISP project.

Run ON THE SERVER. Downloads to /mnt/data (the big mounted drive), not the boot disk.

  python download_data.py --list
  python download_data.py --targets models,haldclut
  python download_data.py --targets fivek_dng --max-files 200 --max-gb 15   # subset, recommended
  python download_data.py --targets fivek_full        # ~47GB, slow
  python download_data.py --manual                     # print instructions for gated sets

All downloads are resumable: re-running skips completed files and continues partial ones.
"""

import os, sys, json, shutil, argparse, subprocess
import urllib.request, urllib.error, urllib.parse

DATA_ROOT = os.environ.get("CAMERA_LOOK_DATA", "/mnt/data")

# ---------------------------------------------------------------- targets

AUTO_TARGETS = {
    "models": {
        "desc": "Backbone weights: Qwen3-VL-4B-Instruct + FLUX.2-klein-4B (~18GB)",
        "size_gb": 18,
        "kind": "hf_model",
        "repos": ["Qwen/Qwen3-VL-4B-Instruct", "black-forest-labs/FLUX.2-klein-4B"],
        "dest": "models",
    },
    "haldclut": {
        "desc": "RawTherapee Film Simulation HaldCLUT pack (~402MB) - film/camera looks as LUTs",
        "size_gb": 0.5,
        "kind": "http",
        "url": "https://rawtherapee.com/shared/HaldCLUT.zip",
        "dest": "luts",
        "unzip": True,
    },
    "fivek_dng": {
        "desc": "FiveK DNG SUBSET - N individual RAWs stratified across camera bodies (use --max-files)",
        "size_gb": 6,
        "kind": "fivek_subset",
        "dest": "fivek_dng",
    },
    "fivek_experts": {
        "desc": "FiveK expert retouches (tiff16 a-e) for the DNGs you already have - REAL human edits",
        "size_gb": 8,
        "kind": "fivek_experts",
        "dest": "fivek_experts",
    },
    "fivek_hf": {
        "desc": "MIT-Adobe FiveK, community webp mirror - WARNING: full repo, can be very large",
        "size_gb": 25,
        "kind": "hf_dataset",
        "repos": ["logasja/mit-adobe-fivek"],
        "dest": "fivek_hf",
    },
    "fivek_full": {
        "desc": "MIT-Adobe FiveK full archive from archive.org: 5000 DNG + 5 expert TIFFs (~47GB)",
        "size_gb": 50,
        "kind": "archive_org",
        "identifier": "fivek",
        "dest": "fivek_full",
    },
}

FIVEK_INDEX_URLS = [
    "https://huggingface.co/datasets/yuukicammy/MIT-Adobe-FiveK/raw/main/training.json",
    "https://huggingface.co/datasets/yuukicammy/MIT-Adobe-FiveK/raw/main/validation.json",
    "https://huggingface.co/datasets/yuukicammy/MIT-Adobe-FiveK/raw/main/testing.json",
]

MANUAL_TARGETS = {
    "SID (See-in-the-Dark)": (
        "Sony 25GB / Fuji 52GB. The authors REMOVED their download script in Sep 2025 after\n"
        "    large Google storage bills, and explicitly ask that you download manually and keep\n"
        "    a local copy rather than re-downloading per run. Please respect that.\n"
        "    https://github.com/cchen156/Learning-to-See-in-the-Dark"
    ),
    "ZRR (Zurich RAW to RGB)": (
        "Requires filling a registration form; no direct link.\n"
        "    https://people.ee.ethz.ch/~ihnatova/pynet.html"
    ),
    "RAISE": (
        "Served as a CSV of per-file URLs after agreeing to terms; download the CSV, then fetch.\n"
        "    http://loki.disi.unitn.it/RAISE/"
    ),
    "PPR10K": (
        "Hosted on Google Drive / Baidu; links rotate. See the repo README.\n"
        "    https://github.com/csjliang/PPR10K"
    ),
    "NUS-8 / Gehler-Shi": (
        "Illuminant-estimation sets, per-camera archives from the project pages.\n"
        "    https://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html"
    ),
    "MDRAW / AceTone-800K / CameraMaster-78K / CIC-2025 profile set": (
        "These come from recent papers. I could not verify a public download for any of them;\n"
        "    several are 'request from the authors'. Email the corresponding author, and do not\n"
        "    block your build on them - the HaldCLUT pack + FiveK covers the same training need."
    ),
}

# ---------------------------------------------------------------- helpers

def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def free_gb(path):
    while not os.path.exists(path):
        path = os.path.dirname(path) or "/"
    return shutil.disk_usage(path).free / 1024 ** 3


def ensure_space(need_gb, path):
    have = free_gb(path)
    if have < need_gb * 1.15:
        print(f"  !! need ~{need_gb}GB (+15% headroom), only {have:.1f}GB free at {path}")
        print("     Free space or mount a bigger volume, then re-run. Skipping.")
        return False
    return True


def http_download(url, dest_path):
    """Resumable single-file download via wget -c (handles redirects, retries, partials)."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  -> {url}")
    cmd = ["wget", "-c", "--tries=5", "--timeout=30",
           "--progress=dot:giga", "-O", dest_path, url]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"  !! download failed (exit {r.returncode}): {url}")
        return False
    return True


def maybe_unzip(zip_path, out_dir):
    marker = os.path.join(out_dir, ".unzipped")
    if os.path.exists(marker):
        print("  already extracted, skipping")
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f"  extracting {os.path.basename(zip_path)} ...")
    r = subprocess.run(["unzip", "-q", "-o", zip_path, "-d", out_dir])
    if r.returncode == 0:
        open(marker, "w").close()
        print("  extracted")
    else:
        print("  !! unzip failed; is `unzip` installed? (apt install -y unzip)")


def hf_snapshot(repo_id, dest, repo_type):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  !! huggingface_hub not installed: pip install huggingface_hub")
        return False
    print(f"  -> {repo_type}: {repo_id}")
    try:
        snapshot_download(repo_id=repo_id, repo_type=repo_type,
                          local_dir=dest, resume_download=True,
                          max_workers=8)
        return True
    except Exception as e:
        print(f"  !! failed: {e}")
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            print("     Looks gated/unauthorized. Run: huggingface-cli login")
        return False


def fetch_fivek_index():
    """Merge FiveK split indexes -> {basename: {dng_url, camera}}."""
    merged = {}
    for url in FIVEK_INDEX_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            merged.update(data)
            print(f"  index ok: {url.rsplit('/',1)[-1]} ({len(data)} entries)")
        except Exception as e:
            print(f"  index miss: {url.rsplit('/',1)[-1]} ({e})")
    return merged


def fivek_subset(dest, max_files, max_gb):
    """Download N FiveK DNGs, round-robin across camera models for device diversity."""
    idx = fetch_fivek_index()
    if not idx:
        print("  !! could not fetch any FiveK index; check outbound internet")
        return
    by_cam = {}
    for name, rec in idx.items():
        try:
            cam = rec["camera"]
            key = f"{cam.get('make','?')} {cam.get('model','?')}".strip()
            url = rec["urls"]["dng"]
        except Exception:
            continue
        by_cam.setdefault(key, []).append((name, url))

    for k in by_cam:
        by_cam[k].sort()
    cams = sorted(by_cam)
    print(f"  {len(idx)} images across {len(cams)} camera bodies")

    picked, i = [], 0
    while len(picked) < max_files:
        added = False
        for c in cams:
            if i < len(by_cam[c]):
                picked.append((c, *by_cam[c][i]))
                added = True
                if len(picked) >= max_files:
                    break
        if not added:
            break
        i += 1

    cams_used = sorted({c for c, _, _ in picked})
    print(f"  selected {len(picked)} files spanning {len(cams_used)} bodies")

    os.makedirs(dest, exist_ok=True)
    budget = max_gb * 1024 ** 3
    used = sum(os.path.getsize(os.path.join(dest, f))
               for f in os.listdir(dest) if f.endswith(".dng"))
    got = 0
    for n, (cam, name, url) in enumerate(picked, 1):
        out = os.path.join(dest, f"{name}.dng")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            got += 1
            continue
        if used >= budget:
            print(f"  budget {max_gb}GB reached, stopping at {got} files")
            break
        if not ensure_space(1, dest):
            break
        print(f"  [{n}/{len(picked)}] {cam}: {name}")
        if http_download(url, out) and os.path.exists(out):
            used += os.path.getsize(out)
            got += 1
    print(f"  {got} DNGs in {dest} ({human(used)})")
    print(f"  bodies: {', '.join(cams_used[:12])}{' ...' if len(cams_used) > 12 else ''}")


def fivek_experts(dest, dng_dir, experts, max_gb):
    """Download expert retouches for the DNGs already present.

    These are the FiveK dataset's five human retouches per photo. Unlike the HaldCLUT pack
    (a creative-effect collection), they are global tonal/colour edits of real photographs --
    the kind of transform the parametric ISP can actually represent, and the standard
    benchmark targets for learned-ISP work (expert C is the usual choice).
    """
    idx = fetch_fivek_index()
    if not idx:
        print("  !! could not fetch the FiveK index")
        return
    if not os.path.isdir(dng_dir):
        print(f"  !! no DNG directory at {dng_dir}; run --targets fivek_dng first")
        return
    have = {os.path.splitext(f)[0] for f in os.listdir(dng_dir) if f.lower().endswith(".dng")}
    print(f"  {len(have)} local DNGs, fetching experts {','.join(experts)}")
    if not have:
        print("  !! nothing to match; run --targets fivek_dng first")
        return

    budget = max_gb * 1024 ** 3
    used, got = 0, 0
    for ex in experts:
        out_dir = os.path.join(dest, f"tiff16_{ex}")
        os.makedirs(out_dir, exist_ok=True)
        for n, name in enumerate(sorted(have), 1):
            rec = idx.get(name)
            if not rec:
                continue
            try:
                url = rec["urls"]["tiff16"][ex]
            except (KeyError, TypeError):
                continue
            out = os.path.join(out_dir, f"{name}.tif")
            if os.path.exists(out) and os.path.getsize(out) > 0:
                got += 1
                continue
            if used >= budget:
                print(f"  budget {max_gb}GB reached at {got} files")
                return
            print(f"  [{ex} {n}/{len(have)}] {name}")
            if http_download(url, out) and os.path.exists(out):
                used += os.path.getsize(out)
                got += 1
    print(f"  {got} expert TIFFs in {dest} ({human(used)})")


def archive_org_files(identifier):
    """List downloadable files for an archive.org item via its metadata API."""
    url = f"https://archive.org/metadata/{identifier}"
    with urllib.request.urlopen(url, timeout=30) as r:
        meta = json.load(r)
    files = meta.get("files", [])
    out = []
    for f in files:
        name = f.get("name", "")
        if name.startswith("_") or name.endswith((".xml", ".sqlite", ".torrent")):
            continue
        out.append((name, int(f.get("size", 0) or 0)))
    return out

# ---------------------------------------------------------------- runners

def run_target(key, spec, max_files=200, max_gb=20, experts=("c",)):
    dest = os.path.join(DATA_ROOT, spec["dest"])
    print(f"\n=== {key}: {spec['desc']}")
    need = min(spec.get("size_gb", 1), max_gb) if spec["kind"] == "fivek_subset" else spec.get("size_gb", 1)
    if not ensure_space(need, DATA_ROOT):
        return
    os.makedirs(dest, exist_ok=True)
    kind = spec["kind"]

    if kind == "fivek_subset":
        fivek_subset(dest, max_files, max_gb)

    elif kind == "fivek_experts":
        fivek_experts(dest, os.path.join(DATA_ROOT, "fivek_dng"), experts, max_gb)

    elif kind == "http":
        fname = spec["url"].rsplit("/", 1)[-1]
        path = os.path.join(dest, fname)
        if http_download(spec["url"], path) and spec.get("unzip"):
            maybe_unzip(path, dest)

    elif kind in ("hf_model", "hf_dataset"):
        repo_type = "model" if kind == "hf_model" else "dataset"
        for repo in spec["repos"]:
            sub = os.path.join(dest, repo.replace("/", "__"))
            hf_snapshot(repo, sub, repo_type)

    elif kind == "archive_org":
        ident = spec["identifier"]
        print(f"  listing files for archive.org/{ident} ...")
        try:
            files = archive_org_files(ident)
        except Exception as e:
            print(f"  !! could not list item: {e}")
            return
        total = sum(s for _, s in files)
        print(f"  {len(files)} files, {human(total)} total")
        for name, size in files:
            out_path = os.path.join(dest, name)
            if os.path.exists(out_path) and os.path.getsize(out_path) == size and size > 0:
                print(f"  ok (have) {name}")
                continue
            url = f"https://archive.org/download/{ident}/{urllib.parse.quote(name)}"
            http_download(url, out_path)

    print(f"  done -> {dest}")


def print_manual():
    print("\nThese require manual steps (registration, author request, or an explicit")
    print("request from the maintainers not to automate downloads):\n")
    for name, note in MANUAL_TARGETS.items():
        print(f"  * {name}\n    {note}\n")
    print(f"Put anything you fetch by hand under {DATA_ROOT}/<name>/ to keep paths consistent.\n")


def print_list():
    print(f"\nData root: {DATA_ROOT}  (free: {free_gb(DATA_ROOT):.1f}GB)\n")
    print("Automatic targets (--targets a,b,c):\n")
    for k, v in AUTO_TARGETS.items():
        print(f"  {k:<12} ~{v['size_gb']:>4}GB  {v['desc']}")
    print("\nManual-only sources: run with --manual\n")


def main():
    global DATA_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--targets", default="", help="comma-separated, or 'all'")
    ap.add_argument("--max-files", type=int, default=200,
                    help="fivek_dng: how many RAW files to fetch (default 200)")
    ap.add_argument("--max-gb", type=float, default=20,
                    help="fivek_dng: stop after this many GB (default 20)")
    ap.add_argument("--experts", default="c",
                    help="fivek_experts: which retouches (a-e). C is the usual benchmark.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--manual", action="store_true")
    args = ap.parse_args()
    DATA_ROOT = args.data_root
    os.makedirs(DATA_ROOT, exist_ok=True)

    if args.list or (not args.targets and not args.manual):
        print_list()
        return
    if args.manual:
        print_manual()
        return

    keys = list(AUTO_TARGETS) if args.targets == "all" else \
        [k.strip() for k in args.targets.split(",") if k.strip()]
    unknown = [k for k in keys if k not in AUTO_TARGETS]
    if unknown:
        print(f"unknown target(s): {unknown}\nrun --list to see valid names")
        sys.exit(1)
    for k in keys:
        run_target(k, AUTO_TARGETS[k], max_files=args.max_files, max_gb=args.max_gb,
                   experts=[e.strip() for e in args.experts.split(",") if e.strip()])
    print("\nAll requested targets processed.")


if __name__ == "__main__":
    main()
