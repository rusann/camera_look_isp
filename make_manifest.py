import os, json, sys

RAW_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/raws"
JPEG_DIR = sys.argv[2] if len(sys.argv) > 2 else "/mnt/data/jpegs"
OUT_PATH = sys.argv[3] if len(sys.argv) > 3 else "/mnt/data/manifest.jsonl"
CAMERA = sys.argv[4] if len(sys.argv) > 4 else "camera1"
PROMPT = sys.argv[5] if len(sys.argv) > 5 else "default look"

raw_files = {os.path.splitext(f)[0]: f for f in os.listdir(RAW_DIR)} if os.path.isdir(RAW_DIR) else {}
jpeg_files = {os.path.splitext(f)[0]: f for f in os.listdir(JPEG_DIR)}
names = sorted(set(raw_files) & set(jpeg_files)) if raw_files else sorted(jpeg_files)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as out:
    for name in names:
        entry = {
            "jpeg_path": os.path.abspath(os.path.join(JPEG_DIR, jpeg_files[name])),
            "camera": CAMERA,
            "prompt": PROMPT,
            "style_key": f"{CAMERA}_{PROMPT}",
        }
        if name in raw_files:
            entry["raw_path"] = os.path.abspath(os.path.join(RAW_DIR, raw_files[name]))
        out.write(json.dumps(entry) + "\n")
print(f"wrote {len(names)} entries to {OUT_PATH}")