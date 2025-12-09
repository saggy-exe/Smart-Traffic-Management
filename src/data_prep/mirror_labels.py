# mirror_labels.py
import shutil
from pathlib import Path

ROOT = Path(r"D:\huggingface_datasets\UVH-26\UVH-26-Train")
src_labels = list(ROOT.rglob("labels/*.txt"))
print("Found", len(src_labels), "label files under '.../data/.../labels/'")

copied = 0
for s in src_labels:
    # s is like ...\data\000\labels\100103.txt
    # we want dest: ROOT/labels/data/000/100103.txt
    rel = s.parent.parent.relative_to(ROOT)  # data\000
    dest_dir = ROOT / "labels" / rel
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / s.name
    if not dest.exists():
        shutil.copy2(s, dest)
        copied += 1

print("Copied", copied, "files to", ROOT / "labels")
