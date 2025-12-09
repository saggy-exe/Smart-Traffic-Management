# debug_label_paths.py
from pathlib import Path
ROOT = Path(r"D:\huggingface_datasets\UVH-26\UVH-26-Train")
img_exts = ('.jpg','.jpeg','.png','.bmp')
imgs = [p for p in ROOT.rglob('*') if p.suffix.lower() in img_exts]
print("Total images found:", len(imgs))
print("Showing first 20 image checks...\n")
for img in imgs[:20]:
    stem = img.stem
    # 3 common label locations to check
    label_candidates = [
        img.parent / 'labels' / (stem + '.txt'),  # labels folder next to images
        img.parent / (stem + '.txt'),             # label in same folder as image
        ROOT / 'labels' / img.parent.relative_to(ROOT) / (stem + '.txt')  # mirrored labels root (what the fixer created)
    ]
    print("IMAGE:", img)
    for lc in label_candidates:
        print("  CHECK:", lc, "->", "EXISTS" if lc.exists() else "MISSING")
    print("")
