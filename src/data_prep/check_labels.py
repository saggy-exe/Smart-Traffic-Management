# check_labels.py
import pathlib, sys

TRAIN_ROOT = pathlib.Path(r"D:\huggingface_datasets\UVH-26\UVH-26-Train")

img_exts = ('.jpg','.jpeg','.png','.bmp')
images = list(TRAIN_ROOT.rglob('*'))
images = [p for p in images if p.suffix.lower() in img_exts]

missing = []
found = 0
for img in images:
    stem = img.stem
    # two common label locations to check:
    # 1) labels file in same parent/labels/<stem>.txt
    lbl1 = img.parent / 'labels' / (stem + '.txt')
    # 2) labels file next to image (same dir) with same stem
    lbl2 = img.with_suffix('.txt')
    if lbl1.exists() or lbl2.exists():
        found += 1
    else:
        missing.append(img)

print("Total images scanned:", len(images))
print("Images with label found:", found)
print("Images missing label:", len(missing))
if missing:
    print("\nSample missing images (up to 10):")
    for p in missing[:10]:
        print(p)
else:
    print("No missing labels found.")
