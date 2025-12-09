# coco2yolo.py
import json, os, sys
from pathlib import Path

ROOT = Path(r"D:\huggingface_datasets\UVH-26")  # change if your dataset root is elsewhere
# Try to detect typical train/val names
candidates = {"train": ["UVH-26-Train","train","train_images"], "val": ["UVH-26-Val","val","validation"]}
def find_folder(names):
    for n in names:
        p = ROOT / n
        if p.exists(): return p
    # fallback: first folder with a .json inside
    for p in ROOT.iterdir():
        if p.is_dir():
            for f in p.rglob("*.json"):
                return p
    return None

train_root = find_folder(candidates["train"])
val_root   = find_folder(candidates["val"])

print("train_root:", train_root)
print("val_root:  ", val_root)

def convert(folder):
    # find json
    jsons = list(folder.rglob("*.json"))
    if not jsons:
        print("No JSON found in", folder); return
    ann = jsons[0]
    print("Using annotation:", ann)
    a = json.load(open(ann))
    images = {img['id']:img for img in a.get('images',[])}
    annotations = a.get('annotations',[])
    categories = a.get('categories',[])
    cat_map = {c['id']:i for i,c in enumerate(categories)}  # 0..nc-1
    print("Found", len(images), "images and", len(annotations), "annotations. classes:", len(categories))

    # ensure labels folder
    for img_id, img in images.items():
        img_path = folder / img.get('file_name')
        if not img_path.exists():
            # try images/... format
            img_path = next(folder.rglob(img.get('file_name')), None)
        if img_path is None:
            continue
        lbl_dir = img_path.parent / "labels"
        lbl_dir.mkdir(exist_ok=True)
    # group annotations by image
    ann_by_img = {}
    for ann_item in annotations:
        iid = ann_item['image_id']
        ann_by_img.setdefault(iid,[]).append(ann_item)

    for img_id, img in images.items():
        file_name = img.get('file_name')
        img_path = next(folder.rglob(file_name), None)
        if img_path is None:
            # skip if image missing
            continue
        w,h = img.get('width'), img.get('height')
        lbl_path = img_path.parent / "labels" / (Path(file_name).stem + ".txt")
        lines = []
        for ann_item in ann_by_img.get(img_id,[]):
            cat = ann_item['category_id']
            cls = cat_map.get(cat,0)
            x,y,wbox,hbox = ann_item['bbox']  # COCO format (x,y,w,h)
            # convert to x_center y_center w h (normalized)
            xc = (x + wbox/2)/w
            yc = (y + hbox/2)/h
            nw = wbox / w
            nh = hbox / h
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        if lines:
            lbl_path.write_text("\n".join(lines))
    print("Conversion done for", folder)

if train_root: convert(train_root)
if val_root: convert(val_root)
