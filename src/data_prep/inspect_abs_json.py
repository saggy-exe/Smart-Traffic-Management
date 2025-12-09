# inspect_abs_json.py
import json, pathlib, sys

p = pathlib.Path(r"D:\huggingface_datasets\UVH-26\UVH-26-Train\UVH-26-MV-Train-abs.json")
print("Inspecting JSON:", p)
d = json.load(open(p, "r", encoding="utf-8"))
imgs = d.get("images", [])
print("Total images in JSON:", len(imgs))

exists = 0
missing_examples = []
for i, img in enumerate(imgs):
    fn = img.get("file_name")
    fp = pathlib.Path(fn)
    if fp.exists():
        exists += 1
    else:
        if len(missing_examples) < 20:
            missing_examples.append((i, fn, str(fp)))
# print summary
print("Exists on disk:", exists)
print("Missing on disk:", len(imgs) - exists)

print("\nFirst 30 image entries and exists checks:")
for img in imgs[:30]:
    fn = img.get("file_name")
    fp = pathlib.Path(fn)
    print(fn, "->", "EXISTS" if fp.exists() else "MISSING")

if missing_examples:
    print("\nSample missing entries (index, file_name, attempted path):")
    for me in missing_examples[:10]:
        print(me)
