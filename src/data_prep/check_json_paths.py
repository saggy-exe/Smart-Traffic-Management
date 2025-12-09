# check_json_paths.py
import json, os, pathlib
json_path = pathlib.Path(r"D:\huggingface_datasets\UVH-26\UVH-26-Train\UVH-26-MV-Train.json")
print("JSON:", json_path)
d = json.load(open(json_path, "r", encoding="utf-8"))
imgs = d.get("images", [])
print("Total images in JSON:", len(imgs))
print("\nFirst 20 images and existence checks:\n")
for img in imgs[:20]:
    fname = img.get("file_name")
    # two checks: path relative to JSON dir, and absolute if already absolute
    rel = json_path.parent / fname
    abs_path = pathlib.Path(fname) if os.path.isabs(fname) else rel
    print(fname)
    print("   exists at relative-to-json:", rel.exists(), "->", rel)
    if abs_path != rel:
        print("   exists at absolute:", abs_path.exists(), "->", abs_path)
    print("")
