# make_json_abs.py
# Usage: run this once for a dataset root; it will write ROOT/NAME-abs.json
import json, pathlib, sys

def make_abs_json(root, json_name):
    ROOT = pathlib.Path(root)
    jin = ROOT / json_name
    jout = ROOT / (json_name.replace(".json","-abs.json"))
    print("Processing", jin)
    d = json.load(open(jin,"r",encoding="utf-8"))
    images = d.get("images",[])
    # index disk images by basename
    disk_index = {}
    for p in ROOT.rglob("*"):
        if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp"):
            disk_index[p.name] = p
    missing = 0
    for img in images:
        fn = img.get("file_name")
        p = pathlib.Path(fn)
        # if absolute and exists, keep
        if p.is_absolute() and p.exists():
            img['file_name'] = str(p)
            continue
        # try relative to ROOT
        rel = ROOT / fn
        if rel.exists():
            img['file_name'] = str(rel)
            continue
        # try basename lookup
        base = pathlib.Path(fn).name
        if base in disk_index:
            img['file_name'] = str(disk_index[base])
        else:
            missing += 1
    print("Missing images not found on disk:", missing)
    json.dump(d, open(jout,"w",encoding="utf-8"), ensure_ascii=False)
    print("Wrote:", jout)
    return missing, str(jout)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python make_json_abs.py <dataset_root> <json_filename>")
        sys.exit(1)
    m, out = make_abs_json(sys.argv[1], sys.argv[2])
    print("Done. Missing:", m)
