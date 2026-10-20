"""Klipper ut kvart annotert eple frå datasettbileta til data/crops/{ripe,unripe}/
og skriv data/crops/metadata.csv med éi rad per eple.

Støttar COCO (.json) og YOLO (.txt per bilete). Sjekk formatet etter nedlasting:
    python src/prepare_crops.py --inspect

Kolonna `group` i metadata.csv seier kva foto eit eple kjem frå (tiles som
_MG_2657_24.png og _MG_2657_25.png får same gruppe). Notatboka brukar denne til
å splitte trening/test slik at eple frå same foto aldri havnar på begge sider.
"""
import argparse
import json
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def label_from_name(name: str):
    """Map kategorinamn til ripe/unripe. 'unripe' må sjekkast før 'ripe'."""
    n = name.lower()
    if "unripe" in n or "green" in n or "umoden" in n:
        return "unripe"
    if "ripe" in n or "red" in n or "moden" in n:
        return "ripe"
    return None


def group_key(stem: str) -> str:
    """'_MG_2657_24' -> '_MG_2657' så tiles frå same foto blir halde saman."""
    return re.sub(r"_\d+$", "", stem)


def find_images(raw: Path):
    return {p.stem: p for p in raw.rglob("*") if p.suffix.lower() in IMG_EXT}


def load_coco(raw: Path, images_by_stem):
    """Returnerer liste av (bildesti, [x0,y0,x1,y1], label)."""
    out = []
    for jf in raw.rglob("*.json"):
        try:
            d = json.loads(jf.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(d, dict) or "annotations" not in d or "images" not in d:
            continue
        cats = {c["id"]: label_from_name(c["name"]) for c in d.get("categories", [])}
        imgs = {im["id"]: im["file_name"] for im in d["images"]}
        for a in d["annotations"]:
            lab = cats.get(a["category_id"])
            fname = imgs.get(a["image_id"])
            if lab is None or fname is None:
                continue
            path = images_by_stem.get(Path(fname).stem)
            if path is None:
                continue
            x, y, w, h = a["bbox"]
            out.append((path, [x, y, x + w, y + h], lab))
    return out


def load_yolo(raw: Path, images_by_stem, class_names):
    out = []
    for tf in raw.rglob("*.txt"):
        path = images_by_stem.get(tf.stem)
        if path is None:
            continue
        W, H = Image.open(path).size
        for line in tf.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            lab = label_from_name(class_names[cls]) if cls < len(class_names) else None
            if lab is None:
                continue
            out.append((path, [(cx - w / 2) * W, (cy - h / 2) * H,
                               (cx + w / 2) * W, (cy + h / 2) * H], lab))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/crops")
    ap.add_argument("--format", choices=["auto", "coco", "yolo"], default="auto")
    ap.add_argument("--yolo-classes", default="ripe,unripe",
                    help="Klassenamn i YOLO-rekkjefølgje, kommaseparert")
    ap.add_argument("--min-size", type=int, default=20, help="Hopp over eple mindre enn dette (px)")
    ap.add_argument("--pad", type=float, default=0.1, help="Ekstra kant rundt boksen (andel)")
    ap.add_argument("--inspect", action="store_true", help="Berre vis kva som vart funne")
    args = ap.parse_args()

    raw = Path(args.raw)
    images = find_images(raw)
    print(f"Fann {len(images)} bilete under {raw}")

    anns = []
    if args.format in ("auto", "coco"):
        anns = load_coco(raw, images)
        if anns:
            print(f"COCO: {len(anns)} annoteringar")
    if not anns and args.format in ("auto", "yolo"):
        anns = load_yolo(raw, images, args.yolo_classes.split(","))
        if anns:
            print(f"YOLO: {len(anns)} annoteringar")
    if not anns:
        raise SystemExit("Fann ingen annoteringar med ripe/unripe-klassar. "
                         "Køyr med --inspect og sjå på filene; kanskje må "
                         "label_from_name() eller --yolo-classes justerast.")

    print("Klassefordeling:", Counter(lab for _, _, lab in anns))
    if args.inspect:
        for p, box, lab in anns[:5]:
            print(" ", p.name, [round(v) for v in box], lab)
        return

    by_image = defaultdict(list)
    for p, box, lab in anns:
        by_image[p].append((box, lab))

    out = Path(args.out)
    rows, skipped = [], 0
    for p, items in sorted(by_image.items()):
        img = Image.open(p).convert("RGB")
        W, H = img.size
        for i, (box, lab) in enumerate(items):
            x0, y0, x1, y1 = box
            if min(x1 - x0, y1 - y0) < args.min_size:
                skipped += 1
                continue
            px, py = (x1 - x0) * args.pad, (y1 - y0) * args.pad
            crop = img.crop((max(0, x0 - px), max(0, y0 - py),
                             min(W, x1 + px), min(H, y1 + py)))
            d = out / lab
            d.mkdir(parents=True, exist_ok=True)
            fname = f"{p.stem}_{i:03d}.png"
            crop.save(d / fname)
            rows.append({"file": f"{lab}/{fname}", "label": lab, "source_image": p.name,
                         "group": group_key(p.stem), "width": crop.width, "height": crop.height})

    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"Hoppa over {skipped} for små eple (< {args.min_size}px)")
    print(f"Lagra {len(rows)} utklipp frå {len({r['group'] for r in rows})} foto i {out}/")
    print("Per klasse:", Counter(r["label"] for r in rows))

if __name__ == "__main__":
    main()
