"""Klipper ut kvart annotert eple frå datasettbileta til data/crops/{ripe,unripe}/
og skriv data/crops/metadata.csv med éi rad per eple.

Fuji-datasettet brukar VIA-format (VGG Image Annotator): éi JSON-fil per split
(train/val/test), der kvart eple har ein boks eller polygon og attributtet
"Ripeness": "ripe"/"unripe". COCO og YOLO er òg støtta, om de byter datasett.

    python src/prepare_crops.py --inspect   # vis kva som blir funne, lagrar ingenting
    python src/prepare_crops.py

Vi brukar ALLE bileta og lagar vår eigen trenings-/testsplitt seinare, gruppert
per foto. Kolonna `group` i metadata.csv seier kva foto/scene eit eple kjem frå:
tiles som _MG_2657_24.png og _MG_2657_25.png er utsnitt av same foto, og
2022_06_17_1_0m_HIGH_P1.png / ..._LOW_P1.png er same scene frå ulik høgd.
Desse må aldri hamne på kvar si side av splitten.
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
    """Kva foto/scene eit bilete høyrer til.

    '_MG_2657_24'               -> '_MG_2657'       (tiles av same foto)
    '2022_06_17_1_0m_HIGH_P1'   -> '2022_06_17_P1'  (same dag og posisjon, ulik høgd/avstand)
    """
    m = re.match(r"^(\d{4}_\d{2}_\d{2})_.*_(P\d+)$", stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return re.sub(r"_\d+$", "", stem)


def find_images(raw: Path):
    return {p.stem: p for p in raw.rglob("*") if p.suffix.lower() in IMG_EXT}


def region_label(attrs: dict):
    """Hent ripe/unripe frå region_attributes, t.d. {"Ripeness": "ripe"}."""
    for k, v in attrs.items():
        if "ripe" in k.lower() and isinstance(v, str):
            return label_from_name(v)
    for v in attrs.values():  # reserve: ein verdi som sjølv heiter ripe/unripe
        if isinstance(v, str) and label_from_name(v):
            return label_from_name(v)
    return None


def shape_to_box(sa: dict):
    if sa.get("name") == "rect":
        return [sa["x"], sa["y"], sa["x"] + sa["width"], sa["y"] + sa["height"]]
    if "all_points_x" in sa:  # polygon / polyline
        xs, ys = sa["all_points_x"], sa["all_points_y"]
        return [min(xs), min(ys), max(xs), max(ys)]
    if sa.get("name") == "circle":
        return [sa["cx"] - sa["r"], sa["cy"] - sa["r"], sa["cx"] + sa["r"], sa["cy"] + sa["r"]]
    if sa.get("name") == "ellipse":
        return [sa["cx"] - sa["rx"], sa["cy"] - sa["ry"], sa["cx"] + sa["rx"], sa["cy"] + sa["ry"]]
    return None


def load_via(raw: Path, images_by_stem):
    """VIA-format: {"<filnamn><storleik>": {"filename", "regions": [...]}, ...}."""
    out, seen, stats = [], set(), Counter()
    for jf in sorted(raw.rglob("*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(d, dict) and "_via_img_metadata" in d:  # VIA-prosjektfil
            d = d["_via_img_metadata"]
        if not isinstance(d, dict) or not d:
            continue
        entries = [v for v in d.values() if isinstance(v, dict) and "regions" in v]
        if not entries:
            continue
        n_before = len(out)
        for e in entries:
            fname = e["filename"]
            if fname in seen:  # same bilete i fleire filer: bruk berre første
                stats["duplikat-bilete"] += 1
                continue
            seen.add(fname)
            # Bileta ligg i <split>/images/ ved sida av JSON-fila; reserve: søk på namn
            path = jf.parent / "images" / fname
            if not path.exists():
                path = images_by_stem.get(Path(fname).stem)
            if path is None:
                stats["manglar bildefil"] += 1
                continue
            regions = e["regions"].values() if isinstance(e["regions"], dict) else e["regions"]
            for r in regions:
                stats[f"form: {r['shape_attributes'].get('name')}"] += 1
                lab = region_label(r.get("region_attributes", {}))
                box = shape_to_box(r["shape_attributes"])
                if lab is None or box is None:
                    stats["utan etikett eller ukjend form"] += 1
                    continue
                out.append((path, box, lab))
        print(f"  {jf.relative_to(raw)}: {len(entries)} bilete, {len(out) - n_before} eple")
    for k, v in sorted(stats.items()):
        print(f"    {k}: {v}")
    return out


def load_coco(raw: Path, images_by_stem):
    """Returnerer liste av (bildesti, [x0,y0,x1,y1], label)."""
    out = []
    for jf in raw.rglob("*.json"):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
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
        for line in tf.read_text(encoding="utf-8").splitlines():
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
    ap.add_argument("--format", choices=["auto", "via", "coco", "yolo"], default="auto")
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
    if args.format in ("auto", "via"):
        anns = load_via(raw, images)
        if anns:
            print(f"VIA: {len(anns)} annoteringar")
    if not anns and args.format in ("auto", "coco"):
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
    groups = {group_key(p.stem) for p, _, _ in anns}
    print(f"Fordelt på {len({p for p, _, _ in anns})} bilete og {len(groups)} foto/scener")
    if args.inspect:
        sides = sorted(min(b[2] - b[0], b[3] - b[1]) for _, b, _ in anns)
        print(f"Eplestorleik (minste side): median {sides[len(sides) // 2]:.0f}px, "
              f"{sum(s < args.min_size for s in sides)} under --min-size {args.min_size}px")
        print("Døme:")
        for p, box, lab in anns[:5]:
            print(f"  {p.name}  {[round(v) for v in box]}  {lab}  gruppe={group_key(p.stem)}")
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

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"Hoppa over {skipped} for små eple (< {args.min_size}px)")
    print(f"Lagra {len(rows)} utklipp frå {len({r['group'] for r in rows})} foto i {out}/")
    print("Per klasse:", Counter(r["label"] for r in rows))

if __name__ == "__main__":
    main()
