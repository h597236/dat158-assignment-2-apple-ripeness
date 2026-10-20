"""Lastar ned Fuji Ripeness & Size Dataset frå Kaggle til data/raw/.

    python scripts/download_data.py

Krev at Kaggle-tokenet er sett opp først:  python scripts/setup_kaggle.py

Kjelde: Zhu et al. (2025), "Foundation Model-Based Apple Ripeness and Size
Estimation for Selective Harvesting", arXiv:2502.01850
"""
import os
import subprocess
import sys
from pathlib import Path

DATASET = "zhukeyi1/fuji-ripeness-and-size-dataset"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "raw"
KAGGLE_DIR = Path.home() / ".kaggle"


def check_token():
    """Gi ei forståeleg feilmelding i staden for ein lang traceback frå Kaggle."""
    if os.environ.get("KAGGLE_API_TOKEN") or (KAGGLE_DIR / "kaggle.json").exists():
        return
    candidates = [KAGGLE_DIR / "access_token", KAGGLE_DIR / "access_token.txt"]
    found = [p for p in candidates if p.exists()]
    if not found:
        sys.exit("Fann ikkje noko Kaggle-token. Køyr først:  python scripts/setup_kaggle.py")
    raw = found[0].read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or b"\x00" in raw or raw[:3] == b"\xef\xbb\xbf":
        sys.exit(f"{found[0]} er lagra med feil teiknkoding.\n"
                 "Køyr  python scripts/setup_kaggle.py  for å lagre tokenet på nytt.")


def main():
    check_token()
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Lastar ned {DATASET} til {DEST} (kan ta ei stund) ...")
    # Kaggle-CLI-en ligg i same venv som Python; køyr han via same tolk.
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download",
           "-d", DATASET, "-p", str(DEST), "--unzip"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit("Nedlastinga feila. Sjekk feilmeldinga over, og at tokenet er gyldig "
                 "(python scripts/setup_kaggle.py).")

    images = [p for p in DEST.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    annots = [p for p in DEST.rglob("*") if p.suffix.lower() in {".json", ".xml", ".txt"}]
    print(f"\nFerdig: {len(images)} bilete og {len(annots)} annotasjonsfiler i {DEST}")
    print("Neste steg:  python src/prepare_crops.py --inspect")


if __name__ == "__main__":
    main()
