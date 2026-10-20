"""Lagrar Kaggle API-tokenet ditt rett, på Windows, Mac og Linux.

    python scripts/setup_kaggle.py

1. Gå til https://www.kaggle.com/settings -> API Tokens -> "Generate New Token"
2. Kopier tokenet
3. Køyr dette skriptet og lim det inn når du blir spurd (det blir ikkje vist)

Tokenet blir lagra i ~/.kaggle/access_token (utanfor repoet, så det aldri hamnar
på GitHub). Skriptet skriv fila som rein tekst; å lage henne sjølv i PowerShell
kan gi feil teiknkoding (UTF-16), som Kaggle ikkje kan lese.
"""
import getpass
import os
import sys
from pathlib import Path

KAGGLE_DIR = Path.home() / ".kaggle"
TOKEN_FILE = KAGGLE_DIR / "access_token"


def main():
    print("Lim inn Kaggle API-tokenet (frå kaggle.com/settings -> API Tokens).")
    token = getpass.getpass("Token: ").strip().strip('"').strip("'")
    if not token:
        sys.exit("Ingen token skrive inn. Ingenting vart lagra.")
    if any(c.isspace() for c in token) or not token.isprintable():
        sys.exit("Tokenet inneheld mellomrom eller rare teikn. Kopier det på nytt frå Kaggle.")

    KAGGLE_DIR.mkdir(exist_ok=True)
    # Fjern gamle variantar så det berre finst éin token-fil
    for old in (KAGGLE_DIR / "access_token.txt",):
        if old.exists():
            old.unlink()
            print(f"Sletta gammal fil: {old}")

    TOKEN_FILE.write_text(token, encoding="ascii")  # rein tekst, ingen BOM, ingen linjeskift
    if os.name != "nt":
        TOKEN_FILE.chmod(0o600)  # berre du kan lese fila
    print(f"Lagra: {TOKEN_FILE}")
    print("Test med:  kaggle datasets list -s fuji")


if __name__ == "__main__":
    main()
