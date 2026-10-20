"""Gjer om eit eplebilete til ein talvektor som scikit-learn kan bruke.

Tre sett med trekk, slik at vi kan samanlikne dei (feature engineering, jf. DAT158-2.9):

* color_features: 30 handlaga fargetrekk (fargetonehistogram, raud-/grønandel,
  middel og spreiing i HSV og RGB). Få, tolkbare trekk.
* hue_features: 24 trekk som berre handlar om fargetone (andel raudt, gult og
  grønt, histogram over fargetone). Toler anna kamera og lys betre, sjå
  src/robustness.py.
* pixel_features: bildet skalert ned til 16x16 og flata ut til 768 tal,
  same idé som MNIST-dømet i DAT158-1.4.

Same funksjonar blir brukt i notatboka og i app.py, så nettsida reknar ut
trekka nøyaktig slik modellen vart trena på.
"""
import numpy as np
from PIL import Image

HUE_BINS = 18
PIXEL_SIZE = 16

# PIL lagrar HSV med alle kanalar i 0-255 (fargetone 0-255 tilsvarer 0-360 grader).
RED_HUE = (15, 230)      # raud: hue < 15 eller hue > 230 (ca. 0-21 og 325-360 grader)
GREEN_HUE = (45, 115)    # grøn/gulgrøn: ca. 65-160 grader

COLOR_FEATURE_NAMES = (
    [f"hue_bin_{i:02d}" for i in range(HUE_BINS)]
    + ["red_fraction", "green_fraction",
       "sat_mean", "sat_std", "val_mean", "val_std",
       "r_mean", "g_mean", "b_mean", "r_minus_g", "r_std", "g_std"]
)


def _center(img: Image.Image, frac: float = 0.8) -> Image.Image:
    """Kutt bort kanten av utklippet, der det ofte er blad og himmel."""
    w, h = img.size
    m = (1 - frac) / 2
    return img.crop((int(w * m), int(h * m), int(w * (1 - m)), int(h * (1 - m))))


def color_features(img: Image.Image) -> np.ndarray:
    img = _center(img.convert("RGB")).resize((64, 64))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Berre piksel med tydeleg farge (ikkje grå himmel eller mørke skuggar)
    colored = (sat > 60) & (val > 40)
    n = max(colored.sum(), 1)
    hist = np.histogram(hue[colored], bins=HUE_BINS, range=(0, 256))[0] / n
    red = (colored & ((hue < RED_HUE[0]) | (hue > RED_HUE[1]))).sum() / n
    green = (colored & (hue >= GREEN_HUE[0]) & (hue <= GREEN_HUE[1])).sum() / n

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rest = [red, green,
            sat.mean() / 255, sat.std() / 255, val.mean() / 255, val.std() / 255,
            r.mean(), g.mean(), b.mean(), (r - g).mean(), r.std(), g.std()]
    return np.concatenate([hist, rest]).astype(np.float32)


def pixel_features(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((PIXEL_SIZE, PIXEL_SIZE))
    return (np.asarray(img, dtype=np.float32) / 255.0).ravel()


def pixel_feature_names():
    return [f"px_{y}_{x}_{c}" for y in range(PIXEL_SIZE) for x in range(PIXEL_SIZE) for c in "rgb"]


YELLOW_HUE = (15, 45)    # gul/oransje: ca. 21-63 grader

HUE_FEATURE_NAMES = (
    [f"hue_bin_{i:02d}" for i in range(HUE_BINS)]
    + ["red_fraction", "yellow_fraction", "green_fraction",
       "red_minus_green", "chroma_r", "chroma_g"]
)


def hue_features(img: Image.Image) -> np.ndarray:
    """Berre trekk som handlar om KVA farge eplet har, ikkje kor lyst eller sterkt biletet er.

    color_features tek òg med lysstyrke, metning og rå RGB-snitt. Dei skildrar mykje
    kamera og lys, og gjer modellen usikker på bilete frå andre kjelder enn Fuji-
    datasettet. Her er alt anten fargetone (uavhengig av lysstyrke) eller
    kromatisitet r/(r+g+b) (fargefordeling utan lysstyrke).
    """
    img = _center(img.convert("RGB")).resize((64, 64))
    rgb = np.asarray(img, dtype=np.float32)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Lågare krav til metning enn color_features, så matte eller blasse bilete
    # framleis blir rekna med
    colored = (sat > 40) & (val > 30)
    n = max(colored.sum(), 1)
    hist = np.histogram(hue[colored], bins=HUE_BINS, range=(0, 256))[0] / n
    red = (colored & ((hue < RED_HUE[0]) | (hue > RED_HUE[1]))).sum() / n
    yellow = (colored & (hue >= YELLOW_HUE[0]) & (hue < YELLOW_HUE[1])).sum() / n
    green = (colored & (hue >= GREEN_HUE[0]) & (hue <= GREEN_HUE[1])).sum() / n

    total = rgb.sum(axis=2, keepdims=True) + 1e-6
    chroma = (rgb / total)[colored] if colored.any() else np.full((1, 3), 1 / 3)
    rest = [red, yellow, green, red - green, chroma[:, 0].mean(), chroma[:, 1].mean()]
    return np.concatenate([hist, rest]).astype(np.float32)


# Berre dei 6 samandragstrekka, utan histogrammet. Med så få trekk kan alle få ei
# retning i ein monoton modell (sjå MONOTONIC i train.py), og modellen kan ikkje
# «snike seg rundt» regelen via enkeltbitar i histogrammet.
SIMPLE_FEATURE_NAMES = HUE_FEATURE_NAMES[HUE_BINS:]


def simple_features(img: Image.Image) -> np.ndarray:
    return hue_features(img)[HUE_BINS:]


def extract(img: Image.Image, feature_set: str) -> np.ndarray:
    if feature_set == "simple":
        return simple_features(img)
    if feature_set == "color":
        return color_features(img)
    if feature_set == "hue":
        return hue_features(img)
    if feature_set == "pixels":
        return pixel_features(img)
    if feature_set == "color+pixels":
        return np.concatenate([color_features(img), pixel_features(img)])
    raise ValueError(f"Ukjent trekksett: {feature_set}")


def feature_names(feature_set: str):
    return {"color": COLOR_FEATURE_NAMES,
            "hue": HUE_FEATURE_NAMES,
            "simple": SIMPLE_FEATURE_NAMES,
            "pixels": pixel_feature_names(),
            "color+pixels": COLOR_FEATURE_NAMES + pixel_feature_names()}[feature_set]
