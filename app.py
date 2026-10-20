"""Gradio-nettside: last opp eit bilete av eit epletre og klikk på epla.

Modellen er trena på utklipp der eitt eple fyller biletet. Difor klipper appen ut
eit kvadrat rundt kvart klikk, slik at modellen får sjå det same som under
treninga, og merkar resultatet på biletet. Same arbeidsflyt som ein plukkerobot:
først finne eplet, så vurdere om det er modent.

    python app.py
"""
import sys
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from features import COLOR_FEATURE_NAMES, color_features, extract, feature_names  # noqa: E402

bundle = joblib.load(ROOT / "models" / "apple_model.joblib")
model = bundle["model"]
FEATURE_SET = bundle["feature_set"]
THRESHOLD = bundle["threshold"]

# Treningsutklippa har 10 % ekstra kant på kvar side av eplet (prepare_crops.py --pad 0.1)
PAD = 0.1
RIPE_COLOR = (220, 40, 40)
UNRIPE_COLOR = (60, 180, 60)
TABLE_COLUMNS = ["Nr", "Vurdering", "P(moden)"]


def predict(crop: Image.Image) -> float:
    x = pd.DataFrame([extract(crop, FEATURE_SET)], columns=feature_names(FEATURE_SET))
    return float(model.predict_proba(x)[0, 1])


def crop_around(img: Image.Image, cx: int, cy: int, apple_px: int) -> tuple[Image.Image, tuple]:
    """Kvadratisk utklipp med sentrum i klikket, same proporsjonar som treningsdataa."""
    half = apple_px * (1 + 2 * PAD) / 2
    box = (max(0, int(cx - half)), max(0, int(cy - half)),
           min(img.width, int(cx + half)), min(img.height, int(cy + half)))
    return img.crop(box), box


def apple_size_px(img: Image.Image, size_pct: float) -> int:
    return max(10, int(min(img.size) * size_pct / 100))


def verdict(p: float) -> str:
    return "Moden" if p >= THRESHOLD else "Umoden"


def draw(img: Image.Image, marks: list) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    width = max(2, min(img.size) // 200)
    try:
        font = ImageFont.load_default(size=max(14, min(img.size) // 35))
    except TypeError:  # eldre Pillow
        font = ImageFont.load_default()
    for i, m in enumerate(marks, start=1):
        color = RIPE_COLOR if m["ripe"] else UNRIPE_COLOR
        x0, y0, x1, y1 = m["box"]
        d.rectangle(m["box"], outline=color, width=width)
        label = f"{i}"
        tb = d.textbbox((x0, y0), label, font=font)
        d.rectangle((x0, y0 - (tb[3] - tb[1]) - 6, x0 + (tb[2] - tb[0]) + 8, y0), fill=color)
        d.text((x0 + 4, y0 - (tb[3] - tb[1]) - 4), label, fill="white", font=font)
    return out


def table(marks: list) -> pd.DataFrame:
    return pd.DataFrame([{"Nr": i, "Vurdering": verdict(m["p"]), "P(moden)": f"{m['p']:.0%}"}
                         for i, m in enumerate(marks, start=1)], columns=TABLE_COLUMNS)


def summary(marks: list) -> str:
    if not marks:
        return "Klikk på eit eple i biletet."
    n_ripe = sum(m["ripe"] for m in marks)
    last = marks[-1]
    red = color_features(last["crop"])[COLOR_FEATURE_NAMES.index("red_fraction")]
    return (f"### Eple {len(marks)}: {'🍎 Moden – klar for plukking' if last['ripe'] else '🍏 Umoden – la det henge litt til'}\n"
            f"Sannsyn for moden: **{last['p']:.0%}** (må vere minst {THRESHOLD:.0%} for «moden»)  \n"
            f"Andel raude piksel i utklippet: {red:.0%}\n\n"
            f"**Totalt: {n_ripe} av {len(marks)} vurderte eple er klare for plukking.**")


# ------------------------------------------------------------------ hendingar
def on_upload(img):
    if img is None:
        return None, [], None, table([]), "Last opp eit bilete av eit epletre."
    img = img.convert("RGB")
    return img, [], img, table([]), summary([])


def on_click(original, marks, size_pct, evt: gr.SelectData):
    if original is None:
        return gr.skip(), marks, gr.skip(), gr.skip(), "Last opp eit bilete først."
    cx, cy = evt.index
    crop, box = crop_around(original, cx, cy, apple_size_px(original, size_pct))
    p = predict(crop)
    marks = marks + [{"box": box, "p": p, "ripe": p >= THRESHOLD, "crop": crop}]
    return draw(original, marks), marks, crop, table(marks), summary(marks)


def on_undo(original, marks):
    if original is None:
        return gr.skip(), marks, gr.skip(), gr.skip(), gr.skip()
    marks = marks[:-1]
    last_crop = marks[-1]["crop"] if marks else None
    return draw(original, marks), marks, last_crop, table(marks), summary(marks)


def on_reset(original):
    return original, [], None, table([]), summary([]) if original is not None else "Last opp eit bilete av eit epletre."


def on_whole(original, marks):
    """For nærbilete der eitt eple fyller heile biletet."""
    if original is None:
        return gr.skip(), marks, gr.skip(), gr.skip(), "Last opp eit bilete først."
    p = predict(original)
    box = (0, 0, original.width - 1, original.height - 1)
    marks = marks + [{"box": box, "p": p, "ripe": p >= THRESHOLD, "crop": original}]
    return draw(original, marks), marks, original, table(marks), summary(marks)


# ------------------------------------------------------------------- grensesnitt
with gr.Blocks(title="Er eplet klart for plukking?") as demo:
    gr.Markdown(
        "# Er eplet klart for plukking?\n"
        "Last opp eit bilete av eit epletre og **klikk på eit eple**. Appen klipper ut eplet, "
        "vurderer om det er modent, og merkar det på biletet: "
        "<span style='color:rgb(220,40,40)'>■ raud = moden</span>, "
        "<span style='color:rgb(60,180,60)'>■ grøn = umoden</span>.  \n"
        "Juster eplestorleiken til ruta dekkjer heile eplet, men lite blad rundt."
    )
    original = gr.State(None)
    marks = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            image = gr.Image(type="pil", label="Epletre (klikk på eit eple)",
                             sources=["upload", "clipboard"], height=560)
            size = gr.Slider(3, 60, value=15, step=1, label="Eplestorleik (% av biletet)")
            with gr.Row():
                undo_btn = gr.Button("Angre siste")
                reset_btn = gr.Button("Fjern alle merke")
                whole_btn = gr.Button("Vurder heile biletet (nærbilete)")
        with gr.Column(scale=2):
            result = gr.Markdown("Last opp eit bilete av eit epletre.")
            crop_view = gr.Image(label="Dette ser modellen", interactive=False, height=220)
            results = gr.Dataframe(value=table([]), headers=TABLE_COLUMNS, label="Vurderte eple",
                                   interactive=False)

    gr.Markdown(
        f"Modell: {bundle['model_name']} på trekksettet «{FEATURE_SET}», trena på "
        "Fuji Ripeness & Size Dataset (Zhu et al., 2025). «Moden» tyder her at overflata "
        f"hovudsakleg er raud. Terskelen ({THRESHOLD:.0%}) er sett høgt for å unngå å plukke umodne eple."
    )

    outputs = [image, marks, crop_view, results, result]
    image.upload(on_upload, inputs=image, outputs=[original, marks, crop_view, results, result])
    image.clear(lambda: (None, [], None, table([]), "Last opp eit bilete av eit epletre."),
                outputs=[original, marks, crop_view, results, result])
    image.select(on_click, inputs=[original, marks, size], outputs=outputs)
    undo_btn.click(on_undo, inputs=[original, marks], outputs=outputs)
    reset_btn.click(on_reset, inputs=original, outputs=outputs)
    whole_btn.click(on_whole, inputs=[original, marks], outputs=outputs)

if __name__ == "__main__":
    demo.launch()
