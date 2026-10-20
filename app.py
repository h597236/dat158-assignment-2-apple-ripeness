"""Gradio-nettside: last opp eit bilete av eitt eple og få modningsvurdering.

Køyr notatboka (notebooks/eplemodning.ipynb) først, så models/apple_model.joblib finst.
    python app.py
"""
import sys
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from features import COLOR_FEATURE_NAMES, color_features, extract, feature_names  # noqa: E402

bundle = joblib.load(ROOT / "models" / "apple_model.joblib")
model = bundle["model"]
FEATURE_SET = bundle["feature_set"]
THRESHOLD = bundle["threshold"]


def predict_ripeness(image):
    if image is None:
        return "Last opp eit bilete først.", {}, ""

    x = pd.DataFrame([extract(image, FEATURE_SET)], columns=feature_names(FEATURE_SET))
    p_ripe = float(model.predict_proba(x)[0, 1])

    verdict = ("🍎 Moden – klar for plukking" if p_ripe >= THRESHOLD
               else "🍏 Umoden – la det henge litt til")
    probs = {"Moden": p_ripe, "Umoden": 1 - p_ripe}

    red = color_features(image)[COLOR_FEATURE_NAMES.index("red_fraction")]
    details = (f"Andel raude piksel: **{red:.0%}**  \n"
               f"Terskel for «moden»: {THRESHOLD:.2f} (sett høgt for å unngå å plukke umodne eple)  \n"
               f"Modell: {bundle['model_name']} på trekksettet «{FEATURE_SET}»")
    return verdict, probs, details


iface = gr.Interface(
    fn=predict_ripeness,
    inputs=gr.Image(type="pil", label="Bilete av eitt eple"),
    outputs=[
        gr.Textbox(label="Vurdering"),
        gr.Label(label="Sannsyn"),
        gr.Markdown(),
    ],
    title="Er eplet klart for plukking?",
    description="""Last opp eit nærbilete av eitt eple, gjerne på treet.
    Data: Fuji Ripeness & Size Dataset (Zhu et al., 2025). «Moden» tyder her at overflata hovudsakleg er raud.""",
    examples=[str(p) for p in sorted((ROOT / "examples").glob("*"))[:6]] or None,
    flagging_mode="never",
)

if __name__ == "__main__":
    iface.launch()
