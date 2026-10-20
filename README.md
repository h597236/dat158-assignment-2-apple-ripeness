# Er eplet klart for plukking?

ML-basert nettside som vurderer om eit eple er modent nok til å plukkast, ut frå eit bilete.
Sluttprosjekt i DAT158. Tenkt som første steg mot persepsjon for ein epleplukkarrobot.

**Definisjon:** Eit eple reknast som _moden_ når eplet er raudt.
Dette er ein proxy, ikkje ei måling av stivelse, fastleik eller sukker – sjå _Avgrensingar_.

Alt er bygd med verktøya frå kurset: scikit-learn, pandas, matplotlib, Jupyter og Gradio.

## Oppsett

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Steg for steg

```bash
# 1. Last ned data (krev Kaggle API-nøkkel, sjå scripts/download_data.sh)
bash scripts/download_data.sh

# 2. Sjekk at annotasjonane blir lesne rett, deretter klipp ut kvart eple
python src/prepare_crops.py --inspect
python src/prepare_crops.py

# 3. Tren, evaluer og lagre modellen (lagar models/ og reports/)
python src/train.py              # --quick for eit raskare hyperparametersøk

# 4. Start nettsida
python app.py
```

Vil de sjå på data og resultat, opne `notebooks/explore.ipynb`. Den trenar ingenting.

## Struktur

| Fil                         | Innhald                                                         |
| --------------------------- | --------------------------------------------------------------- |
| `scripts/download_data.sh`  | Lastar ned datasettet frå Kaggle                                |
| `src/prepare_crops.py`      | Klipper ut kvart eple og lagar `metadata.csv`                   |
| `src/features.py`           | Gjer eit eplebilete om til trekk (fargetrekk eller pikslar)     |
| `src/dataset.py`            | Lastar utklipp og trekk, og lagar trenings-/testsplitten        |
| `src/train.py`              | Samanlikning, hyperparametersøk, terskel, endeleg test, lagring |
| `notebooks/explore.ipynb`   | Utforsking av data og visning av resultat og figurar            |
| `app.py`                    | Gradio-nettside                                                 |
| `models/apple_model.joblib` | Ferdig trena modell, brukt av nettsida                          |
| `reports/`                  | Resultattabell, `metrics.json` og figurar til rapporten         |

## Metode

`train.py` følgjer same oppbygging som kursnotatbøkene (DAT158-1.3, 1.6, 2.4, 2.10):

1. **Trekk.** Kvart eple blir til ein talvektor på to måtar: 30 handlaga fargetrekk, eller
   bildet skalert ned til 16×16 piksel (som MNIST-dømet i DAT158-1.4).
2. **Trenings-/testsplitt per foto** med `GroupShuffleSplit`. Eit foto inneheld mange eple med
   same lys og bakgrunn, så eple frå same foto må aldri vere i både trening og test.
3. **Baseline** med `DummyClassifier`.
4. **Samanlikning** av logistisk regresjon, `SGDClassifier`, random forest og SVC, med
   `cross_val_score` og `GroupKFold`.
5. **Hyperparametersøk** med `GridSearchCV`.
6. **Val av terskel** med `precision_recall_curve`. For ein robot er det verre å plukke eit umodent
   eple enn å la eit modent henge, så terskelen blir sett slik at precision for «moden» er minst 95 %.
7. **Endeleg test**, éin gong, på det tilbakehaldne testsettet.

## Publisering

Nettsida treng berre `app.py`, `src/features.py`, `models/apple_model.joblib` og `requirements.txt`.
Treninga blir ikkje køyrd ved publisering, så hugs å committe den endelege modellen.

## Samarbeid

- Kvar person jobbar på sin eigen branch og lagar pull request til `main`.
- Tøm output i notatboka før commit, så blir det færre konfliktar.
- `data/` er ikkje i Git; kvar person lastar ned sjølv med `scripts/download_data.sh`.

## Avgrensingar

- Berre éin sort (Fuji). Raud = moden passar ikkje for grøne sortar.
- Etikettane byggjer på farge og fotodato, ikkje målt modning.
- Nettsida tek eitt eple per bilete. Å finne alle epla i eit tre-bilete er neste steg.

## Kjelder

- **Datasett:** Zhu, K. et al. (2025). _Foundation Model-Based Apple Ripeness and Size Estimation
  for Selective Harvesting._ arXiv:2502.01850. Data: https://www.kaggle.com/datasets/zhukeyi1/fuji-ripeness-and-size-dataset
- Datasettet byggjer på mellom anna PFuji-Size (Gené-Mola et al., 2021, CC BY 4.0, berre forsking
  og undervisning) og OpenAccess RGBD Apple (GPL-3.0). Rådata er difor ikkje lagt i dette repoet.
- Kursmateriell DAT158, HVL: https://github.com/HVL-ML/DAT158
- Pedregosa, F. et al. (2011). _Scikit-learn: Machine Learning in Python._ JMLR 12.

## Bruk av AI-verktøy

Claude (Anthropic) vart brukt til å kartleggje tilgjengelege datasett.
