# Er eplet klart for plukking?

ML-basert nettside som vurderer om eit eple er modent nok til å plukkast, ut frå eit bilete.
Sluttprosjekt i DAT158. Tenkt som første steg mot persepsjon for ein epleplukkarrobot.

**Definisjon:** Eit eple reknast som _moden_ når eplet er raudt.
Dette er ein proxy, ikkje ei måling av stivelse, fastleik eller sukker – sjå _Avgrensingar_.

Alt er bygd med verktøya frå kurset: scikit-learn, pandas, matplotlib, Jupyter og Gradio.

## Oppsett

Krev Python 3.10 eller nyare. Alle kommandoane under blir køyrde frå rotmappa i repoet,
éin linje om gongen.

### 1. Lag eit virtuelt miljø og installer pakkane

```bash
py -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Kaggle-token (berre første gong)

1. Logg inn på [kaggle.com/settings](https://www.kaggle.com/settings) → **API Tokens** → **Generate New Token**.
2. Kopier tokenet.
3. Køyr skriptet under og lim inn tokenet når du blir spurd. Det blir ikkje vist når du limer inn.

```
python scripts/setup_kaggle.py
```

Tokenet blir lagra i `.kaggle`-mappa i heimemappa di, utanfor repoet. Kvar person treng sitt eige.
Lag ikkje fila sjølv i PowerShell: då kan ho få feil teiknkoding, og Kaggle klarer ikkje lese ho.

### 3. Last ned data

**⚠️ Datasettet er om lag 14 GB.** Sidan det blir pakka ut etter nedlasting, bør du ha
rundt dobbelt så mykje ledig plass

```
python scripts/download_data.py
```

## Køyring

```
python src/prepare_crops.py --inspect
python src/prepare_crops.py
python src/train.py
python app.py
```

1. `--inspect` viser kor mange eple som vart funne og fordelinga mellom `ripe` og `unripe`, utan å lagre noko.
   Sjå at begge klassane er med før de går vidare.
2. `prepare_crops.py` klipper ut kvart eple til `data/crops/`.
3. `train.py` trenar, evaluerer og lagrar modellen i `models/` og resultata i `reports/`.
   Bruk `--quick` for ei raskare prøvekøyring.
4. `app.py` startar nettsida. Opne lenkja som blir skriven ut.

Vil de sjå på data og resultat, opne `notebooks/explore.ipynb`. Ho trenar ingenting.

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
7. **Endeleg test**, ein gong, på det tilbakehaldne testsettet.

## Publisering

Nettsida treng berre `app.py`, `src/features.py`, `models/apple_model.joblib` og `requirements.txt`.
Treninga blir ikkje køyrd ved publisering, så hugs å committe den endelege modellen.

## Avgrensingar

- Berre éin sort (Fuji). Raud = moden passar ikkje for grøne sortar.
- Etikettane byggjer på farge og fotodato, ikkje målt modning.
- Nettsida tek eitt eple per bilete. Å finne alle epla i eit tre-bilete er neste steg.

## Kjelder

- **Datasett:** Zhu, K. et al. (2025). _Foundation Model-Based Apple Ripeness and Size Estimation
  for Selective Harvesting._ arXiv:2502.01850. Data: https://www.kaggle.com/datasets/zhukeyi1/fuji-ripeness-and-size-dataset
- Datasettet byggjer på mellom anna PFuji-Size (Gené-Mola et al., 2021, CC BY 4.0, berre forsking
  og undervisning) og OpenAccess RGBD Apple (GPL-3.0). Rådata er difor ikkje lagt i dette repoet.
- Pedregosa, F. et al. (2011). _Scikit-learn: Machine Learning in Python._ JMLR 12.

## Bruk av AI-verktøy

Claude (Anthropic) vart brukt til å kartleggje tilgjengelege datasett.
