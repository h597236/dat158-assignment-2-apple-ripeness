Readme utan kjelder · MD

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

```
py -m venv .venv                   # Mac/Linux: python3 -m venv .venv
.venv\Scripts\activate             # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Når miljøet er aktivt, startar terminalen med `(.venv)`, og `python` fungerer på alle maskinar.

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
rundt dobbelt så mykje ledig plass.

```
python scripts/download_data.py
```

## Køyring

```
python src/prepare_crops.py --inspect
python src/prepare_crops.py
python src/train.py --feature-set simple --model "Gradient boosting (monoton)" --target-precision 0.9
python app.py
```

1. `--inspect` viser kor mange eple som vart funne og fordelinga mellom `ripe` og `unripe`, utan å lagre noko.
   Sjå at begge klassane er med før de går vidare.
2. `prepare_crops.py` klipper ut kvart eple til `data/crops/`.
3. `train.py` trenar, evaluerer og lagrar modellen i `models/` og resultata i `reports/`.
   Flagga vel trekka, modellen og precision-kravet som den endelege modellen brukar (sjå _Metode_).
   Utan flagga vel skriptet det som er best i kryssvalideringa, som ikkje er det same.
   Bruk `--quick` for ei raskare prøvekøyring.
4. `app.py` startar nettsida. Opne lenkja som blir skriven ut.
   Valfritt, til analyse og rapport:

```
python src/robustness.py
python src/check_labels.py
```

- `robustness.py` testar kor godt trekksetta toler endra lys og kamera, og lagar `reports/robustness.png`.
- `check_labels.py` viser kor stor del av epla som er merkte modne, etter kor raude dei er, og lagar
  `reports/label_check.png`.
  Vil de sjå på data og resultat, opne `notebooks/explore.ipynb`. Ho trenar ingenting.

## Struktur

| Fil                         | Innhald                                                            |
| --------------------------- | ------------------------------------------------------------------ |
| `scripts/setup_kaggle.py`   | Lagrar Kaggle-tokenet rett (éin gong per maskin)                   |
| `scripts/download_data.py`  | Lastar ned datasettet frå Kaggle                                   |
| `src/prepare_crops.py`      | Klipper ut kvart eple og lagar `metadata.csv`                      |
| `src/features.py`           | Gjer eit eplebilete om til trekk (fire trekksett, sjå _Metode_)    |
| `src/dataset.py`            | Lastar utklipp og trekk, og lagar trenings-/testsplitten           |
| `src/train.py`              | Samanlikning, hyperparametersøk, terskel, endeleg test, lagring    |
| `src/robustness.py`         | Testar trekksetta på bilete med endra lys og fargar                |
| `src/check_labels.py`       | Viser samanhengen mellom raudandel og etiketten «moden»            |
| `notebooks/explore.ipynb`   | Utforsking av data og visning av resultat og figurar               |
| `app.py`                    | Gradio-nettside: klikk på eple i eit tre-bilete for å få vurdering |
| `models/apple_model.joblib` | Ferdig trena modell, brukt av nettsida                             |
| `reports/`                  | Resultattabellar, `metrics.json` og figurar til rapporten          |

## Metode

`train.py` følgjer same oppbygging som kursnotatbøkene (DAT158-1.3, 1.6, 2.4, 2.10):

1. **Trekk.** Kvart eple blir til ein talvektor på fire måtar: 30 handlaga fargetrekk (`color`),
   24 trekk som berre handlar om fargetone (`hue`), 6 samandragstrekk av fargetone utan histogram
   (`simple`), eller bildet skalert ned til 16×16 piksel (`pixels`, som MNIST-dømet i DAT158-1.4).
2. **Trenings-/testsplitt per foto** med `StratifiedGroupKFold`. Eit foto inneheld mange eple med
   same lys og bakgrunn, så eple frå same foto må aldri vere i både trening og test. Stratifiseringa
   gjer at trening og test får om lag same andel modne eple. Datasettet har sin eigen
   train/val/test-inndeling, men der kan utsnitt (tiles) av same foto og same scene teken frå ulik
   høgd hamne i ulike delar. Vi slår difor saman alle bileta og lagar vår eigen splitt.
3. **Baseline** med `DummyClassifier`.
4. **Samanlikning** av logistisk regresjon, `SGDClassifier`, random forest, SVC og
   `HistGradientBoostingClassifier`, med `cross_val_score` og `StratifiedGroupKFold`. Berre ca. 14 %
   av epla er modne, så modellane brukar `class_weight="balanced"`, og F1 for «moden» er hovudmålet
   i staden for accuracy.
5. **Hyperparametersøk** med `GridSearchCV`.
6. **Val av terskel** med `precision_recall_curve`. For ein robot er det verre å plukke eit umodent
   eple enn å la eit modent henge, så terskelen blir sett slik at precision for «moden» er minst 90 %.
   95 % vart prøvd først, men då fann modellen berre om lag ein tredel av dei modne epla.
7. **Endeleg test**, éin gong, på det tilbakehaldne testsettet.
8. **Robustheit og monoton modell.** På bilete frå nettet var modellen usikker, sjølv på eple som
   var nesten heilt raude. `check_labels.py` viste at eit typisk modent eple i datasettet berre er
   ca. 40 % raudt, og at nesten ingen eple er over 80 % raude. Random forest kan ikkje ekstrapolere
   til eple som er raudare enn treningsdataa. Den endelege modellen er difor
   `HistGradientBoostingClassifier` med `monotonic_cst` på trekksettet `simple`: meir raudt gjer
   aldri eplet mindre modent, og meir grønt gjer det aldri meir modent. Han er omtrent like god i
   kryssvalideringa, men gir fornuftige svar òg for eple utanfor treningsdataa.

## Nettsida

Modellen er trena på utklipp der eitt eple fyller biletet. Nettsida lèt difor brukaren klikke på
eple i eit bilete av eit epletre. Appen klipper ut eit kvadrat rundt klikket, viser utklippet
(«Dette ser modellen»), og merkar eplet med raud ramme for moden og grøn for umoden. Ein glidebrytar
styrer storleiken på utklippet; ruta bør dekkje eitt eple med litt blad rundt.

## Publisering

Nettsida treng berre `app.py`, `src/features.py`, `models/apple_model.joblib` og `requirements.txt`.
Treninga blir ikkje køyrd ved publisering, så hugs å committe den endelege modellen.

## Avgrensingar

- Berre éin sort (Fuji). Raud = moden passar ikkje for grøne sortar.
- Etikettane byggjer på farge og fotodato, ikkje målt modning.
- Brukaren må sjølv klikke på epla og stille inn storleiken. Å finne epla automatisk krev ein
  objektdetektor (djuplæring), som er neste steg mot ein plukkerobot.

## Kjelder

- **Datasett:** Zhu, K. et al. (2025). _Foundation Model-Based Apple Ripeness and Size Estimation
  for Selective Harvesting._ arXiv:2502.01850. Data: https://www.kaggle.com/datasets/zhukeyi1/fuji-ripeness-and-size-dataset
  (lisens på Kaggle: GPL-3.0). 4027 bilete i VIA-format, med boksar og etiketten `Ripeness` per eple.
- Datasettet byggjer på mellom anna PFuji-Size (Gené-Mola et al., 2021, CC BY 4.0, berre forsking
  og undervisning) og OpenAccess RGBD Apple (GPL-3.0). Rådata er difor ikkje lagt i dette repoet.
- Pedregosa, F. et al. (2011). _Scikit-learn: Machine Learning in Python._ JMLR 12.

## Bruk av AI-verktøy

Claude (Anthropic) vart brukt til å kartleggje tilgjengelege datasett.
