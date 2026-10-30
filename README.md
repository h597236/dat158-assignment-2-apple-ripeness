# Is the apple ready to pick?

ML-based website that judges whether an apple is ripe enough to pick, from a photo.
Final project in DAT158. Intended as a first step towards perception for an apple-picking robot.

**🍎 Try the website: https://dat158-assignment-2-apple-ripeness.onrender.com/** (the first load can take about a minute)

**Definition:** An apple counts as _ripe_ when it is red.
This is a proxy, not a measurement of starch, firmness or sugar – see _Limitations_.

Everything is built with the tools from the course: scikit-learn, pandas, matplotlib, Jupyter and Gradio.

## Setup

Requires Python 3.10 or newer. All commands below are run from the root folder of the repo,
one line at a time.

### 1. Create a virtual environment and install the packages

```
py -m venv .venv                   # Mac/Linux: python3 -m venv .venv
.venv\Scripts\activate             # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

When the environment is active, the terminal prompt starts with `(.venv)`, and `python` works on every machine.

### 2. Kaggle token (first time only)

1. Log in at [kaggle.com/settings](https://www.kaggle.com/settings) → **API Tokens** → **Generate New Token**.
2. Copy the token.
3. Run the script below and paste the token when asked. It is not shown while you paste.

```
python scripts/setup_kaggle.py
```

The token is saved in the `.kaggle` folder in your home folder, outside the repo. Each person needs their own.
Don't create the file yourself in PowerShell: it may get the wrong text encoding, and Kaggle can't read it.

### 3. Download the data

**⚠️ The dataset is about 14 GB.** Since it is unzipped after downloading, you should have
roughly twice as much free space.

```
python scripts/download_data.py
```

## Running

```
python src/prepare_crops.py --inspect
python src/prepare_crops.py
python src/train.py --feature-set simple --model "Gradient boosting (monoton)" --target-precision 0.9
python app.py
```

1. `--inspect` shows how many apples were found and the split between `ripe` and `unripe`, without saving anything.
   Check that both classes are present before continuing.
2. `prepare_crops.py` cuts out each apple into `data/crops/`.
3. `train.py` trains, evaluates and saves the model in `models/` and the results in `reports/`.
   The flags select the features, model and precision target used by the final model (see _Method_).
   Without the flags, the script picks whatever is best in cross-validation, which is not the same.
   Use `--quick` for a faster test run.
4. `app.py` starts the website. Open the link it prints.
   Optional, for analysis and the report:

```
python src/robustness.py
python src/check_labels.py
```

- `robustness.py` tests how well the feature sets handle changed lighting and cameras, and creates `reports/robustness.png`.
- `check_labels.py` shows what share of the apples are labelled ripe, depending on how red they are, and creates
  `reports/label_check.png`.
  To look at the data and results, open `notebooks/explore.ipynb`. It doesn't train anything.

## Deployment

The website runs on Render (free tier): **https://dat158-assignment-2-apple-ripeness.onrender.com/**

Render pulls the code from `main` on GitHub and redeploys on every push.
The free tier goes to sleep after 15 minutes without visitors, so the first load can take about a minute.

The website only needs `app.py`, `src/features.py`, `models/apple_model.joblib` and `requirements.txt`.
Training is not run during deployment, so:

- The trained model must be committed. If you retrain, the new `apple_model.joblib` must be committed.
- `scikit-learn` is pinned in `requirements.txt` to the same version the model was trained with.
  A saved model can't always be loaded with a different version.
- `.python-version` sets the Python version on Render to the same one used locally.
  Setup on Render: _New → Web Service_, choose the repo, _Build Command_ `pip install -r requirements.txt`,
  _Start Command_ `python app.py`, _Instance Type_ Free.

## Structure

| File                        | Contents                                                             |
| --------------------------- | -------------------------------------------------------------------- |
| `scripts/setup_kaggle.py`   | Saves the Kaggle token correctly (once per machine)                  |
| `scripts/download_data.py`  | Downloads the dataset from Kaggle                                    |
| `src/prepare_crops.py`      | Cuts out each apple and creates `metadata.csv`                       |
| `src/features.py`           | Turns an apple image into features (four feature sets, see _Method_) |
| `src/dataset.py`            | Loads crops and features, and makes the train/test split             |
| `src/train.py`              | Comparison, hyperparameter search, threshold, final test, saving     |
| `src/robustness.py`         | Tests the feature sets on images with changed lighting and colours   |
| `src/check_labels.py`       | Shows the relationship between redness and the label "ripe"          |
| `notebooks/explore.ipynb`   | Data exploration and display of results and figures                  |
| `app.py`                    | Gradio website: click on apples in a tree photo to get an assessment |
| `models/apple_model.joblib` | Trained model, used by the website                                   |
| `reports/`                  | Result tables, `metrics.json` and figures for the report             |
| `.python-version`           | The Python version Render should use                                 |

## Method

`train.py` follows the same structure as the course notebooks (DAT158-1.3, 1.6, 2.4, 2.10):

1. **Features.** Each apple is turned into a vector of numbers in one of four ways: 30 hand-crafted colour
   features (`color`), 24 features describing only hue (`hue`), 6 summary hue features without the histogram
   (`simple`), or the image scaled down to 16×16 pixels (`pixels`, like the MNIST example in DAT158-1.4).
2. **Train/test split per photo** with `StratifiedGroupKFold`. A photo contains many apples with the
   same lighting and background, so apples from the same photo must never be in both training and test.
   Stratification gives training and test roughly the same share of ripe apples. The dataset has its own
   train/val/test split, but there, crops (tiles) of the same photo, and the same scene taken from different
   heights, can end up in different parts. We therefore combine all images and make our own split.
3. **Baseline** with `DummyClassifier`.
4. **Comparison** of logistic regression, `SGDClassifier`, random forest, SVC and
   `HistGradientBoostingClassifier`, with `cross_val_score` and `StratifiedGroupKFold`. Only about 14 %
   of the apples are ripe, so the models use `class_weight="balanced"`, and F1 for "ripe" is the main
   metric instead of accuracy.
5. **Hyperparameter search** with `GridSearchCV`.
6. **Threshold selection** with `precision_recall_curve`. For a robot, picking an unripe apple is worse
   than leaving a ripe one on the tree, so the threshold is set so that precision for "ripe" is at least 90 %.
   95 % was tried first, but then the model found only about a third of the ripe apples.
7. **Final test**, once, on the held-out test set.
8. **Robustness and monotonic model.** On images from the internet, the model was uncertain even about
   apples that were almost entirely red. `check_labels.py` showed that a typical ripe apple in the dataset
   is only about 40 % red, and that almost no apples are more than 80 % red. Random forest cannot
   extrapolate to apples that are redder than the training data. The final model is therefore
   `HistGradientBoostingClassifier` with `monotonic_cst` on the `simple` feature set: more red never makes
   an apple less ripe, and more green never makes it more ripe. It performs about as well in
   cross-validation, but also gives sensible answers for apples outside the training data.

## The website

The model is trained on crops where a single apple fills the image. The website therefore lets the user
click on apples in a photo of an apple tree. The app cuts out a square around the click, shows the crop
("What the model sees"), and marks the apple with a red frame for ripe and green for unripe. A slider
controls the size of the crop; the box should cover one apple with a little foliage around it.

## Limitations

- Only one variety (Fuji). Red = ripe does not apply to green varieties.
- The labels are based on colour and photo date, not measured ripeness.
- The user has to click on the apples and set the size themselves. Finding the apples automatically
  requires an object detector (deep learning), which is the next step towards a picking robot.

## Sources

- **Dataset:** Zhu, K. et al. (2025). _Foundation Model-Based Apple Ripeness and Size Estimation
  for Selective Harvesting._ arXiv:2502.01850. Data: https://www.kaggle.com/datasets/zhukeyi1/fuji-ripeness-and-size-dataset
  (licence on Kaggle: GPL-3.0). 4027 images in VIA format, with boxes and a `Ripeness` label per apple.
- The dataset builds on, among others, PFuji-Size (Gené-Mola et al., 2021, CC BY 4.0, research and
  teaching only) and OpenAccess RGBD Apple (GPL-3.0). The raw data is therefore not included in this repo.
- Pedregosa, F. et al. (2011). _Scikit-learn: Machine Learning in Python._ JMLR 12.

## Use of AI tools

Claude (Anthropic) was used to survey available datasets, and for deploying the website, as we were not familiar with Gradio and Render,
we also wrote the readme in Norwegian initially and used AI to translate it to English.
Claude was also used to help set up the model selection. It suggested which models from the
course to compare, how to compare them fairly (cross-validation grouped by photo, F1 instead of
accuracy because of the class imbalance), and how to choose the decision threshold. The choice
itself was based on the results: the cross-validation scores from `train.py`, and our own testing
of the website on images from outside the dataset.
AI was used to generate a first draft of our readme file for Github to help with documentation.
