# Skin cancer screening — front end + backend

Serves the Phase II binary (benign vs malignant) model from `notebook44444`.

## Files you need in this folder

| File | Where it comes from |
|---|---|
| `index.html` | here |
| `app.py` | here |
| `requirements.txt` | here |
| `ckpt/best.keras` | notebook output — see below |
| `decision_threshold.json` | notebook output |

The model checkpoint is **not** in this repo — it's several hundred MB, well past
GitHub's limit, and `.gitignore` excludes `ckpt/` on purpose. If you're cloning this
repo fresh, get it from the notebook's Kaggle page:

```powershell
kaggle kernels output rnihal/notebooke848599116 -p .
```

That pulls `ckpt/best.keras`, `decision_threshold.json`, and the rest of the run's
output directly into this folder. Requires the Kaggle CLI authenticated
(`kaggle auth login`). Alternatively, download `best.keras` by hand from the
notebook's **Output** tab and place it at `ckpt/best.keras`.

`app.py` defaults to `skin_cancer_v3.keras` in `MODEL_PATH`. Since this deployment
uses the checkpoint instead, either rename the file to `skin_cancer_v3.keras`, or set
the environment variable before running:

```powershell
$env:MODEL_PATH="ckpt\best.keras"
```

## Run

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:MODEL_PATH="ckpt\best.keras"
python app.py
```

Then open http://127.0.0.1:5000

## What the backend does

It reproduces the notebook's inference path exactly:

1. `crop_vignette` -> resize to 384 -> `adaptive_hair_removal` -> `shades_of_gray`
   -> `apply_clahe`, all copied verbatim from section 9.
2. BGR to RGB, divide by 255. The model's own `Rescaling(255.0)` layer converts back.
3. Metadata encoded exactly as `encode_meta`: age/100 clipped to [0,1] at index 0,
   sex one-hot at 1-3, site one-hot at 4-13, explicit `unknown` slot.
4. Optional 8x TTA (identity, both flips, three rotations, two flip+rotate),
   averaged over the binary head.
5. Compare against the frozen threshold from `decision_threshold.json`.

The auxiliary 8-class head is returned by `/predict` as supporting evidence, but the
front end only surfaces the benign/malignant call — the same binary head the notebook
uses for its own decision.

Grad-CAM uses the explicit head-replay pattern from section 17, since auto-detection
cannot work through the metadata branch.

## Two things that will silently ruin the results

**Missing `decision_threshold.json`.** The backend falls back to 0.5, which is not
your validation-fitted operating point. The UI shows a red warning when this happens
and prints the threshold source under the results.

**Editing the preprocessing functions in `app.py`.** They are duplicated from the
notebook on purpose. If they drift, the served predictions are no longer the ones
you evaluated, and your reported metrics stop describing this app.

## Endpoints

- `GET /health` — model name, threshold, threshold source, image size, class list
- `POST /predict` — multipart: `image`, plus optional `age`, `sex`, `site`,
  `tta` (default true), `gradcam` (default false)

## Demo mode

With no model present the app still runs, labels itself "Demo mode", and generates a
deterministic stand-in probability. Never present those numbers as results.
