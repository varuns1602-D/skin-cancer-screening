"""
Backend for the skin cancer front end.

Mirrors the notebook's inference path exactly:
  5-stage preprocessing -> 384px RGB [0,1] -> {"image", "meta"} -> binary head
  -> optional 8x TTA -> compare against the validation-fitted frozen threshold.

Any drift between this file and the notebook's preprocess_bgr / encode_meta means
the served predictions are not the ones you evaluated. The functions below are
copied verbatim from the notebook for that reason -- do not "clean them up".

Endpoints
    GET  /health    model, threshold, image size, class names
    POST /predict   image (+ age, sex, site, tta, gradcam) -> probabilities
    GET  /          serves index.html

Run
    pip install -r requirements.txt
    python app.py
"""

import base64
import io
import json
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# ---------------------------------------------------------------- artifacts
MODEL_PATH = os.environ.get("MODEL_PATH", "skin_cancer_v3.keras")
CFG_PATH   = os.environ.get("CFG_PATH",   "decision_threshold.json")

# From the notebook. DX8 is the auxiliary head's output order; MALIGNANT is the
# grouping used to build the binary label.
DX8       = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc", "scc"]
MALIGNANT = {"mel", "bcc", "scc", "akiec"}
DX_NAMES  = {
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevus",
    "bcc":   "Basal cell carcinoma",
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bkl":   "Benign keratosis-like lesion",
    "df":    "Dermatofibroma",
    "vasc":  "Vascular lesion",
    "scc":   "Squamous cell carcinoma",
}
SITES = ["head/neck", "upper extremity", "lower extremity", "torso",
         "palms/soles", "oral/genital", "anterior torso", "posterior torso",
         "lateral torso", "unknown"]
SITE2I = {s: i for i, s in enumerate(SITES)}

app = Flask(__name__, static_folder=".")
model = None
THRESHOLD = 0.5
IMG_SIZE = 384
META_DIM = 1 + 3 + len(SITES)


# ============================================================== preprocessing
# Verbatim from notebook section 9.
def crop_vignette(img_bgr, dark_thresh=28, min_keep=0.45):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bright = (gray > dark_thresh).astype(np.uint8)
    k = max(9, (int(min(gray.shape) * 0.06) | 1))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    if n <= 1:
        return img_bgr
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[i]
    H, W = gray.shape
    if area / (H * W) > 0.97:
        return img_bgr
    side = int(min(w, h) * 0.72)
    if side < min_keep * min(H, W):
        return img_bgr
    cx, cy = x + w // 2, y + h // 2
    x0 = max(0, min(W - side, cx - side // 2))
    y0 = max(0, min(H - side, cy - side // 2))
    return img_bgr[y0:y0 + side, x0:x0 + side]


def _directional_ses(length, angles=(0, 45, 90, 135)):
    ses = []
    for a in angles:
        se = np.zeros((length, length), np.uint8)
        se[length // 2, :] = 1
        M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), a, 1.0)
        ses.append(cv2.warpAffine(se, M, (length, length),
                                  flags=cv2.INTER_NEAREST).astype(np.uint8))
    return ses


def adaptive_hair_removal(img_bgr, scales=(0.020, 0.035), min_elong=2.5,
                          max_cov=0.35, min_cov=0.0005):
    h, w = img_bgr.shape[:2]
    diag = (h ** 2 + w ** 2) ** 0.5
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resp = np.zeros_like(gray)
    for s in scales:
        L = max(7, int(round(diag * s)) | 1)
        for se in _directional_ses(L):
            resp = np.maximum(resp, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se))
    if resp.max() < 5:
        mask = np.zeros_like(gray)
    else:
        _, mask = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if mask.mean() / 255 > max_cov:
            _, mask = cv2.threshold(resp, np.percentile(resp, 100 * (1 - max_cov / 2)),
                                    255, cv2.THRESH_BINARY)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        _, _, bw, bh_, area = stats[i]
        if area < 3:
            continue
        elong = max(bw, bh_) / max(1, min(bw, bh_))
        fill = area / max(1, bw * bh_)
        if elong >= min_elong or (fill < 0.35 and max(bw, bh_) > 10):
            keep[lab == i] = 255
    mask = cv2.dilate(cv2.morphologyEx(keep, cv2.MORPH_CLOSE,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))),
                      np.ones((3, 3), np.uint8), iterations=1)
    cov = mask.mean() / 255
    if cov < min_cov or cov > max_cov:
        return img_bgr
    return cv2.inpaint(img_bgr, mask, 1, cv2.INPAINT_TELEA)


def shades_of_gray(img_bgr, p=6):
    img = img_bgr.astype(np.float32)
    illum = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1.0 / p)
    illum = illum / (np.sqrt(np.sum(illum ** 2)) + 1e-8)
    return np.clip(img / (illum * np.sqrt(3.0) + 1e-8), 0, 255).astype(np.uint8)


def apply_clahe(img_bgr, clip_limit=2.0, tile=(8, 8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_bgr(img, size):
    img = crop_vignette(img)                                          # 1
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)  # 2
    img = adaptive_hair_removal(img)                                   # 3
    img = shades_of_gray(img)                                          # 4
    img = apply_clahe(img)                                             # 5
    return img


def encode_meta(age=None, sex=None, site=None):
    """Matches the notebook's encode_meta: age/100 clipped, sex one-hot at 1..3,
    site one-hot at 4.., with an explicit unknown slot rather than imputation."""
    v = np.zeros((1, META_DIM), np.float32)
    if age not in (None, ""):
        try:
            v[0, 0] = min(max(float(age) / 100.0, 0.0), 1.0)
        except (TypeError, ValueError):
            pass
    s = str(sex or "").lower()
    v[0, 1 + (0 if s.startswith("m") else 1 if s.startswith("f") else 2)] = 1.0
    v[0, 4 + SITE2I.get(str(site or "").lower().strip(), SITE2I["unknown"])] = 1.0
    return v


# ================================================================== model I/O
def load_artifacts():
    global model, THRESHOLD, IMG_SIZE, META_DIM
    if os.path.exists(CFG_PATH):
        c = json.load(open(CFG_PATH))
        THRESHOLD = float(c.get("threshold", 0.5))
        IMG_SIZE  = int(c.get("img_size", 384))
        META_DIM  = int(c.get("meta_dim", META_DIM))
        print(f"[ok] threshold {THRESHOLD} | img_size {IMG_SIZE} | meta_dim {META_DIM}")
    else:
        print(f"[warn] no {CFG_PATH} -- falling back to threshold 0.5, which is NOT "
              f"the operating point you evaluated")

    if not os.path.exists(MODEL_PATH):
        print(f"[warn] no model at {MODEL_PATH} -- front end runs in demo mode")
        return
    from tensorflow import keras
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(f"[ok] loaded {MODEL_PATH}")


def tta_batch(x):
    """The notebook's 8 TTA views, as numpy. Averaging is order-independent."""
    return [
        x,
        x[:, :, ::-1, :],
        x[:, ::-1, :, :],
        np.rot90(x, 1, (1, 2)),
        np.rot90(x, 2, (1, 2)),
        np.rot90(x, 3, (1, 2)),
        np.rot90(x, 1, (1, 2))[:, :, ::-1, :],
        np.rot90(x, 1, (1, 2))[:, ::-1, :, :],
    ]


def gradcam(img_batch, meta_batch):
    """Keras-3-safe Grad-CAM: explicit replay of the head inside the tape, by
    layer name. Auto-detection cannot work here -- the metadata branch
    disconnects the gradient graph."""
    import tensorflow as tf
    bb = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    if bb is None:
        raise RuntimeError("no nested backbone found")
    with tf.GradientTape() as tape:
        x = model.get_layer("to_model_range")(tf.cast(img_batch, tf.float32))
        conv = tf.cast(bb(x, training=False), tf.float32)
        tape.watch(conv)
        p = model.get_layer("gap")(conv)
        p = model.get_layer("ln")(p)
        p = model.get_layer("drop1")(p, training=False)
        m = model.get_layer("meta_dense")(tf.cast(meta_batch, tf.float32))
        z = model.get_layer("fuse")([p, m])
        z = model.get_layer("fuse_dense")(z)
        z = model.get_layer("drop2")(z, training=False)
        score = tf.cast(model.get_layer("binary")(z), tf.float32)[:, 0]
    g = tape.gradient(score, conv)
    if g is None:
        raise RuntimeError("Grad-CAM graph disconnected")
    pooled = tf.reduce_mean(g, axis=(0, 1, 2))
    hm = tf.squeeze(conv[0] @ pooled[..., tf.newaxis])
    hm = tf.maximum(hm, 0) / (tf.reduce_max(hm) + 1e-8)
    return np.nan_to_num(np.asarray(hm, np.float32), nan=0.0)


def overlay_png(rgb01, hm, alpha=0.42):
    h = cv2.resize(hm.astype(np.float32), (rgb01.shape[1], rgb01.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(h, 0, 1)), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
    out = np.clip((1 - alpha) * rgb01 + alpha * heat, 0, 1)
    bgr = cv2.cvtColor(np.uint8(out * 255), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("could not encode overlay")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


# =================================================================== routes
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/health")
def health():
    if model is None:
        return jsonify(error="no model loaded"), 503
    return jsonify(
        model=os.path.basename(MODEL_PATH),
        threshold=THRESHOLD,
        threshold_source="decision_threshold.json" if os.path.exists(CFG_PATH) else "default 0.5",
        img_size=IMG_SIZE,
        dx_classes=[{"code": d, "name": DX_NAMES[d],
                     "tier": "malignant" if d in MALIGNANT else "benign"} for d in DX8],
        sites=SITES,
        gradcam=True,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(error="No model loaded on the server."), 503
    if "image" not in request.files:
        return jsonify(error="No image in the request."), 400

    raw = np.frombuffer(request.files["image"].read(), np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify(error="That file could not be read as an image."), 400

    prepped = preprocess_bgr(bgr, IMG_SIZE)
    rgb01 = cv2.cvtColor(prepped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = rgb01[None]

    mv = encode_meta(request.form.get("age"),
                     request.form.get("sex"),
                     request.form.get("site"))

    use_tta = request.form.get("tta", "true").lower() in ("1", "true", "yes")
    views = tta_batch(x) if use_tta else [x]

    bins, dxs = [], []
    for v in views:
        out = model({"image": np.ascontiguousarray(v), "meta": mv}, training=False)
        if isinstance(out, dict):
            bins.append(float(np.asarray(out["binary"]).flatten()[0]))
            if "dx" in out:
                dxs.append(np.asarray(out["dx"], np.float32).flatten())
        else:
            bins.append(float(np.asarray(out).flatten()[0]))

    p_mal = float(np.mean(bins))
    dx = np.mean(dxs, axis=0).tolist() if dxs else None

    payload = {
        "malignant_probability": round(p_mal, 6),
        "benign_probability": round(1.0 - p_mal, 6),
        "threshold": THRESHOLD,
        "prediction": "malignant" if p_mal >= THRESHOLD else "benign",
        "tta": len(views),
        "dx": dx,
        "dx_classes": DX8,
    }

    if request.form.get("gradcam", "").lower() in ("1", "true", "yes"):
        try:
            payload["gradcam"] = overlay_png(rgb01, gradcam(x, mv))
        except Exception as e:
            payload["gradcam_error"] = f"{type(e).__name__}: {e}"

    return jsonify(payload)


if __name__ == "__main__":
    load_artifacts()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
