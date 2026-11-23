"""WaferSense AI - Streamlit dashboard.  Run:  streamlit run app.py"""
import base64
import io
import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from common import CLASSES, WaferCNN, gradcam, onehot_batch, preprocess_map

ART = os.environ.get("WAFER_ARTIFACTS", "artifacts")
WAFER_CMAP = ListedColormap(["#d9dde3", "#2a9d8f", "#e63946"])  # background, pass, fail (visual choice only)
LABELS_NICE = {c: ("None" if c == "none" else c) for c in CLASSES}

st.set_page_config(page_title="WaferSense AI", page_icon="🔬", layout="wide")


@st.cache_resource
def load_model():
    ck = torch.load(os.path.join(ART, "model.pt"), map_location="cpu")
    m = WaferCNN(len(ck["classes"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


@st.cache_data
def load_metrics():
    with open(os.path.join(ART, "metrics.json")) as f:
        return json.load(f)


@st.cache_data
def load_samples():
    d = np.load(os.path.join(ART, "samples.npz"))
    return d["x"], d["y"]


if not os.path.exists(os.path.join(ART, "model.pt")):
    st.title("🔬 WaferSense AI")
    st.error("Trained model not found. Run `python train.py --data <path-to-WM811K>` first "
             "(or `python train.py --synthetic` for a quick code test). It creates the `artifacts/` folder.")
    st.stop()

model, M = load_model(), load_metrics()
SX, SY = load_samples()
st.session_state.setdefault("log", [])
st.session_state.setdefault("result", None)

if "SYNTHETIC" in M["dataset"]:
    st.warning("⚠️ This model was trained on SYNTHETIC test data, not WM-811K. Numbers are meaningless.")


# ---------------------------------------------------------------- helpers
def map_fig(idx, title="Wafer map"):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(idx, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(title, fontsize=10); ax.axis("off")
    return fig


def cam_figs(idx, cam):
    f1, a1 = plt.subplots(figsize=(3.2, 3.2))
    a1.imshow(cam, cmap="jet", vmin=0, vmax=1); a1.set_title("Grad-CAM heatmap", fontsize=10); a1.axis("off")
    f2, a2 = plt.subplots(figsize=(3.2, 3.2))
    a2.imshow(idx, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
    a2.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    a2.set_title("Overlay", fontsize=10); a2.axis("off")
    return f1, f2


def run_prediction(idx, true=None, target=None):
    x = onehot_batch(torch.from_numpy(idx[None]))
    cam, probs, k = gradcam(model, x, target)
    pred = int(np.argmax(probs))
    return {"idx": idx, "cam": cam, "probs": probs, "pred": pred, "cam_class": k, "true": true}


def load_uploaded(f):
    name = f.name.lower()
    if name.endswith(".npy"):
        a = np.load(f)
    elif name.endswith(".csv"):
        a = np.loadtxt(f, delimiter=",")
    else:  # image: dark = background, mid = pass, bright = fail (approximate)
        from PIL import Image
        g = np.array(Image.open(f).convert("L"))
        a = np.digitize(g, [85, 170])
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("Wafer map must be a 2D array")
    return preprocess_map(a)


def fig_to_b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🔬 WaferSense AI")
page = st.sidebar.radio("Navigate", ["Home", "Wafer Inspector", "Explainability", "Analytics",
                                      "Model Performance", "Report Export"])
st.sidebar.caption("Decision-support demo. Not for production control or quality certification.")

# ---------------------------------------------------------------- Home
if page == "Home":
    st.title("🔬 WaferSense AI")
    st.subheader("Explainable semiconductor wafer-map defect classification")
    st.write("A software-only prototype that learns spatial failure patterns from wafer maps, predicts the "
             "defect-pattern category, and shows Grad-CAM heatmaps of the regions influencing the prediction.")
    st.markdown("**Workflow:** EDA → preprocessing → stratified split → baseline (Random Forest) → CNN → "
                "class-weighted loss → class-aware evaluation → Grad-CAM → dashboard.")
    s = M["dataset_summary"]
    c = st.columns(4)
    c[0].metric("Dataset", "WM-811K" if "WM-811K" == M["dataset"] else "Synthetic")
    c[1].metric("Total records", f"{s['total_records']:,}")
    c[2].metric("Labelled records", f"{s['labelled_records']:,}")
    c[3].metric("Subset used", f"{M['subset_size']:,}")
    st.info("**Responsible use:** The model recognises *spatial patterns on labelled wafer maps*. It does not inspect "
            "physical wafers, diagnose root causes, or certify quality. Grad-CAM is an inspection aid, not proof "
            "of causal reasoning. Results may not generalise to other fabs, tools or process nodes.")

# ---------------------------------------------------------------- Inspector
elif page == "Wafer Inspector":
    st.title("Wafer Inspector")
    src = st.radio("Input", ["Demo sample (held-out test set)", "Upload wafer map"], horizontal=True)
    idx, true = None, None
    if src.startswith("Demo"):
        c1, c2 = st.columns(2)
        cls = c1.selectbox("Class", CLASSES, format_func=lambda c: LABELS_NICE[c])
        ids = np.where(SY == CLASSES.index(cls))[0]
        if len(ids) == 0:
            st.warning("No samples of this class in the saved demo set.")
        else:
            n = c2.number_input("Sample #", 1, len(ids), 1)
            idx, true = SX[ids[n - 1]], cls
    else:
        up = st.file_uploader("Upload .npy / .csv (values 0=background, 1=pass, 2=fail) or a wafer-map image",
                              type=["npy", "csv", "png", "jpg", "jpeg"])
        st.caption("Images are converted approximately: dark → background, mid → pass, bright → fail.")
        if up is not None:
            try:
                idx = load_uploaded(up)
            except Exception as e:
                st.error(f"Could not read file: {e}")
    if idx is not None and st.button("Predict", type="primary"):
        res = run_prediction(idx, true)
        st.session_state["result"] = res
        st.session_state["log"].append({"prediction": CLASSES[res["pred"]], "confidence": float(res["probs"].max()),
                                        "true_label": true or "—"})
    r = st.session_state["result"]
    if r is not None:
        c1, c2 = st.columns([1, 1])
        c1.pyplot(map_fig(r["idx"], "Input wafer map"))
        pred = CLASSES[r["pred"]]
        c2.metric("Predicted pattern", LABELS_NICE[pred], f"{r['probs'].max()*100:.1f}% confidence")
        if r["true"]:
            c2.write(f"Known label: **{LABELS_NICE[r['true']]}** — " + ("✅ match" if r["true"] == pred else "❌ mismatch"))
        if r["probs"].max() < 0.6:
            c2.warning("Low confidence — send for human review.")
        df = pd.DataFrame({"Class": [LABELS_NICE[c] for c in CLASSES], "Probability": r["probs"]}).set_index("Class")
        c2.bar_chart(df)
        st.caption("Go to **Explainability** to see which regions influenced this prediction.")

# ---------------------------------------------------------------- Explainability
elif page == "Explainability":
    st.title("Explainability (Grad-CAM)")
    r = st.session_state["result"]
    if r is None:
        st.info("Run a prediction in **Wafer Inspector** first.")
    else:
        tgt = st.selectbox("Explain class", CLASSES, index=r["pred"], format_func=lambda c: LABELS_NICE[c])
        res = run_prediction(r["idx"], r["true"], CLASSES.index(tgt))
        f1, f2 = cam_figs(res["idx"], res["cam"])
        c = st.columns(3)
        c[0].pyplot(map_fig(res["idx"], "Original"))
        c[1].pyplot(f1)
        c[2].pyplot(f2)
        st.caption("Red/yellow = regions that pushed the model's score for the selected class. This is a coarse "
                   "inspection aid, not proof of causality or of prediction correctness.")

# ---------------------------------------------------------------- Analytics
elif page == "Analytics":
    st.title("Analytics")
    st.subheader("Class distribution (all labelled records)")
    st.bar_chart(pd.Series(M["full_class_counts"]).reindex(CLASSES).fillna(0))
    st.subheader("Samples per split (subset used)")
    st.dataframe(pd.DataFrame(M["split_counts"]).T.assign(Total=lambda d: d.sum(axis=1)), width="stretch")
    st.subheader("Predictions in this session")
    if st.session_state["log"]:
        st.dataframe(pd.DataFrame(st.session_state["log"]), width="stretch")
    else:
        st.caption("No predictions yet.")

# ---------------------------------------------------------------- Model performance
elif page == "Model Performance":
    st.title("Model Performance (held-out test set)")
    c = st.columns(3)
    c[0].metric("Accuracy", f"{M['cnn']['accuracy']*100:.2f}%")
    c[1].metric("Macro-F1", f"{M['cnn']['macro_f1']:.3f}")
    c[2].metric("Balanced accuracy", f"{M['cnn']['balanced_accuracy']:.3f}")
    st.subheader("CNN vs baseline (Random Forest)")
    st.dataframe(pd.DataFrame({"CNN": M["cnn"], "Random Forest": M["baseline_rf"]}).T.round(4))
    st.subheader("Per-class precision / recall / F1")
    pc = pd.DataFrame({k: v for k, v in M["per_class"].items() if k in CLASSES}).T[["precision", "recall", "f1-score", "support"]]
    st.dataframe(pc.round(3), width="stretch")
    st.subheader("Confusion matrix")
    cm = np.array(M["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    st.pyplot(fig)
    st.subheader("Training curves")
    h = M["history"]
    st.line_chart(pd.DataFrame({"train loss": h["train_loss"], "val macro-F1": h["val_macro_f1"], "val acc": h["val_acc"]}))
    st.caption("Evaluated once on the held-out test set. A public-dataset benchmark is not evidence of production readiness.")

# ---------------------------------------------------------------- Report export
elif page == "Report Export":
    st.title("Report Export")
    r = st.session_state["result"]
    if r is None:
        st.info("Run a prediction in **Wafer Inspector** first.")
    else:
        f0 = map_fig(r["idx"], "Original"); f1, f2 = cam_figs(r["idx"], r["cam"])
        imgs = "".join(f'<img src="data:image/png;base64,{fig_to_b64(f)}" width="300">' for f in (f0, f1, f2))
        rows = "".join(f"<tr><td>{LABELS_NICE[c]}</td><td>{p*100:.2f}%</td></tr>" for c, p in zip(CLASSES, r["probs"]))
        html = f"""<html><body style="font-family:sans-serif;max-width:1000px;margin:auto">
<h1>WaferSense AI - Inspection Summary</h1>
<p><b>Predicted pattern:</b> {LABELS_NICE[CLASSES[r['pred']]]} ({r['probs'].max()*100:.1f}% confidence)<br>
<b>Dataset used for training:</b> {M['dataset']}</p>
<div>{imgs}</div><h3>Class probabilities</h3><table border="1" cellpadding="5">{rows}</table>
<p style="color:#666"><i>Decision-support output only. Grad-CAM is an inspection aid, not proof of causality. Not a
certification of wafer quality.</i></p></body></html>"""
        c = st.columns(3)
        c[0].pyplot(f0); c[1].pyplot(f1); c[2].pyplot(f2)
        st.dataframe(pd.DataFrame({"Class": [LABELS_NICE[k] for k in CLASSES],
                                   "Probability (%)": (r["probs"] * 100).round(2)}), hide_index=True)
        st.download_button("⬇️ Download report (HTML)", html, file_name="wafersense_report.html", mime="text/html")
