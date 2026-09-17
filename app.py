from io import BytesIO
from pathlib import Path
import pickle
import threading
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ============================================================
# 1. Application and model configuration
# ============================================================
st.set_page_config(
    page_title="LDAR Risk Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "svm_model.pkl"
SCALER_PATH = APP_DIR / "svm_scaler.pkl"
PLOT_LOCK = threading.Lock()

# The order must exactly match the feature order used during model training.
FEATURES = [
    {
        "name": "Age",
        "label": "Age",
        "unit": "years",
        "default": 68.0,
        "step": 1.0,
        "min": 26.58,
        "max": 93.33,
    },
    {
        "name": "LDH",
        "label": "Lactate dehydrogenase",
        "unit": "U/L",
        "default": 178.0,
        "step": 1.0,
        "min": 98.0,
        "max": 2159.0,
    },
    {
        "name": "FDP",
        "label": "Fibrin degradation products",
        "unit": "mg/L",
        "default": 1.26,
        "step": 0.01,
        "min": 0.0,
        "max": 67.43,
    },
    {
        "name": "CA125",
        "label": "CA 125",
        "unit": "U/mL",
        "default": 9.60,
        "step": 0.10,
        "min": 2.2,
        "max": 2055.9,
    },
    {
        "name": "CEA",
        "label": "Carcinoembryonic antigen",
        "unit": "ng/mL",
        "default": 3.80,
        "step": 0.10,
        "min": 0.08,
        "max": 689.03,
    },
    {
        "name": "ALB",
        "label": "Albumin",
        "unit": "g/L",
        "default": 39.30,
        "step": 0.10,
        "min": 21.7,
        "max": 53.2,
    },
    {
        "name": "CA199",
        "label": "CA 19-9",
        "unit": "U/mL",
        "default": 14.80,
        "step": 0.10,
        "min": 1.0,
        "max": 1868.2,
    },
]

FEATURE_NAMES = [item["name"] for item in FEATURES]
DISPLAY_NAMES = ["CA 19-9" if name == "CA199" else name for name in FEATURE_NAMES]
POSITIVE_CLASS = 1
OUTCOME_TEXT = "LDAR ≥ 5.27"


# ============================================================
# 2. Visual design
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {color-scheme: light;}
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#f4f7fb;color:#17243d;}
[data-testid="stHeader"] {background:rgba(244,247,251,.88);backdrop-filter:blur(10px);}
.block-container {max-width:1380px;padding:4.5rem 2.2rem 2rem;}
#MainMenu, footer {visibility:hidden;}

/* Sidebar */
[data-testid="stSidebar"] {background:linear-gradient(180deg,#0d1c35 0%,#132849 100%);border-right:0;}
[data-testid="stSidebar"] * {color:#eaf2ff;}
[data-testid="stSidebar"] hr {border-color:#ffffff17;}
.side-brand {padding:9px 2px 18px;}
.side-mark {display:inline-grid;place-items:center;width:38px;height:38px;border-radius:12px;
    background:linear-gradient(135deg,#1bc7b0,#37a7df);color:#071e2d;font-weight:800;margin-bottom:14px;}
.side-brand h2 {color:white!important;font:700 20px/1.3 'Manrope',sans-serif!important;margin:0!important;padding:0!important;}
.side-brand p {color:#9bb0ce!important;font-size:11px;line-height:1.7;margin:5px 0 0;}
.status-card {background:#ffffff0c;border:1px solid #ffffff13;border-radius:14px;padding:14px 15px;margin:5px 0 18px;}
.status-label {font-size:9px;letter-spacing:1.7px;color:#86a0c4;margin-bottom:9px;}
.status-line {font-size:12px;color:#e8f2ff;display:flex;align-items:center;gap:8px;}
.status-dot {width:8px;height:8px;border-radius:50%;background:#2cddab;box-shadow:0 0 0 5px #2cddab15;}
.side-section {font-size:10px;letter-spacing:1.6px;color:#7f99bc;margin:20px 0 9px;text-transform:uppercase;}
.side-row {display:flex;justify-content:space-between;gap:10px;padding:8px 0;border-bottom:1px solid #ffffff0b;font-size:11px;}
.side-row span:first-child {color:#91a7c5;}.side-row span:last-child {color:#eef5ff;font-weight:600;text-align:right;}
.side-note {font-size:10px;line-height:1.75;color:#8fa5c4;background:#07172d55;border-radius:12px;padding:12px 13px;margin-top:16px;}

/* Hero */
.hero {position:relative;overflow:hidden;isolation:isolate;border-radius:25px;padding:34px 38px;
    background:linear-gradient(120deg,#102b4e 0%,#17436b 55%,#126b72 100%);
    box-shadow:0 18px 50px #12335418;margin-bottom:22px;}
.hero:before {content:"";position:absolute;z-index:-1;width:370px;height:370px;right:-80px;top:-180px;
    border-radius:50%;border:64px solid #ffffff0a;box-shadow:0 0 0 55px #ffffff05;}
.hero-grid {display:grid;grid-template-columns:auto 1fr;gap:23px;align-items:center;}
.hero-icon {width:70px;height:70px;border-radius:22px;display:grid;place-items:center;
    background:linear-gradient(145deg,#28d0b1,#49b4db);box-shadow:0 12px 26px #07192f35;
    color:#08263d;font:800 31px 'Manrope',sans-serif;}
.hero-kicker {font-size:10px;letter-spacing:2.5px;color:#74e5d1;text-transform:uppercase;font-weight:700;margin-bottom:7px;}
.hero h1 {font:800 clamp(27px,3vw,40px)/1.25 'Manrope',sans-serif!important;color:#fff!important;margin:0!important;padding:0!important;letter-spacing:-.7px;}
.hero p {font-size:13px;color:#c8dbed!important;line-height:1.8;margin:9px 0 16px;max-width:770px;}
.pills {display:flex;flex-wrap:wrap;gap:8px;}
.pill {font-size:10px;color:#e5f5ff;border:1px solid #ffffff20;background:#ffffff0d;padding:6px 11px;border-radius:30px;}
.pill.accent {color:#a8ffeb;background:#22ba9a22;border-color:#65e8cc35;}

/* Common cards and headings */
.st-key-input_card,.st-key-result_card {background:#fff;border:1px solid #e6ecf4;border-radius:20px;
    box-shadow:0 8px 28px #18325408;padding:23px 25px!important;margin-bottom:18px;}
.section-head {display:flex;align-items:flex-start;gap:12px;margin-bottom:17px;}
.section-no {display:grid;place-items:center;flex:0 0 auto;width:35px;height:35px;border-radius:11px;
    background:#e6f7f4;color:#0d8c7d;font:700 12px 'Manrope',sans-serif;}
.section-no.blue {background:#e9f1fb;color:#246aa0;}
.section-head h2 {font:700 19px/1.4 'Manrope',sans-serif!important;color:#172a46!important;margin:0!important;padding:0!important;}
.section-head p {font-size:11px;color:#8290a5!important;margin:3px 0 0;line-height:1.6;}
.micro-note {font-size:10px;color:#8996a8;line-height:1.75;margin-top:3px;}

/* Inputs */
[data-testid="stForm"] {border:0!important;padding:0!important;}
[data-testid="stNumberInput"] label p {font-size:11px!important;font-weight:600;color:#3d506d;}
[data-testid="stNumberInput"] [data-baseweb="input"] {background:#f4f7fb;border:1px solid #e3eaf3;border-radius:10px;}
[data-testid="stNumberInput"] input {background:#f4f7fb;color:#152946!important;font-size:14px;font-weight:600;}
[data-testid="stNumberInput"] button {background:#f4f7fb;color:#738199;}
[data-testid="stFormSubmitButton"] button {border:0!important;border-radius:11px!important;min-height:45px;
    background:linear-gradient(105deg,#137da0,#0d9b8d)!important;box-shadow:0 7px 18px #147e9727;
    color:#fff!important;font-weight:700;transition:transform .18s,filter .18s;}
[data-testid="stFormSubmitButton"] button p {color:#fff!important;}
[data-testid="stFormSubmitButton"] button:hover {filter:brightness(1.06);transform:translateY(-1px);}
.range-note {border-radius:11px;padding:10px 12px;background:#f0f8f7;border:1px solid #e1f0ed;
    font-size:10px;color:#587d79;line-height:1.7;margin:10px 0 1px;}

/* Empty result */
.empty-result {min-height:258px;display:grid;place-items:center;text-align:center;border-radius:17px;
    background:radial-gradient(circle at 70% 20%,#e6f6f3 0,transparent 33%),linear-gradient(145deg,#f5f8fc,#f0f6f8);
    border:1px dashed #d6e2eb;padding:25px;}
.empty-symbol {width:82px;height:82px;border-radius:24px;background:linear-gradient(145deg,#dbeaf5,#dcf3ed);
    display:grid;place-items:center;margin:0 auto 14px;font-size:28px;color:#178d83;transform:rotate(-5deg);}
.empty-result h3 {font:700 17px 'Manrope',sans-serif;color:#27415f;margin:0 0 7px;}
.empty-result p {font-size:11px;color:#8090a6!important;line-height:1.75;margin:0;}
.empty-facts {display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin-top:12px;}
.empty-fact {border-radius:12px;padding:12px 8px;text-align:center;background:#f5f8fc;border:1px solid #ebeff5;}
.empty-fact strong {font:700 16px 'Manrope',sans-serif;color:#206c8f;display:block;}
.empty-fact span {font-size:9px;color:#8290a4;}

/* Result summary */
.report-strip {display:flex;align-items:center;justify-content:space-between;gap:18px;border-radius:19px;
    background:linear-gradient(125deg,#eef6fb,#eef9f6);border:1px solid #dcebee;padding:20px 23px;margin-bottom:12px;}
.report-label {font-size:10px;text-transform:uppercase;letter-spacing:1.3px;color:#6a8298;font-weight:700;}
.report-value {font:800 43px/1.25 'Manrope',sans-serif;color:#146f87;letter-spacing:-1.3px;margin:4px 0;}
.report-value span {font-size:21px;margin-left:3px;}
.report-sub {font-size:10px;color:#8694a7;}
.gauge {width:92px;height:92px;display:grid;place-items:center;border-radius:50%;flex:0 0 auto;}
.gauge-inner {width:70px;height:70px;border-radius:50%;background:#f6fafc;display:grid;place-items:center;
    color:#41647d;font-size:10px;text-align:center;line-height:1.4;font-weight:700;}
.result-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px;}
.result-stat {border-radius:13px;padding:13px 14px;background:#f7f9fc;border:1px solid #e9eef4;}
.result-stat:nth-child(2) {background:#f0f9f6;border-color:#e0f0eb;}
.result-stat:nth-child(3) {background:#fff6ef;border-color:#f5e8dc;}
.result-stat small {display:block;font-size:9px;text-transform:uppercase;letter-spacing:.7px;color:#8491a4;margin-bottom:6px;}
.result-stat strong {font:700 13px/1.45 'Manrope',sans-serif;color:#28435e;}
.result-stat:nth-child(2) strong {color:#137d70;}.result-stat:nth-child(3) strong {color:#a96935;}
.alert-positive,.alert-negative {padding:11px 14px;border-radius:11px;font-size:11px;line-height:1.7;margin:8px 0 13px;}
.alert-positive {background:#fff2ed;color:#9e4f38;border-left:4px solid #e87958;}
.alert-negative {background:#eaf8f4;color:#267769;border-left:4px solid #31aa91;}

/* Tabs, charts, table, downloads */
[data-testid="stTabs"] [role="tablist"] {gap:18px;border-bottom:1px solid #e7edf4;overflow-x:auto;}
[data-testid="stTabs"] [role="tab"] {height:42px;white-space:nowrap;}
[data-testid="stTabs"] [role="tab"] p {font-size:10px;font-weight:700;color:#728097;}
[data-testid="stTabs"] [aria-selected="true"] p {color:#0a887c!important;}
.chart-guide {display:flex;gap:17px;flex-wrap:wrap;padding:10px 2px 3px;font-size:10px;color:#78869b;}
.chart-guide i {display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;}
[data-testid="stDataFrame"] {border:1px solid #e9eef4;border-radius:11px;overflow:hidden;}
[data-testid="stDownloadButton"] button {border:1px solid #dce7ed;border-radius:10px;background:#f8fbfc;min-height:39px;}
[data-testid="stDownloadButton"] button p {color:#247082!important;font-size:10px;font-weight:700;}
.method-note {border-radius:12px;background:#f5f8fb;padding:13px 15px;color:#6c7d92;font-size:10px;line-height:1.85;}

.page-footer {display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap;padding:20px 3px 0;
    border-top:1px solid #e2e9f1;color:#8b98a9;font-size:9px;margin-top:8px;}

@media(max-width:900px) {
    .block-container {padding:4.3rem 1rem 1.5rem;}.hero {padding:26px 23px;}.hero-grid {grid-template-columns:1fr;gap:13px;}
    .hero-icon {width:52px;height:52px;border-radius:15px;font-size:23px;}.hero h1 {font-size:29px!important;}
    .card {padding:19px 17px;}.result-grid {grid-template-columns:1fr;}.report-value {font-size:37px;}
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 3. Load and validate model files
# ============================================================
@st.cache_resource
def load_model_files():
    missing = [path.name for path in (MODEL_PATH, SCALER_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing file(s) in the GitHub repository root: " + ", ".join(missing)
        )

    # Only load trusted model files created by the project owner.
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    with SCALER_PATH.open("rb") as file:
        scaler = pickle.load(file)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The SVM does not provide predict_proba. It must be trained with probability=True."
        )
    if getattr(model, "n_features_in_", None) != len(FEATURE_NAMES):
        raise ValueError("The model does not expect exactly seven predictors.")

    scaler_names = getattr(scaler, "feature_names_in_", None)
    if scaler_names is not None and list(scaler_names) != FEATURE_NAMES:
        raise ValueError(
            f"Feature order mismatch. Scaler: {list(scaler_names)}; app: {FEATURE_NAMES}."
        )

    classes = np.asarray(getattr(model, "classes_", []))
    positive_positions = np.flatnonzero(classes == POSITIVE_CLASS)
    if len(positive_positions) != 1:
        raise ValueError(f"Positive class 1 was not found in model classes {classes.tolist()}.")

    return model, scaler, int(positive_positions[0])


try:
    model, scaler, positive_index = load_model_files()
except Exception as error:
    st.error(f"Application startup failed: {error}")
    st.stop()


# ============================================================
# 4. Prediction and SHAP computation
# ============================================================
def positive_probability(standardized_data):
    standardized_data = np.asarray(standardized_data, dtype=float)
    return model.predict_proba(standardized_data)[:, positive_index]


def calculate_prediction(input_values):
    patient = pd.DataFrame(
        [[input_values[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
        dtype=float,
    )
    if not np.isfinite(patient.to_numpy()).all():
        raise ValueError("All seven predictors must contain finite numeric values.")
    if (patient.to_numpy() < 0).any():
        raise ValueError("Predictor values cannot be negative.")

    standardized_patient = scaler.transform(patient)
    probability = float(positive_probability(standardized_patient)[0])
    predicted_class = int(model.predict(standardized_patient)[0])

    # For StandardScaler, zero represents the training-set mean profile.
    # It provides a fixed, reproducible patient-level SHAP reference.
    background = np.zeros((1, len(FEATURE_NAMES)), dtype=float)
    explainer = shap.KernelExplainer(
        positive_probability,
        background,
        feature_names=FEATURE_NAMES,
        link="identity",
    )
    raw_shap = explainer.shap_values(
        standardized_patient,
        nsamples=2 ** len(FEATURE_NAMES),
        l1_reg=0.0,
        silent=True,
    )
    shap_values = np.asarray(raw_shap, dtype=float).reshape(-1)
    baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    if shap_values.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"Unexpected SHAP result shape: {shap_values.shape}.")
    if not np.isclose(baseline + shap_values.sum(), probability, atol=1e-6):
        raise ValueError("SHAP additivity check failed for this prediction.")

    out_of_range = [
        item["label"]
        for item in FEATURES
        if not item["min"] <= input_values[item["name"]] <= item["max"]
    ]
    return {
        "probability": probability,
        "predicted_class": predicted_class,
        "baseline": baseline,
        "shap_values": shap_values,
        "input_values": {name: float(input_values[name]) for name in FEATURE_NAMES},
        "out_of_range": out_of_range,
    }


def _save_current_figure(width=None, height=None):
    if width is not None and height is not None:
        plt.gcf().set_size_inches(width, height)
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=185, bbox_inches="tight", facecolor="white")
    return buffer.getvalue()


def make_waterfall_plot(result):
    explanation = shap.Explanation(
        values=result["shap_values"],
        base_values=result["baseline"],
        data=np.array([result["input_values"][name] for name in FEATURE_NAMES]),
        feature_names=DISPLAY_NAMES,
    )
    with PLOT_LOCK:
        plt.figure()
        try:
            shap.plots.waterfall(explanation, max_display=7, show=False)
            plt.title("Local SHAP waterfall", fontsize=14, weight="bold", pad=18, color="#19324d")
            return _save_current_figure(9.2, 5.4)
        finally:
            plt.close("all")


def make_force_plot(result):
    patient_values = np.array([result["input_values"][name] for name in FEATURE_NAMES])
    with PLOT_LOCK:
        plt.figure()
        try:
            shap.force_plot(
                result["baseline"],
                result["shap_values"],
                patient_values,
                feature_names=DISPLAY_NAMES,
                matplotlib=True,
                show=False,
                figsize=(12, 3.2),
                text_rotation=12,
            )
            plt.title("Local SHAP force plot", fontsize=14, weight="bold", pad=17, color="#19324d")
            return _save_current_figure(12, 3.5)
        finally:
            plt.close("all")


def make_nightingale_plot(result):
    values = np.asarray(result["shap_values"])
    order = np.argsort(np.abs(values))[::-1]
    ordered_values = values[order]
    ordered_names = np.array(DISPLAY_NAMES)[order]
    radii = np.abs(ordered_values)
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    width = 2 * np.pi / len(values) * 0.78
    maximum = max(float(radii.max()), 1e-9)
    colors = []
    for value, radius in zip(ordered_values, radii):
        intensity = 0.38 + 0.52 * radius / maximum
        colors.append(plt.cm.Reds(intensity) if value >= 0 else plt.cm.Blues(intensity))

    with PLOT_LOCK:
        fig, axis = plt.subplots(figsize=(8.2, 7.4), subplot_kw={"polar": True})
        try:
            axis.bar(angles, radii, width=width, color=colors, edgecolor="white", linewidth=1.7, alpha=.94)
            axis.set_theta_offset(np.pi / 2)
            axis.set_theta_direction(-1)
            axis.set_xticks(angles)
            axis.set_xticklabels(ordered_names, fontsize=10, color="#354b65")
            axis.tick_params(axis="x", pad=12)
            axis.set_yticklabels([])
            axis.grid(color="#dfe7ef", linestyle=":", linewidth=.8)
            axis.spines["polar"].set_visible(False)
            for angle, radius, value in zip(angles, radii, ordered_values):
                axis.text(
                    angle,
                    radius + maximum * .075,
                    f"{value:+.4f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#c4543e" if value >= 0 else "#286b9f",
                    weight="bold",
                )
            axis.set_title(
                "Individual SHAP Nightingale chart\nbar length = absolute contribution",
                fontsize=14,
                weight="bold",
                color="#19324d",
                pad=24,
            )
            return _save_current_figure()
        finally:
            plt.close(fig)


def make_bar_plot(result):
    values = np.asarray(result["shap_values"])
    order = np.argsort(np.abs(values))
    ordered_values = values[order]
    ordered_names = np.array(DISPLAY_NAMES)[order]
    colors = np.where(ordered_values >= 0, "#e96f55", "#3098ca")
    with PLOT_LOCK:
        fig, axis = plt.subplots(figsize=(8.8, 5.0))
        try:
            axis.barh(ordered_names, ordered_values, color=colors, height=.62)
            axis.axvline(0, color="#73859a", linewidth=.9)
            axis.grid(axis="x", color="#e8edf2", linestyle=":", linewidth=.8)
            axis.set_axisbelow(True)
            axis.set_xlabel("Contribution to predicted probability", fontsize=10, color="#53667d")
            axis.set_title("Signed SHAP contribution profile", fontsize=14, weight="bold", color="#19324d", pad=14)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.tick_params(axis="y", length=0, labelsize=10, colors="#354b65")
            axis.tick_params(axis="x", labelsize=9, colors="#63758a")
            span = max(float(np.max(np.abs(ordered_values))), 1e-6)
            for row, value in enumerate(ordered_values):
                axis.text(
                    value + np.sign(value or 1) * span * .035,
                    row,
                    f"{value:+.4f}",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=8.5,
                    color="#b84d39" if value >= 0 else "#226d9a",
                    weight="bold",
                )
            axis.set_xlim(
                min(float(ordered_values.min()) - span * .25, -span * .35),
                max(float(ordered_values.max()) + span * .25, span * .35),
            )
            fig.tight_layout()
            return _save_current_figure()
        finally:
            plt.close(fig)


def build_charts(result):
    return {
        "waterfall": make_waterfall_plot(result),
        "force": make_force_plot(result),
        "nightingale": make_nightingale_plot(result),
        "bar": make_bar_plot(result),
    }


def build_chart_zip(charts):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("LDAR_SHAP_waterfall.png", charts["waterfall"])
        archive.writestr("LDAR_SHAP_force.png", charts["force"])
        archive.writestr("LDAR_SHAP_nightingale.png", charts["nightingale"])
        archive.writestr("LDAR_SHAP_bar.png", charts["bar"])
    return buffer.getvalue()


# ============================================================
# 5. Sidebar
# ============================================================
with st.sidebar:
    st.markdown(
        """<div class="side-brand"><div class="side-mark">R</div>
        <h2>LDAR Risk Intelligence</h2>
        <p>Patient-level prediction and transparent model attribution.</p></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="status-card"><div class="status-label">SYSTEM STATUS</div>
        <div class="status-line"><span class="status-dot"></span>SVM engine ready</div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-section">Model profile</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="side-row"><span>Outcome</span><span>LDAR ≥ 5.27</span></div>
        <div class="side-row"><span>Positive class</span><span>1</span></div>
        <div class="side-row"><span>Predictors</span><span>7 clinical variables</span></div>
        <div class="side-row"><span>Algorithm</span><span>Support Vector Machine</span></div>
        <div class="side-row"><span>Explanation</span><span>Kernel SHAP</span></div>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-section">How to use</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="side-note">1. Enter all seven preoperative measurements.<br><br>
        2. Run the individual assessment.<br><br>
        3. Review probability, model class, and SHAP attribution.<br><br>
        4. Download the result table and figures if required.</div>""",
        unsafe_allow_html=True,
    )


# ============================================================
# 6. Main interface
# ============================================================
st.markdown(
    """<section class="hero"><div class="hero-grid"><div class="hero-icon">R</div><div>
    <div class="hero-kicker">Clinical prediction · interpretable machine learning</div>
    <h1>LDAR Risk Intelligence</h1>
    <p>Estimate the probability of LDAR ≥ 5.27 from seven preoperative indicators and explore how each
    measurement influenced the individual prediction.</p>
    <div class="pills"><span class="pill accent">● Model online</span><span class="pill">SVM probability</span>
    <span class="pill">Kernel SHAP</span><span class="pill">Patient-level report</span></div>
    </div></div></section>""",
    unsafe_allow_html=True,
)

input_panel = st.container(key="input_card")
input_panel.markdown(
    """<div class="section-head"><span class="section-no">01</span><div>
    <h2>Clinical parameter input matrix</h2>
    <p>Enter measurements in the same units used during model development.</p></div></div>""",
    unsafe_allow_html=True,
)

with input_panel.form("prediction_form"):
    input_values = {}
    for row in range(0, len(FEATURES), 4):
        columns = st.columns(4, gap="medium")
        for offset, item in enumerate(FEATURES[row : row + 4]):
            with columns[offset]:
                input_values[item["name"]] = st.number_input(
                    f"{item['label']} ({item['unit']})",
                    min_value=0.0,
                    value=float(item["default"]),
                    step=float(item["step"]),
                    format="%.2f",
                    key=item["name"],
                    help=f"Observed training range: {item['min']:g} to {item['max']:g} {item['unit']}",
                )
    st.markdown(
        '<div class="range-note">Values outside the observed training range are accepted for transparency but will trigger an extrapolation warning.</div>',
        unsafe_allow_html=True,
    )
    submitted = st.form_submit_button("Run individual LDAR assessment  →", type="primary", width="stretch")

if submitted:
    st.session_state.pop("latest_result", None)
    st.session_state.pop("latest_charts", None)
    try:
        with st.spinner("Running the SVM and generating four SHAP explanations…"):
            latest_result = calculate_prediction(input_values)
            latest_charts = build_charts(latest_result)
            st.session_state["latest_result"] = latest_result
            st.session_state["latest_charts"] = latest_charts
    except Exception as error:
        st.error(f"The assessment could not be completed: {error}")

result_panel = st.container(key="result_card")
result_panel.markdown(
    """<div class="section-head"><span class="section-no blue">02</span><div>
    <h2>Individual risk inference report</h2>
    <p>Prediction summary and patient-specific model attribution.</p></div></div>""",
    unsafe_allow_html=True,
)

result = st.session_state.get("latest_result")
charts = st.session_state.get("latest_charts")

if result is None or charts is None:
    result_panel.markdown(
        """<div class="empty-result"><div><div class="empty-symbol">◇</div>
        <h3>Your individual report will appear here</h3>
        <p>Complete the seven measurements above and run the assessment.<br>
        The app will generate a probability estimate and four SHAP views.</p></div></div>
        <div class="empty-facts"><div class="empty-fact"><strong>7</strong><span>predictors</span></div>
        <div class="empty-fact"><strong>5.27</strong><span>LDAR outcome cutoff</span></div>
        <div class="empty-fact"><strong>4</strong><span>explanation charts</span></div></div>""",
        unsafe_allow_html=True,
    )
else:
    probability_percent = result["probability"] * 100
    class_positive = result["predicted_class"] == POSITIVE_CLASS
    class_text = "Class 1 · LDAR ≥ 5.27" if class_positive else "Class 0 · LDAR < 5.27"
    net_effect = float(np.sum(result["shap_values"]))
    alert_class = "alert-positive" if class_positive else "alert-negative"
    alert_text = (
        "The SVM classified this profile as LDAR ≥ 5.27. Interpret the estimate alongside the patient's full clinical context."
        if class_positive
        else "The SVM classified this profile as LDAR < 5.27. Continue to interpret the estimate within the full clinical context."
    )

    result_panel.markdown(
        f"""<div class="report-strip"><div><div class="report-label">Predicted probability of LDAR ≥ 5.27</div>
        <div class="report-value">{probability_percent:.2f}<span>%</span></div>
        <div class="report-sub">Calculated from the most recently submitted patient profile</div></div>
        <div class="gauge" style="background:conic-gradient(#16a594 0% {probability_percent:.6f}%,#dce8ed {probability_percent:.6f}% 100%)">
        <div class="gauge-inner">P(LDAR<br>≥ 5.27)</div></div></div>
        <div class="result-grid"><div class="result-stat"><small>Model classification</small><strong>{class_text.replace('<', '&lt;')}</strong></div>
        <div class="result-stat"><small>SHAP reference probability</small><strong>{result['baseline']:.2%}</strong></div>
        <div class="result-stat"><small>Net SHAP effect</small><strong>{net_effect:+.4f}</strong></div></div>
        <div class="{alert_class}">{alert_text}</div>""",
        unsafe_allow_html=True,
    )

    if result["out_of_range"]:
        result_panel.warning(
            "Extrapolation warning: the following value(s) fall outside the observed training range: "
            + ", ".join(result["out_of_range"])
            + "."
        )

    contribution_table = pd.DataFrame(
        {
            "Predictor": [item["label"] for item in FEATURES],
            "Unit": [item["unit"] for item in FEATURES],
            "Patient value": [result["input_values"][name] for name in FEATURE_NAMES],
            "SHAP contribution": result["shap_values"],
            "Probability change (percentage points)": result["shap_values"] * 100,
        }
    ).sort_values("SHAP contribution", key=np.abs, ascending=False)

    waterfall_tab, force_tab, nightingale_tab, bar_tab, table_tab, guide_tab = result_panel.tabs(
        ["Waterfall", "Force", "Nightingale", "Contribution bars", "Values", "Interpretation"]
    )
    with waterfall_tab:
        st.markdown(
            '<div class="chart-guide"><span><i style="background:#ff0051"></i>increases predicted probability</span>'
            '<span><i style="background:#008bfb"></i>decreases predicted probability</span></div>',
            unsafe_allow_html=True,
        )
        st.image(charts["waterfall"], width="stretch")
        st.caption("The waterfall starts at the fixed SHAP reference and accumulates all seven contributions to reach this prediction.")
    with force_tab:
        st.image(charts["force"], width="stretch")
        st.caption("The force plot shows the competing upward and downward contributions around the reference probability.")
    with nightingale_tab:
        st.image(charts["nightingale"], width="stretch")
        st.caption("This is an individual, patient-level Nightingale chart. Bar length represents absolute SHAP magnitude; colour represents direction.")
    with bar_tab:
        st.image(charts["bar"], width="stretch")
        st.caption("Signed bars provide a direct comparison of the direction and size of all seven patient-level contributions.")
    with table_tab:
        st.dataframe(
            contribution_table.style.format(
                {
                    "Patient value": "{:.4f}",
                    "SHAP contribution": "{:+.6f}",
                    "Probability change (percentage points)": "{:+.4f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    with guide_tab:
        st.markdown(
            """<div class="method-note"><b>Outcome.</b> Class 1 denotes LDAR ≥ 5.27; class 0 denotes LDAR &lt; 5.27.
            The value 5.27 defines the clinical outcome and is not a probability threshold.<br><br>
            <b>Prediction.</b> The displayed percentage is the model's estimated probability for class 1.
            The SVM class is taken from the saved model's decision rule, so it need not switch at exactly 50%.<br><br>
            <b>SHAP reference.</b> Because only the model and scaler are deployed, the fixed reference is the profile located at
            zero in standardised feature space—the training mean for every predictor. E[f(X)] is therefore common to all patients.<br><br>
            <b>Nightingale chart.</b> This chart describes the current patient's absolute SHAP contribution profile.
            It is not a global feature-importance analysis and does not imply causality.</div>""",
            unsafe_allow_html=True,
        )

    export_table = contribution_table.copy()
    export_table["Predicted outcome"] = OUTCOME_TEXT
    export_table["Predicted probability"] = result["probability"]
    export_table["Predicted class"] = result["predicted_class"]
    export_table["SHAP reference probability"] = result["baseline"]
    chart_zip = build_chart_zip(charts)
    download_1, download_2 = result_panel.columns(2)
    download_1.download_button(
        "Download patient result (CSV)",
        export_table.to_csv(index=False).encode("utf-8-sig"),
        file_name="LDAR_patient_prediction.csv",
        mime="text/csv",
        width="stretch",
    )
    download_2.download_button(
        "Download all explanation charts (ZIP)",
        chart_zip,
        file_name="LDAR_SHAP_charts.zip",
        mime="application/zip",
        width="stretch",
    )

st.markdown(
    """<div class="page-footer"><span>LDAR Risk Intelligence · SVM + Kernel SHAP</span>
    <span>Research-use decision support · not a substitute for clinical judgement</span></div>""",
    unsafe_allow_html=True,
)
