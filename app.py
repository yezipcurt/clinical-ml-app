from io import BytesIO
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
import shap
import streamlit as st


# ============================================================
# 1. PAGE AND MODEL CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LDAR Risk Prediction",
    page_icon="📊",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "svm_model.pkl"
SCALER_PATH = APP_DIR / "svm_scaler.pkl"

# IMPORTANT:
# The order below must be exactly the same as the feature order
# used when training the model.
FEATURES = [
    {
        "name": "Age",
        "label": "Age",
        "default": 68.0,
        "step": 1.0,
        "min": 26.58,
        "max": 93.33,
    },
    {
        "name": "LDH",
        "label": "Lactate Dehydrogenase (LDH)",
        "default": 178.0,
        "step": 1.0,
        "min": 98.0,
        "max": 2159.0,
    },
    {
        "name": "FDP",
        "label": "Fibrin Degradation Products (FDP)",
        "default": 1.26,
        "step": 0.01,
        "min": 0.0,
        "max": 67.43,
    },
    {
        "name": "CA125",
        "label": "CA125",
        "default": 9.60,
        "step": 0.10,
        "min": 2.2,
        "max": 2055.9,
    },
    {
        "name": "CEA",
        "label": "Carcinoembryonic Antigen (CEA)",
        "default": 3.80,
        "step": 0.10,
        "min": 0.08,
        "max": 689.03,
    },
    {
        "name": "ALB",
        "label": "Albumin (ALB)",
        "default": 39.30,
        "step": 0.10,
        "min": 21.7,
        "max": 53.2,
    },
    {
        "name": "CA199",
        "label": "CA19-9",
        "default": 14.80,
        "step": 0.10,
        "min": 1.0,
        "max": 1868.2,
    },
]

FEATURE_NAMES = [item["name"] for item in FEATURES]
DISPLAY_NAMES = ["CA19-9" if name == "CA199" else name for name in FEATURE_NAMES]

POSITIVE_CLASS = 1
OUTCOME_TEXT = "LDAR ≥ 5.27"


# ============================================================
# 2. LOAD AND VALIDATE MODEL FILES
# ============================================================

@st.cache_resource
def load_model_files():
    missing = [
        path.name
        for path in (MODEL_PATH, SCALER_PATH)
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "The following required file(s) are missing from the app directory: "
            + ", ".join(missing)
        )

    # Only load pickle files that you created and trust.
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    with SCALER_PATH.open("rb") as file:
        scaler = pickle.load(file)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The current SVM model does not provide predict_proba(). "
            "The model must be trained with probability=True."
        )

    model_feature_count = getattr(model, "n_features_in_", None)
    if model_feature_count is not None and model_feature_count != len(FEATURE_NAMES):
        raise ValueError(
            f"The model expects {model_feature_count} features, "
            f"but the web app is configured for {len(FEATURE_NAMES)}."
        )

    scaler_feature_count = getattr(scaler, "n_features_in_", None)
    if scaler_feature_count is not None and scaler_feature_count != len(FEATURE_NAMES):
        raise ValueError(
            f"The scaler expects {scaler_feature_count} features, "
            f"but the web app is configured for {len(FEATURE_NAMES)}."
        )

    scaler_names = getattr(scaler, "feature_names_in_", None)
    if scaler_names is not None and list(scaler_names) != FEATURE_NAMES:
        raise ValueError(
            "Feature order mismatch between the scaler and this app. "
            f"Scaler order: {list(scaler_names)}; "
            f"App order: {FEATURE_NAMES}."
        )

    classes = np.asarray(getattr(model, "classes_", []))
    positive_positions = np.flatnonzero(classes == POSITIVE_CLASS)

    if len(positive_positions) != 1:
        raise ValueError(
            f"Positive class 1 was not found uniquely in model.classes_: "
            f"{classes.tolist()}."
        )

    return model, scaler, int(positive_positions[0])


try:
    model, scaler, positive_index = load_model_files()
except Exception as error:
    st.error(f"Application loading failed: {error}")
    st.stop()


# ============================================================
# 3. PREDICTION AND SHAP CALCULATION
# ============================================================

def positive_probability(standardized_data):
    """Return the probability of the positive class (class 1)."""
    standardized_data = np.asarray(standardized_data, dtype=float)
    return model.predict_proba(standardized_data)[:, positive_index]


def calculate_prediction(input_values):
    """Calculate model prediction and patient-level Kernel SHAP values."""
    patient = pd.DataFrame(
        [[input_values[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
        dtype=float,
    )

    patient_array = patient.to_numpy()

    if not np.isfinite(patient_array).all():
        raise ValueError("All seven predictors must be finite numeric values.")

    if (patient_array < 0).any():
        raise ValueError("Predictor values cannot be negative.")

    standardized_patient = scaler.transform(patient)

    probability = float(
        positive_probability(standardized_patient)[0]
    )

    predicted_class = int(
        model.predict(standardized_patient)[0]
    )

    # For a StandardScaler, zero corresponds to the training-set mean.
    # We use the standardized mean patient as a fixed SHAP background.
    background = np.zeros(
        (1, len(FEATURE_NAMES)),
        dtype=float,
    )

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

    shap_values = np.asarray(
        raw_shap,
        dtype=float,
    ).reshape(-1)

    baseline = float(
        np.asarray(explainer.expected_value).reshape(-1)[0]
    )

    if shap_values.shape != (len(FEATURE_NAMES),):
        raise ValueError(
            f"Unexpected SHAP output shape: {shap_values.shape}"
        )

    # Kernel SHAP with identity link should reconstruct the model probability.
    if not np.isclose(
        baseline + shap_values.sum(),
        probability,
        atol=1e-6,
    ):
        raise ValueError(
            "SHAP contributions do not sum to the predicted probability."
        )

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
        "input_values": input_values,
        "out_of_range": out_of_range,
    }


# ============================================================
# 4. PLOT HELPERS
# ============================================================

def figure_to_png(fig, dpi=190):
    """Convert a Matplotlib figure to PNG bytes."""
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return buffer.getvalue()


def make_waterfall_plot(result):
    """Standard patient-level SHAP waterfall plot."""
    plt.close("all")

    explanation = shap.Explanation(
        values=result["shap_values"],
        base_values=result["baseline"],
        data=np.array(
            [result["input_values"][name] for name in FEATURE_NAMES]
        ),
        feature_names=DISPLAY_NAMES,
    )

    shap.plots.waterfall(
        explanation,
        max_display=len(FEATURE_NAMES),
        show=False,
    )

    fig = plt.gcf()
    fig.set_size_inches(9.4, 5.8)
    fig.patch.set_facecolor("white")

    plt.title(
        "SHAP Waterfall Plot",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    return figure_to_png(fig)


def make_force_plot(result):
    """Static Matplotlib SHAP force plot."""
    plt.close("all")

    patient_values = np.array(
        [result["input_values"][name] for name in FEATURE_NAMES]
    )

    shap.force_plot(
        result["baseline"],
        result["shap_values"],
        patient_values,
        feature_names=DISPLAY_NAMES,
        matplotlib=True,
        show=False,
        contribution_threshold=0.0,
    )

    fig = plt.gcf()
    fig.set_size_inches(12.2, 3.0)
    fig.patch.set_facecolor("white")

    plt.title(
        "SHAP Force Plot",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )

    return figure_to_png(fig)


def make_decision_plot(result):
    """Patient-level SHAP decision plot."""
    plt.close("all")

    patient_values = np.array(
        [result["input_values"][name] for name in FEATURE_NAMES]
    )

    shap.decision_plot(
        result["baseline"],
        result["shap_values"],
        features=patient_values,
        feature_names=DISPLAY_NAMES,
        feature_order="importance",
        link="identity",
        show=False,
    )

    fig = plt.gcf()
    fig.set_size_inches(8.8, 6.2)
    fig.patch.set_facecolor("white")

    plt.title(
        "SHAP Decision Plot",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )

    return figure_to_png(fig)


def make_nightingale_plot(result):
    """
    Nightingale rose chart for patient-level SHAP contributions.

    Sector area represents relative |SHAP contribution|.
    Warm sectors indicate positive contributions.
    Cool sectors indicate negative contributions.

    This is a supplementary visualization, not an official SHAP plot.
    """
    shap_values = np.asarray(
        result["shap_values"],
        dtype=float,
    )

    order = np.argsort(
        np.abs(shap_values)
    )[::-1]

    values = shap_values[order]
    labels = np.array(DISPLAY_NAMES)[order]
    magnitudes = np.abs(values)

    if magnitudes.max() > 0:
        normalized = magnitudes / magnitudes.max()
    else:
        normalized = np.zeros_like(magnitudes)

    # For a rose chart, area is proportional to radius squared.
    radii = np.sqrt(normalized)

    n = len(values)
    theta = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False,
    )

    width = (2 * np.pi / n) * 0.76

    positive_color = "#E35D75"
    negative_color = "#4D7FD8"

    colors = [
        positive_color if value >= 0 else negative_color
        for value in values
    ]

    fig = plt.figure(
        figsize=(8.5, 7.4),
        facecolor="white",
    )

    ax = fig.add_subplot(
        111,
        polar=True,
    )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.bar(
        theta,
        radii,
        width=width,
        bottom=0,
        color=colors,
        alpha=0.90,
        edgecolor="white",
        linewidth=2,
    )

    ax.set_xticks(theta)
    ax.set_xticklabels(
        labels,
        fontsize=10,
        fontweight="semibold",
    )

    ax.set_ylim(0, 1.27)

    ax.set_yticks(
        [0.25, 0.50, 0.75, 1.00]
    )
    ax.set_yticklabels([])

    ax.grid(alpha=0.18)
    ax.spines["polar"].set_visible(False)

    for angle, radius, shap_value in zip(
        theta,
        radii,
        values,
    ):
        contribution_pp = shap_value * 100
        ax.text(
            angle,
            radius + 0.11,
            f"{contribution_pp:+.2f} pp",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="semibold",
        )

    ax.set_title(
        "Nightingale Rose · Individual SHAP Contribution",
        fontsize=14,
        fontweight="bold",
        pad=30,
    )

    legend_items = [
        Patch(
            facecolor=positive_color,
            label="Increases predicted probability",
        ),
        Patch(
            facecolor=negative_color,
            label="Decreases predicted probability",
        ),
    ]

    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    return figure_to_png(fig)


def make_bar_plot(result):
    """Rank patient-level SHAP contributions by absolute magnitude."""
    shap_values = np.asarray(
        result["shap_values"],
        dtype=float,
    )

    order = np.argsort(
        np.abs(shap_values)
    )

    values = shap_values[order]
    labels = np.array(DISPLAY_NAMES)[order]

    positive_color = "#E35D75"
    negative_color = "#4D7FD8"

    colors = [
        positive_color if value >= 0 else negative_color
        for value in values
    ]

    fig, ax = plt.subplots(
        figsize=(8.8, 5.3)
    )

    bars = ax.barh(
        labels,
        values * 100,
        color=colors,
        height=0.62,
        alpha=0.90,
    )

    ax.axvline(
        0,
        color="#AEB7C8",
        linewidth=1,
    )

    max_abs = max(
        float(np.max(np.abs(values * 100))),
        0.01,
    )
    padding = max_abs * 0.04

    for bar, value in zip(
        bars,
        values * 100,
    ):
        if value >= 0:
            x = value + padding
            ha = "left"
        else:
            x = value - padding
            ha = "right"

        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f} pp",
            va="center",
            ha=ha,
            fontsize=9,
            fontweight="semibold",
        )

    ax.set_xlabel(
        "Change in predicted probability (percentage points)",
        fontsize=10,
    )

    ax.set_title(
        "Individual SHAP Contribution Ranking",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.grid(
        axis="x",
        alpha=0.15,
    )

    fig.tight_layout()

    return figure_to_png(fig)


def build_explanation_plots(result):
    """
    Generate each figure independently.

    If one optional plot fails because of a SHAP-version-specific
    rendering issue, the prediction and the remaining plots still work.
    """
    plot_functions = {
        "waterfall": make_waterfall_plot,
        "force": make_force_plot,
        "nightingale": make_nightingale_plot,
        "decision": make_decision_plot,
        "bar": make_bar_plot,
    }

    plots = {}
    plot_errors = {}

    for name, function in plot_functions.items():
        try:
            plots[name] = function(result)
        except Exception as error:
            plots[name] = None
            plot_errors[name] = str(error)

    return plots, plot_errors


# ============================================================
# 5. VISUAL DESIGN
# ============================================================

st.markdown(
    """
<style>
:root {
    color-scheme: light;
}

.stApp {
    background:
        radial-gradient(circle at 8% 4%, rgba(102,92,210,.08), transparent 28%),
        radial-gradient(circle at 96% 20%, rgba(64,181,170,.08), transparent 26%),
        #f5f7fb;
    color: #17243f;
}

[data-testid="stHeader"] {
    background: rgba(245,247,251,.90);
    backdrop-filter: blur(10px);
}

.block-container {
    max-width: 1400px;
    padding: 4.7rem 2.4rem 2.5rem;
}

.stApp p,
.stApp label {
    color: #344567;
}

/* HERO */
.hero {
    position: relative;
    overflow: hidden;
    isolation: isolate;
    padding: 36px 42px;
    border-radius: 26px;
    background: linear-gradient(
        120deg,
        #17284d 0%,
        #333c79 48%,
        #6659c8 100%
    );
    box-shadow: 0 24px 55px rgba(38,51,91,.14);
    margin-bottom: 24px;
}

.hero::before {
    content: "";
    position: absolute;
    z-index: -1;
    width: 360px;
    height: 360px;
    right: -85px;
    top: -155px;
    border-radius: 50%;
    border: 60px solid rgba(255,255,255,.045);
    box-shadow:
        0 0 0 45px rgba(255,255,255,.025),
        0 0 0 95px rgba(255,255,255,.018);
}

.hero::after {
    content: "";
    position: absolute;
    z-index: -1;
    width: 180px;
    height: 180px;
    left: 55%;
    bottom: -145px;
    border-radius: 50%;
    background: rgba(93,224,199,.07);
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #abdfdc;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.2px;
    margin-bottom: 12px;
}

.hero-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #66dcc6;
    box-shadow: 0 0 0 5px rgba(102,220,198,.12);
}

.hero h1 {
    position: relative;
    color: #ffffff !important;
    font-size: clamp(29px, 3.3vw, 43px);
    line-height: 1.22;
    letter-spacing: -1px;
    font-weight: 760;
    margin: 0 0 12px;
    padding: 0;
}

.hero-description {
    position: relative;
    max-width: 790px;
    color: #dce4f5 !important;
    font-size: 14px;
    line-height: 1.9;
    margin: 0 0 22px;
}

.hero-description strong {
    color: #ffffff;
}

.hero-tags {
    position: relative;
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
}

.hero-tag {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 7px 13px;
    font-size: 11px;
    color: #f4f6ff;
    background: rgba(255,255,255,.09);
    border: 1px solid rgba(255,255,255,.12);
}

.hero-tag.primary {
    color: #baf7ea;
    background: rgba(55,201,176,.13);
    border-color: rgba(121,231,210,.21);
}

/* PANELS */
.st-key-input_panel,
.st-key-result_panel,
.st-key-explanation_panel {
    background: rgba(255,255,255,.96);
    border: 1px solid #e7ebf3;
    border-radius: 22px;
    padding: 25px !important;
    box-shadow: 0 9px 30px rgba(36,52,83,.055);
}

.st-key-explanation_panel {
    margin-top: 18px;
}

.section-heading {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}

.step-icon {
    width: 36px;
    height: 36px;
    display: grid;
    place-items: center;
    border-radius: 11px;
    color: #6659c8;
    background: #efedff;
    font-size: 13px;
    font-weight: 750;
    flex-shrink: 0;
}

.step-icon.green {
    color: #168978;
    background: #e7f8f3;
}

.step-icon.blue {
    color: #386dcc;
    background: #edf3ff;
}

.section-heading h2 {
    margin: 0;
    padding: 0;
    font-size: 20px;
    line-height: 1.5;
    color: #1b2a48;
}

.section-subtitle {
    margin: 3px 0 18px 48px;
    font-size: 12px;
    color: #8490a4;
    line-height: 1.8;
}

/* FORM */
[data-testid="stForm"] {
    border: 0 !important;
    padding: 0 !important;
}

[data-testid="stNumberInput"] label p {
    font-size: 12px !important;
    color: #495873;
    font-weight: 650;
}

[data-testid="stNumberInput"] [data-baseweb="input"] {
    min-height: 43px;
    border-radius: 11px;
    background: #f7f8fc;
    border: 1px solid #e4e9f2;
}

[data-testid="stNumberInput"] input {
    background: #f7f8fc !important;
    color: #1e3050 !important;
    font-size: 14px;
}

[data-testid="stNumberInput"] button {
    background: #f7f8fc;
    color: #65728a;
}

[data-testid="stFormSubmitButton"] button {
    margin-top: 7px;
    min-height: 48px;
    border: 0 !important;
    border-radius: 12px;
    color: white !important;
    font-weight: 700;
    background: linear-gradient(
        105deg,
        #5657ca,
        #756ad9
    ) !important;
    box-shadow: 0 8px 20px rgba(93,83,203,.21);
    transition:
        transform .15s ease,
        box-shadow .15s ease,
        filter .15s ease;
}

[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 11px 25px rgba(93,83,203,.27);
    filter: brightness(1.03);
}

[data-testid="stFormSubmitButton"] button p {
    color: #ffffff !important;
}

.field-note {
    margin: 5px 0 10px;
    font-size: 11px;
    color: #939cac;
    line-height: 1.7;
}

.form-note {
    margin-top: 11px;
    padding: 11px 13px;
    border-radius: 11px;
    font-size: 11px;
    line-height: 1.8;
    color: #4d7d75;
    background: #f0f9f6;
    border: 1px solid #dfefe9;
}

/* EMPTY STATE */
.empty-result {
    padding: 34px 22px;
    border-radius: 18px;
    text-align: center;
    background: linear-gradient(
        145deg,
        #f7f7ff,
        #f2fbf9
    );
    border: 1px solid #eeeff7;
}

.empty-ring {
    width: 108px;
    height: 108px;
    margin: 2px auto 19px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    border: 9px solid #e7e7f7;
    border-top-color: #746adc;
    border-right-color: #61cabb;
    color: #737bb6;
    font-size: 30px;
    font-weight: 700;
    box-shadow: 0 9px 26px rgba(74,83,155,.08);
}

.empty-result h3 {
    padding: 0;
    margin: 0 0 7px;
    color: #344361;
    font-size: 18px;
}

.empty-result p {
    margin: 0;
    color: #7c899f !important;
    font-size: 12px;
    line-height: 1.9;
}

.empty-mini-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 13px;
}

.empty-mini-card {
    padding: 13px 10px;
    border-radius: 12px;
    background: #f8f9fd;
    border: 1px solid #eceff5;
    text-align: center;
}

.empty-mini-card strong {
    display: block;
    font-size: 17px;
    color: #5f58b6;
    margin-bottom: 3px;
}

.empty-mini-card span {
    font-size: 10px;
    color: #8791a4;
}

/* PROBABILITY */
.probability-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 22px;
    padding: 23px 25px;
    margin-bottom: 13px;
    border-radius: 18px;
    background: linear-gradient(
        125deg,
        #f4f2ff,
        #f6f9ff
    );
    border: 1px solid #e4e3f8;
}

.probability-label {
    color: #616a85;
    font-size: 12px;
    font-weight: 600;
}

.probability-number {
    margin: 3px 0;
    color: #574bb6;
    font-size: 48px;
    line-height: 1.25;
    font-weight: 780;
    letter-spacing: -2px;
}

.probability-number span {
    font-size: 21px;
    letter-spacing: 0;
}

.probability-sub {
    color: #8b94a8;
    font-size: 10px;
}

.probability-ring {
    width: 98px;
    height: 98px;
    flex-shrink: 0;
    display: grid;
    place-items: center;
    border-radius: 50%;
}

.probability-ring-inner {
    width: 76px;
    height: 76px;
    display: grid;
    place-items: center;
    border-radius: 50%;
    background: #f7f8ff;
    color: #697391;
    font-size: 11px;
    font-weight: 650;
    text-align: center;
    line-height: 1.4;
}

/* RESULT METRICS */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 15px;
}

.metric-card {
    padding: 13px 14px;
    border-radius: 12px;
    background: #f8f9fc;
    border: 1px solid #eaedf3;
}

.metric-card.green {
    background: #eef9f6;
    border-color: #def0ea;
}

.metric-card.purple {
    background: #f6f4fc;
    border-color: #e9e5f5;
}

.metric-card.blue {
    background: #f1f6fd;
    border-color: #e1eaf8;
}

.metric-label {
    font-size: 10px;
    color: #838da0;
}

.metric-value {
    margin-top: 5px;
    font-size: 14px;
    font-weight: 700;
    color: #314360;
    word-break: break-word;
}

.metric-card.green .metric-value {
    color: #277b70;
}

.metric-card.purple .metric-value {
    color: #7162a7;
}

.metric-card.blue .metric-value {
    color: #456fb2;
}

/* DRIVERS */
.driver-title {
    margin: 17px 0 9px;
    color: #384966;
    font-size: 12px;
    font-weight: 700;
}

.driver-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
}

.driver-card {
    min-width: 0;
    padding: 11px 10px;
    border-radius: 11px;
    background: #fafbfe;
    border: 1px solid #eceff5;
}

.driver-name {
    font-size: 11px;
    font-weight: 700;
    color: #47566f;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.driver-value {
    margin-top: 4px;
    font-size: 12px;
    font-weight: 700;
}

.driver-value.up {
    color: #df536c;
}

.driver-value.down {
    color: #4c79d4;
}

/* EXPLANATION SECTION */
.chart-note {
    padding: 10px 13px;
    margin: 8px 0 13px;
    border-radius: 10px;
    background: #f7f9fc;
    color: #77849a;
    font-size: 11px;
    line-height: 1.8;
}

.chart-note strong {
    color: #475673;
}

.chart-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 17px;
    margin: 7px 0 5px;
    color: #727f95;
    font-size: 11px;
}

.legend-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    margin-right: 6px;
    border-radius: 50%;
}

.legend-red {
    background: #e35d75;
}

.legend-blue {
    background: #4d7fd8;
}

/* TABS */
[data-testid="stTabs"] [role="tablist"] {
    gap: 8px;
    border-bottom: 1px solid #ebedf3;
}

[data-testid="stTabs"] button {
    border-radius: 9px 9px 0 0;
}

[data-testid="stTabs"] [role="tab"] p {
    font-size: 12px;
    font-weight: 650;
}

[data-testid="stTabs"] [aria-selected="true"] p {
    color: #6257be !important;
}

/* DOWNLOAD BUTTONS */
[data-testid="stDownloadButton"] button {
    min-height: 42px;
    border-radius: 10px;
    background: #fafaff;
    border: 1px solid #dfdef0;
}

[data-testid="stDownloadButton"] button p {
    color: #655a9c !important;
    font-size: 11px;
}

/* FOOTER */
.footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 28px;
    padding: 19px 3px 0;
    border-top: 1px solid #e5e9f1;
    color: #949daf;
    font-size: 10px;
}

/* MOBILE */
@media (max-width: 760px) {
    .block-container {
        padding: 4.4rem 1rem 1.5rem;
    }

    .hero {
        padding: 28px 23px;
    }

    .hero h1 {
        font-size: 29px;
    }

    .st-key-input_panel,
    .st-key-result_panel,
    .st-key-explanation_panel {
        padding: 18px !important;
    }

    .probability-number {
        font-size: 40px;
    }

    .probability-ring {
        width: 84px;
        height: 84px;
    }

    .probability-ring-inner {
        width: 65px;
        height: 65px;
    }

    .metric-grid,
    .driver-grid,
    .empty-mini-grid {
        grid-template-columns: 1fr;
    }
}
</style>

<section class="hero">
    <div class="hero-kicker">
        <span class="hero-dot"></span>
        INDIVIDUALIZED CLINICAL PREDICTION
    </div>

    <h1>
        LDAR Risk Prediction<br>
        & Individual Interpretation
    </h1>

    <div class="hero-description">
        Estimate the individual probability of
        <strong>LDAR ≥ 5.27</strong>
        using seven preoperative clinical predictors and explore
        how each variable contributes to the prediction with SHAP.
    </div>

    <div class="hero-tags">
        <span class="hero-tag primary">● Individual Prediction</span>
        <span class="hero-tag">7 Predictors</span>
        <span class="hero-tag">SVM</span>
        <span class="hero-tag">Kernel SHAP</span>
        <span class="hero-tag">Explainable AI</span>
    </div>
</section>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 6. INPUT PANEL
# ============================================================

input_column, result_column = st.columns(
    [0.90, 1.32],
    gap="medium",
)

with input_column:
    with st.container(key="input_panel"):
        st.markdown(
            """
            <div class="section-heading">
                <span class="step-icon">01</span>
                <h2>Patient Clinical Data</h2>
            </div>
            <div class="section-subtitle">
                Enter the seven preoperative predictors using the same units
                as those used in model development.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("prediction_form"):
            input_values = {}

            for row in range(0, len(FEATURES), 2):
                fields = st.columns(2, gap="small")

                for offset, item in enumerate(
                    FEATURES[row:row + 2]
                ):
                    with fields[offset]:
                        input_values[item["name"]] = st.number_input(
                            item["label"],
                            min_value=0.0,
                            value=float(item["default"]),
                            step=float(item["step"]),
                            format="%.2f",
                            key=item["name"],
                            help=(
                                f"Range observed in the training data: "
                                f"{item['min']:g} to {item['max']:g}"
                            ),
                        )

            st.markdown(
                """
                <div class="field-note">
                    The prefilled values are demonstration values only.
                    Replace them with the current patient's actual measurements.
                </div>
                """,
                unsafe_allow_html=True,
            )

            submitted = st.form_submit_button(
                "Generate Prediction  →",
                type="primary",
                width="stretch",
            )

        st.markdown(
            """
            <div class="form-note">
                After submission, the app will generate the predicted probability
                together with Waterfall, Force, Nightingale Rose, Decision,
                and contribution-ranking visualizations.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "If any input is changed, click Generate Prediction again "
            "to update the result."
        )


# ============================================================
# 7. RUN PREDICTION
# ============================================================

if submitted:
    st.session_state.pop("latest_result", None)
    st.session_state.pop("latest_plots", None)
    st.session_state.pop("plot_errors", None)

    try:
        with st.spinner(
            "Calculating prediction probability and SHAP explanations..."
        ):
            latest_result = calculate_prediction(
                input_values
            )

            latest_plots, plot_errors = build_explanation_plots(
                latest_result
            )

            st.session_state["latest_result"] = latest_result
            st.session_state["latest_plots"] = latest_plots
            st.session_state["plot_errors"] = plot_errors

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ============================================================
# 8. RESULT PANEL
# ============================================================

with result_column:
    with st.container(key="result_panel"):
        st.markdown(
            """
            <div class="section-heading">
                <span class="step-icon green">02</span>
                <h2>Individual Prediction</h2>
            </div>
            <div class="section-subtitle">
                Review the predicted probability and the strongest
                patient-specific model drivers.
            </div>
            """,
            unsafe_allow_html=True,
        )

        result = st.session_state.get(
            "latest_result"
        )

        if result is None:
            st.markdown(
                """
                <div class="empty-result">
                    <div class="empty-ring">—</div>
                    <h3>Ready for prediction</h3>
                    <p>
                        Enter the patient's preoperative values on the left
                        and click <strong>Generate Prediction</strong>.<br>
                        The model result and individual explanation will appear here.
                    </p>
                </div>

                <div class="empty-mini-grid">
                    <div class="empty-mini-card">
                        <strong>7</strong>
                        <span>Input predictors</span>
                    </div>
                    <div class="empty-mini-card">
                        <strong>5.27</strong>
                        <span>LDAR outcome cut-off</span>
                    </div>
                    <div class="empty-mini-card">
                        <strong>SHAP</strong>
                        <span>Individual explanation</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="chart-note">
                    <strong>Outcome definition:</strong>
                    LDAR ≥ 5.27 is coded as 1 and LDAR &lt; 5.27 is coded as 0.
                    The value 5.27 is the LDAR outcome cut-off, not a predicted
                    probability threshold.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            probability = result["probability"]
            probability_percent = probability * 100
            baseline_percent = result["baseline"] * 100
            delta_pp = (
                result["probability"] - result["baseline"]
            ) * 100

            if result["predicted_class"] == 1:
                class_text = "1 · LDAR ≥ 5.27"
            else:
                class_text = "0 · LDAR &lt; 5.27"

            st.markdown(
                f"""
                <div class="probability-card">
                    <div>
                        <div class="probability-label">
                            Predicted probability of LDAR ≥ 5.27
                        </div>

                        <div class="probability-number">
                            {probability_percent:.2f}<span>%</span>
                        </div>

                        <div class="probability-sub">
                            Individual prediction based on seven preoperative predictors
                        </div>
                    </div>

                    <div
                        class="probability-ring"
                        style="
                            background: conic-gradient(
                                #7064d5 0% {probability_percent:.5f}%,
                                #e5e7f4 {probability_percent:.5f}% 100%
                            );
                        "
                    >
                        <div class="probability-ring-inner">
                            Predicted<br>probability
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            delta_symbol = "+" if delta_pp >= 0 else ""

            st.markdown(
                f"""
                <div class="metric-grid">
                    <div class="metric-card green">
                        <div class="metric-label">Model output</div>
                        <div class="metric-value">{class_text}</div>
                    </div>

                    <div class="metric-card purple">
                        <div class="metric-label">SHAP baseline</div>
                        <div class="metric-value">{baseline_percent:.2f}%</div>
                    </div>

                    <div class="metric-card blue">
                        <div class="metric-label">Difference from baseline</div>
                        <div class="metric-value">
                            {delta_symbol}{delta_pp:.2f} pp
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            abs_order = np.argsort(
                np.abs(result["shap_values"])
            )[::-1]

            top_three = abs_order[:3]
            driver_html = ""

            for index in top_three:
                name = DISPLAY_NAMES[index]

                contribution = (
                    result["shap_values"][index] * 100
                )

                patient_value = result[
                    "input_values"
                ][FEATURE_NAMES[index]]

                if contribution >= 0:
                    css_class = "up"
                    direction = "↑"
                else:
                    css_class = "down"
                    direction = "↓"

                driver_html += f"""
                <div class="driver-card">
                    <div class="driver-name">
                        {name} · {patient_value:g}
                    </div>
                    <div class="driver-value {css_class}">
                        {direction} {contribution:+.2f} pp
                    </div>
                </div>
                """

            st.markdown(
                f"""
                <div class="driver-title">
                    Top Individual Drivers
                </div>

                <div class="driver-grid">
                    {driver_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if result["out_of_range"]:
                st.warning(
                    "The following predictor(s) are outside the range "
                    "observed in the training data: "
                    + ", ".join(result["out_of_range"])
                )


# ============================================================
# 9. FULL-WIDTH INDIVIDUAL EXPLANATION SECTION
# ============================================================

result = st.session_state.get(
    "latest_result"
)

if result is not None:
    plots = st.session_state.get(
        "latest_plots",
        {},
    )

    plot_errors = st.session_state.get(
        "plot_errors",
        {},
    )

    with st.container(key="explanation_panel"):
        st.markdown(
            """
            <div class="section-heading">
                <span class="step-icon blue">03</span>
                <h2>Individual Model Explanation</h2>
            </div>

            <div class="section-subtitle">
                Explore how the seven predictors jointly shape
                the current patient's model output.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="chart-legend">
                <span>
                    <i class="legend-dot legend-red"></i>
                    Increases the predicted probability
                </span>
                <span>
                    <i class="legend-dot legend-blue"></i>
                    Decreases the predicted probability
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        (
            tab_waterfall,
            tab_force,
            tab_nightingale,
            tab_decision,
            tab_bar,
            tab_table,
            tab_help,
        ) = st.tabs(
            [
                "Waterfall",
                "Force",
                "Nightingale",
                "Decision",
                "Contribution",
                "Patient Data",
                "How to Read",
            ]
        )

        # ----------------------------------------------------
        # WATERFALL
        # ----------------------------------------------------
        with tab_waterfall:
            st.markdown(
                """
                <div class="chart-note">
                    <strong>SHAP Waterfall Plot</strong><br>
                    Starting from the model's baseline probability, this plot
                    shows how each predictor pushes the prediction upward or
                    downward until the patient's final predicted probability
                    is reached.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if plots.get("waterfall") is not None:
                st.image(
                    plots["waterfall"],
                    width="stretch",
                )

                st.download_button(
                    "↓ Download Waterfall Plot",
                    plots["waterfall"],
                    file_name="LDAR_SHAP_waterfall.png",
                    mime="image/png",
                    key="download_waterfall",
                )
            else:
                st.warning(
                    "Waterfall plot could not be generated: "
                    + plot_errors.get(
                        "waterfall",
                        "Unknown rendering error.",
                    )
                )

        # ----------------------------------------------------
        # FORCE
        # ----------------------------------------------------
        with tab_force:
            st.markdown(
                """
                <div class="chart-note">
                    <strong>SHAP Force Plot</strong><br>
                    Predictors on one side push the model output toward a higher
                    probability, while predictors on the opposite side push it
                    toward a lower probability. Together they move the prediction
                    from the baseline value to the final patient-specific value.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if plots.get("force") is not None:
                st.image(
                    plots["force"],
                    width="stretch",
                )

                st.download_button(
                    "↓ Download Force Plot",
                    plots["force"],
                    file_name="LDAR_SHAP_force.png",
                    mime="image/png",
                    key="download_force",
                )
            else:
                st.warning(
                    "Force plot could not be generated: "
                    + plot_errors.get(
                        "force",
                        "Unknown rendering error.",
                    )
                )

        # ----------------------------------------------------
        # NIGHTINGALE
        # ----------------------------------------------------
        with tab_nightingale:
            st.markdown(
                """
                <div class="chart-note">
                    <strong>Nightingale Rose Plot</strong><br>
                    Sector area reflects the relative magnitude of the absolute
                    SHAP contribution. The direction is shown by sector color.
                    This is a supplementary visualization for presentation
                    purposes and is not an official SHAP plotting method.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if plots.get("nightingale") is not None:
                st.image(
                    plots["nightingale"],
                    width="stretch",
                )

                st.download_button(
                    "↓ Download Nightingale Plot",
                    plots["nightingale"],
                    file_name="LDAR_Nightingale_SHAP.png",
                    mime="image/png",
                    key="download_nightingale",
                )
            else:
                st.warning(
                    "Nightingale plot could not be generated: "
                    + plot_errors.get(
                        "nightingale",
                        "Unknown rendering error.",
                    )
                )

        # ----------------------------------------------------
        # DECISION
        # ----------------------------------------------------
        with tab_decision:
            st.markdown(
                """
                <div class="chart-note">
                    <strong>SHAP Decision Plot</strong><br>
                    This plot traces the cumulative contribution of each predictor
                    from the SHAP baseline to the patient's final model output.
                    It is useful for visualizing the step-by-step formation of
                    the prediction.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if plots.get("decision") is not None:
                st.image(
                    plots["decision"],
                    width="stretch",
                )

                st.download_button(
                    "↓ Download Decision Plot",
                    plots["decision"],
                    file_name="LDAR_SHAP_decision.png",
                    mime="image/png",
                    key="download_decision",
                )
            else:
                st.warning(
                    "Decision plot could not be generated: "
                    + plot_errors.get(
                        "decision",
                        "Unknown rendering error.",
                    )
                )

        # ----------------------------------------------------
        # CONTRIBUTION BAR
        # ----------------------------------------------------
        with tab_bar:
            st.markdown(
                """
                <div class="chart-note">
                    <strong>Individual SHAP Contribution Ranking</strong><br>
                    Predictors are ranked by their absolute patient-specific
                    SHAP contribution. The horizontal axis is expressed in
                    percentage-point change in predicted probability.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if plots.get("bar") is not None:
                st.image(
                    plots["bar"],
                    width="stretch",
                )

                st.download_button(
                    "↓ Download Contribution Plot",
                    plots["bar"],
                    file_name="LDAR_SHAP_contribution.png",
                    mime="image/png",
                    key="download_bar",
                )
            else:
                st.warning(
                    "Contribution plot could not be generated: "
                    + plot_errors.get(
                        "bar",
                        "Unknown rendering error.",
                    )
                )

        # ----------------------------------------------------
        # PATIENT DATA TABLE
        # ----------------------------------------------------
        contribution_table = pd.DataFrame(
            {
                "Predictor": DISPLAY_NAMES,
                "Patient Value": [
                    result["input_values"][name]
                    for name in FEATURE_NAMES
                ],
                "SHAP Contribution": result["shap_values"],
                "Probability Change (pp)": (
                    result["shap_values"] * 100
                ),
            }
        ).sort_values(
            "SHAP Contribution",
            key=np.abs,
            ascending=False,
        )

        with tab_table:
            st.markdown(
                """
                <div class="chart-note">
                    This table reports the patient's input value for each predictor,
                    the corresponding SHAP contribution, and the equivalent change
                    in predicted probability expressed in percentage points.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.dataframe(
                contribution_table.style.format(
                    {
                        "Patient Value": "{:.4f}",
                        "SHAP Contribution": "{:+.6f}",
                        "Probability Change (pp)": "{:+.4f}",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

        # ----------------------------------------------------
        # HOW TO READ
        # ----------------------------------------------------
        with tab_help:
            st.markdown(
                """
                **Predicted probability**  
                This is the SVM model's estimated probability that the patient
                belongs to the outcome group defined as **LDAR ≥ 5.27**.

                **Model output**  
                The displayed class is taken directly from the SVM classifier.
                Because the SVM probability estimates may be calibrated separately,
                the class boundary and a simple 50% probability threshold do not
                necessarily have to be identical.

                **SHAP baseline**  
                The baseline is the model output for the fixed SHAP reference
                patient used in this application. Here, the reference is the
                standardized mean patient.

                **SHAP contribution**  
                A positive SHAP value pushes the prediction toward a higher
                probability of LDAR ≥ 5.27. A negative SHAP value pushes it
                toward a lower probability.

                **Important**  
                SHAP explains how this trained model forms its prediction.
                SHAP contributions are not evidence of a causal clinical effect.
                The value **5.27** is the LDAR outcome cut-off, not a probability
                cut-off.
                """
            )

        # ----------------------------------------------------
        # EXPORT
        # ----------------------------------------------------
        export_table = contribution_table.copy()
        export_table["Outcome Definition"] = OUTCOME_TEXT
        export_table["Predicted Probability"] = result["probability"]
        export_table["Predicted Class"] = result["predicted_class"]
        export_table["SHAP Baseline"] = result["baseline"]

        st.markdown(
            """
            <div class="chart-note">
                <strong>Export result</strong><br>
                Download the complete patient-level prediction and SHAP
                contribution table as a CSV file.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            "↓ Download Complete Prediction Result (CSV)",
            export_table.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="LDAR_prediction_result.csv",
            mime="text/csv",
            width="stretch",
            key="download_csv",
        )


# ============================================================
# 10. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <span>
            LDAR Individual Risk Prediction · Explainable Machine Learning
        </span>
        <span>
            Research use only · Results should be interpreted together with
            relevant clinical information
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
