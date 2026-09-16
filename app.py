from io import BytesIO
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ============================================================
# 1. 页面与模型配置
# ============================================================
st.set_page_config(
    page_title="LDAR风险预测",
    page_icon="📊",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "svm_model.pkl"
SCALER_PATH = APP_DIR / "svm_scaler.pkl"

# 顺序必须与模型训练时完全一致。
FEATURES = [
    {
        "name": "Age",
        "label": "年龄 Age",
        "default": 68.0,
        "step": 1.0,
        "min": 26.58,
        "max": 93.33,
    },
    {
        "name": "LDH",
        "label": "乳酸脱氢酶 LDH",
        "default": 178.0,
        "step": 1.0,
        "min": 98.0,
        "max": 2159.0,
    },
    {
        "name": "FDP",
        "label": "纤维蛋白降解产物 FDP",
        "default": 1.26,
        "step": 0.01,
        "min": 0.0,
        "max": 67.43,
    },
    {
        "name": "CA125",
        "label": "糖类抗原 CA125",
        "default": 9.60,
        "step": 0.10,
        "min": 2.2,
        "max": 2055.9,
    },
    {
        "name": "CEA",
        "label": "癌胚抗原 CEA",
        "default": 3.80,
        "step": 0.10,
        "min": 0.08,
        "max": 689.03,
    },
    {
        "name": "ALB",
        "label": "白蛋白 ALB",
        "default": 39.30,
        "step": 0.10,
        "min": 21.7,
        "max": 53.2,
    },
    {
        "name": "CA199",
        "label": "糖类抗原 CA19-9",
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
# 2. 加载并核对模型
# ============================================================
@st.cache_resource
def load_model_files():
    missing = [path.name for path in (MODEL_PATH, SCALER_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "GitHub仓库根目录缺少文件：" + "、".join(missing)
        )

    # 只加载自己训练并确认可信的pkl文件。
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    with SCALER_PATH.open("rb") as file:
        scaler = pickle.load(file)

    if not hasattr(model, "predict_proba"):
        raise ValueError("当前SVM模型没有predict_proba方法，训练时需要启用probability=True。")

    if getattr(model, "n_features_in_", None) != len(FEATURE_NAMES):
        raise ValueError("模型需要的指标数量不是7个。")

    scaler_names = getattr(scaler, "feature_names_in_", None)
    if scaler_names is not None and list(scaler_names) != FEATURE_NAMES:
        raise ValueError(
            "标准化器的指标顺序与网页配置不一致："
            f"标准化器为 {list(scaler_names)}，网页为 {FEATURE_NAMES}。"
        )

    classes = np.asarray(getattr(model, "classes_", []))
    positive_positions = np.flatnonzero(classes == POSITIVE_CLASS)
    if len(positive_positions) != 1:
        raise ValueError(f"模型类别 {classes.tolist()} 中找不到阳性类别1。")

    return model, scaler, int(positive_positions[0])


try:
    model, scaler, positive_index = load_model_files()
except Exception as error:
    st.error(f"应用加载失败：{error}")
    st.stop()


# ============================================================
# 3. 预测与SHAP解释
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
        raise ValueError("7个指标必须全部为有限数值。")
    if (patient.to_numpy() < 0).any():
        raise ValueError("指标值不能为负数。")

    standardized_patient = scaler.transform(patient)
    probability = float(positive_probability(standardized_patient)[0])
    predicted_class = int(model.predict(standardized_patient)[0])

    # StandardScaler转换后的0代表训练集各指标均值。
    # 以该“平均患者”作为固定SHAP背景，因此所有患者的E[f(X)]相同。
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
        raise ValueError(f"SHAP结果维度异常：{shap_values.shape}")
    if not np.isclose(baseline + shap_values.sum(), probability, atol=1e-6):
        raise ValueError("SHAP贡献之和与模型预测概率不一致。")

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


def make_waterfall_plot(result):
    explanation = shap.Explanation(
        values=result["shap_values"],
        base_values=result["baseline"],
        data=np.array([result["input_values"][name] for name in FEATURE_NAMES]),
        feature_names=DISPLAY_NAMES,
    )

    plt.figure()
    try:
        shap.plots.waterfall(explanation, max_display=7, show=False)
        plt.title("SHAP contribution to predicted probability", fontsize=13, pad=18)
        image_buffer = BytesIO()
        plt.savefig(
            image_buffer,
            format="png",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
        return image_buffer.getvalue()
    finally:
        plt.close()


# ============================================================
# 4. 网页界面
# ============================================================
st.title("LDAR风险预测与SHAP解释")
st.write("输入7项术前指标，预测患者出现 **LDAR ≥ 5.27** 的概率。")
st.caption("模型结局编码：LDAR ≥ 5.27 为1，LDAR < 5.27 为0。模型：SVM。")

input_column, result_column = st.columns([1, 1.35], gap="large")

with input_column:
    st.subheader("输入患者指标")
    st.caption("输入单位必须与模型训练数据使用的单位完全一致。")

    with st.form("prediction_form"):
        input_values = {}
        for item in FEATURES:
            input_values[item["name"]] = st.number_input(
                item["label"],
                min_value=0.0,
                value=float(item["default"]),
                step=float(item["step"]),
                format="%.4f",
                help=f"训练数据范围：{item['min']:g} 至 {item['max']:g}",
            )

        submitted = st.form_submit_button(
            "开始预测",
            type="primary",
            width="stretch",
        )

if submitted:
    # 先清除上一次结果，防止本次失败后仍显示旧患者结果。
    st.session_state.pop("latest_result", None)
    st.session_state.pop("latest_plot", None)

    try:
        with st.spinner("正在计算预测概率和SHAP解释……"):
            latest_result = calculate_prediction(input_values)
            latest_plot = make_waterfall_plot(latest_result)
            st.session_state["latest_result"] = latest_result
            st.session_state["latest_plot"] = latest_plot
    except Exception as error:
        st.error(f"本次预测失败：{error}")

with result_column:
    st.subheader("预测结果")
    result = st.session_state.get("latest_result")

    if result is None:
        st.info("填写左侧7项指标并点击“开始预测”。")
    else:
        if result["out_of_range"]:
            st.warning(
                "以下指标超出训练数据范围，预测可靠性尚未验证："
                + "、".join(result["out_of_range"])
            )

        metric_1, metric_2 = st.columns(2)
        metric_1.metric(
            "LDAR ≥ 5.27的预测概率",
            f"{result['probability']:.2%}",
        )
        metric_2.metric(
            "模型预测类别",
            "1（LDAR ≥ 5.27）"
            if result["predicted_class"] == 1
            else "0（LDAR < 5.27）",
        )

        st.caption(
            "这里的5.27是定义临床结局的LDAR截断值，不是预测概率阈值。"
        )

        st.subheader("SHAP个体解释")
        st.image(st.session_state["latest_plot"], width="stretch")
        st.caption(
            "红色指标提高LDAR ≥ 5.27的预测概率，蓝色指标降低该概率。"
            "E[f(X)]是训练均值患者的共同参考概率。"
        )

        contribution_table = pd.DataFrame(
            {
                "指标": [item["label"] for item in FEATURES],
                "患者值": [result["input_values"][name] for name in FEATURE_NAMES],
                "SHAP贡献": result["shap_values"],
                "概率变化（百分点）": result["shap_values"] * 100,
            }
        ).sort_values("SHAP贡献", key=np.abs, ascending=False)

        st.dataframe(
            contribution_table.style.format(
                {
                    "患者值": "{:.4f}",
                    "SHAP贡献": "{:+.6f}",
                    "概率变化（百分点）": "{:+.4f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )

        export_table = contribution_table.copy()
        export_table["预测结局"] = OUTCOME_TEXT
        export_table["预测概率"] = result["probability"]
        export_table["预测类别"] = result["predicted_class"]
        export_table["SHAP基准概率"] = result["baseline"]

        download_1, download_2 = st.columns(2)
        download_1.download_button(
            "下载结果CSV",
            export_table.to_csv(index=False).encode("utf-8-sig"),
            file_name="LDAR_prediction_result.csv",
            mime="text/csv",
            width="stretch",
        )
        download_2.download_button(
            "下载SHAP图片",
            st.session_state["latest_plot"],
            file_name="LDAR_SHAP_waterfall.png",
            mime="image/png",
            width="stretch",
        )

st.divider()
st.caption(
    "本工具用于研究展示，不能代替医生判断。请勿将可识别患者身份的信息输入公开网页。"
)

