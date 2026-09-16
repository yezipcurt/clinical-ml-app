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
# 4. 网页界面：卡片、渐变配色与响应式排版
# ============================================================
st.markdown("""
<style>
:root {color-scheme:light;}
.stApp {background:radial-gradient(ellipse at 5% 18%,#edf2ff 0,transparent 45%),
    radial-gradient(ellipse at 100% 75%,#e9f8f5 0,transparent 42%),#f6f8fc;color:#1b2947;}
[data-testid="stHeader"] {background:rgba(246,248,252,.92);}
.block-container {max-width:1420px;padding:5rem 2.2rem 2rem;}
.stApp p,.stApp label {color:#344567;}
.hero {position:relative;overflow:hidden;isolation:isolate;padding:32px 38px;
    border-radius:24px;background:linear-gradient(115deg,#172b58 0%,#3e4388 54%,#5d53ae 100%);
    box-shadow:0 16px 42px #283e7418;margin:0 0 22px;}
.hero:after {content:"";position:absolute;z-index:-1;width:330px;height:330px;right:-50px;top:-135px;
    border-radius:50%;border:52px solid #ffffff09;box-shadow:0 0 0 45px #ffffff04;}
.eyebrow {color:#a5dcf0;font:600 11px/1.5 sans-serif;letter-spacing:2.7px;margin-bottom:10px;}
.hero h1 {color:#fff!important;font-size:clamp(26px,3vw,38px);font-weight:750;letter-spacing:-.6px;
    line-height:1.35;margin:0 0 10px;padding:0;}
.hero p {color:#dce3f5!important;font-size:14px;line-height:1.9;margin:0 0 18px;}
.chips {display:flex;flex-wrap:wrap;gap:9px;}
.chip {font-size:12px;border-radius:30px;padding:6px 13px;background:#ffffff12;border:1px solid #ffffff20;color:#f2f4ff;}
.chip.teal {background:#1ab5a32a;border-color:#61e7cf38;color:#b0ffeb;}
.st-key-input_panel,.st-key-result_panel {background:#fff;border:1px solid #e8edf6;
    border-radius:22px;padding:24px!important;box-shadow:0 8px 28px #263d6a07;}
.panel-heading {display:flex;align-items:center;gap:12px;margin-bottom:5px;}
.step-icon {width:35px;height:35px;display:grid;place-items:center;border-radius:11px;
    font:700 14px sans-serif;color:#6354c1;background:#eeebff;flex-shrink:0;}
.step-icon.teal {color:#138674;background:#e0f6f0;}
.panel-heading h2 {margin:0;padding:0;font-size:20px;line-height:1.5;color:#1c2c4c;}
.panel-subtitle {font-size:12px;line-height:1.8;color:#75829a;margin:3px 0 15px;}
[data-testid="stForm"] {border:0!important;padding:0!important;}
[data-testid="stNumberInput"] label p {font-size:12px!important;font-weight:600;color:#4b5872;}
[data-testid="stNumberInput"] [data-baseweb="input"] {background:#f5f7fc;border-radius:10px;border:1px solid #e5eaf4;}
[data-testid="stNumberInput"] input {color:#1d3054!important;font-size:15px;background:#f5f7fc;}
[data-testid="stNumberInput"] button {color:#65718a;background:#f5f7fc;}
[data-testid="stFormSubmitButton"] button {border:0!important;color:#fff!important;min-height:46px;
    border-radius:12px;background:linear-gradient(105deg,#5456cf,#7770df)!important;
    box-shadow:0 5px 14px #665dd52b;font-weight:650;transition:filter .2s;}
[data-testid="stFormSubmitButton"] button p {color:#fff!important;}
[data-testid="stFormSubmitButton"] button:hover {filter:brightness(1.08);}
.field-note {font-size:11px;color:#8c96a8;line-height:1.8;margin:4px 0 9px;}
.form-note {border-radius:12px;background:#f0f8f8;border:1px solid #dff0ec;padding:10px 13px;
    color:#4c7c77;font-size:12px;line-height:1.8;margin-top:9px;}
.empty {text-align:center;border-radius:18px;background:linear-gradient(150deg,#f7f7ff,#effaf9);
    padding:27px 18px 25px;margin:5px 0 18px;}
.empty-ring {width:112px;height:112px;margin:0 auto 18px;border:9px solid #e4e5fa;
    border-top-color:#8a7fe3;border-right-color:#67c9ba;border-radius:50%;display:grid;place-items:center;
    color:#7476ba;font-size:30px;font-weight:700;box-shadow:0 8px 24px #6463b610;}
.empty h3 {font-size:18px;margin:0 0 8px;color:#344564;padding:0;}
.empty p {font-size:13px;color:#7b86a0!important;margin:0;line-height:1.9;}
.mini-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}
.mini {padding:16px 10px;border-radius:14px;background:#f4f1fe;text-align:center;}
.mini:nth-child(2) {background:#eaf8f4;}.mini:nth-child(3) {background:#fff4ec;}
.mini strong {display:block;font-size:20px;color:#6755b9;line-height:1.5;}
.mini:nth-child(2) strong {color:#168776;}.mini:nth-child(3) strong {color:#b47739;}
.mini span {font-size:11px;color:#6a7890;}
.prob-card {display:flex;align-items:center;justify-content:space-between;gap:18px;padding:22px 24px;
    border:1px solid #e3e2f8;background:linear-gradient(125deg,#f3f1ff,#f4faff);border-radius:18px;margin:6px 0 13px;}
.prob-label {font-size:13px;color:#5a6084;}.prob-value {font-size:48px;font-weight:750;color:#5347b5;line-height:1.35;
    letter-spacing:-1.5px;}.prob-value span {font-size:23px;margin-left:3px;}
.prob-foot {font-size:11px;color:#8992aa;}
.prob-ring {flex-shrink:0;width:90px;height:90px;border-radius:50%;display:grid;place-items:center;}
.prob-ring-inner {width:71px;height:71px;border-radius:50%;background:#f6f7ff;display:grid;place-items:center;
    font-size:13px;font-weight:600;color:#697295;}
.secondary-grid {display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:10px;}
.stat-card {background:#eff8f5;border:1px solid #def0e8;border-radius:13px;padding:13px 16px;}
.stat-card.purple {background:#f7f5fd;border-color:#eae5f7;}
.stat-card .label {font-size:11px;color:#7b8799;}.stat-card .value {font-size:16px;font-weight:650;color:#287b70;margin-top:5px;}
.stat-card.purple .value {color:#7262a8;}
.legend {display:flex;gap:20px;flex-wrap:wrap;margin:10px 0;font-size:11px;color:#718099;}
.legend i {display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;}
.info-box {border-radius:12px;background:#f7f9fd;padding:12px 15px;margin:14px 0 0;color:#79869f;font-size:12px;line-height:1.9;}
[data-testid="stTabs"] [role="tablist"] {gap:20px;border-bottom:1px solid #edf0f6;}
[data-testid="stTabs"] [role="tab"] p {font-size:13px;font-weight:600;}
[data-testid="stTabs"] [aria-selected="true"] p {color:#6755c4!important;}
[data-testid="stDownloadButton"] button {border:1px solid #dedcf1;border-radius:10px;background:#faf9ff;min-height:40px;}
[data-testid="stDownloadButton"] button p {color:#675998!important;font-size:12px;}
[data-testid="stCaptionContainer"] p {font-size:11px!important;line-height:1.8;color:#8993a6;}
.footer {display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap;
    color:#909ab0;font-size:11px;padding:20px 4px 0;border-top:1px solid #e5eaf2;margin-top:25px;}
@media(max-width:760px) {
    .block-container {padding:4.5rem 1rem 1.6rem;}.hero {padding:25px 22px;}.hero h1 {font-size:27px;}
    .st-key-input_panel,.st-key-result_panel {padding:18px!important;}.prob-value {font-size:40px;}
    .prob-card {padding:18px;}.mini-grid {gap:7px;}.chip {font-size:11px;padding:5px 10px;}
}
</style>
<section class="hero">
<div class="eyebrow">LDAR · INDIVIDUAL RISK ASSESSMENT</div>
<h1>LDAR 风险预测与解释</h1>
<p>通过 7 项临床指标，评估 LDAR ≥ 5.27 的概率，并查看各指标对本次预测的贡献。</p>
<div class="chips"><span class="chip teal">● LDAR 风险预测</span>
<span class="chip">7 项指标</span><span class="chip">SVM 模型</span>
<span class="chip">SHAP 个体解释</span></div>
</section>
""", unsafe_allow_html=True)

input_column, result_column = st.columns([1, 1.4], gap="medium")
with input_column, st.container(key="input_panel"):
    st.markdown('<div class="panel-heading"><span class="step-icon">01</span>'
                '<h2>输入患者指标</h2></div><div class="panel-subtitle">'
                '请使用与模型训练数据一致的单位。</div>', unsafe_allow_html=True)
    with st.form("prediction_form"):
        input_values = {}
        for row in range(0, len(FEATURES), 2):
            fields = st.columns(2, gap="small")
            for offset, item in enumerate(FEATURES[row:row + 2]):
                with fields[offset]:
                    input_values[item["name"]] = st.number_input(
                        item["label"], min_value=0.0, value=float(item["default"]),
                        step=float(item["step"]), format="%.2f", key=item["name"],
                        help=f"训练数据范围：{item['min']:g} 至 {item['max']:g}",
                    )
        st.markdown('<div class="field-note">预填数值为演示示例，请替换为当前患者的实际指标。</div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("开始预测  →", type="primary", width="stretch")
    st.markdown('<div class="form-note">每次提交后，将同步生成预测概率与 7 个指标的贡献解释。</div>', unsafe_allow_html=True)
    st.caption("修改输入后请重新计算；结果区保留最近一次提交的结果。")

if submitted:
    st.session_state.pop("latest_result", None)
    st.session_state.pop("latest_plot", None)
    try:
        with st.spinner("正在计算预测概率和 SHAP 解释……"):
            latest_result = calculate_prediction(input_values)
            latest_plot = make_waterfall_plot(latest_result)
            st.session_state["latest_result"] = latest_result
            st.session_state["latest_plot"] = latest_plot
    except Exception as error:
        st.error(f"本次预测失败：{error}")

with result_column, st.container(key="result_panel"):
    st.markdown('<div class="panel-heading"><span class="step-icon teal">02</span>'
                '<h2>预测与解释</h2></div><div class="panel-subtitle">'
                '查看个体预测，以及每一项指标的影响。</div>', unsafe_allow_html=True)
    result = st.session_state.get("latest_result")
    if result is None:
        st.markdown('''<div class="empty"><div class="empty-ring">—</div>
<h3>准备好，了解本次预测</h3><p>填写患者指标后，点击「开始预测」。<br>
预测概率与 SHAP 贡献图将在这里呈现。</p></div>
<div class="mini-grid"><div class="mini"><strong>7</strong><span>输入指标</span></div>
<div class="mini"><strong>5.27</strong><span>LDAR 结局截断值</span></div>
<div class="mini"><strong>SHAP</strong><span>逐项贡献解释</span></div></div>
<div class="info-box">结局定义：LDAR ≥ 5.27 为 1，LDAR &lt; 5.27 为 0。<br>
这里的 5.27 是 LDAR 的截断值，不是预测概率的分界线。</div>''', unsafe_allow_html=True)
    else:
        probability_percent = result["probability"] * 100
        class_text = "1 · LDAR ≥ 5.27" if result["predicted_class"] == 1 else "0 · LDAR < 5.27"
        st.markdown(f'''<div class="prob-card"><div><div class="prob-label">LDAR ≥ 5.27 的预测概率</div>
<div class="prob-value">{probability_percent:.2f}<span>%</span></div>
<div class="prob-foot">基于最近一次提交的 7 项指标</div></div>
<div class="prob-ring" style="background:conic-gradient(#7770dc 0% {probability_percent:.5f}%,#e3e5f5 {probability_percent:.5f}% 100%)">
<div class="prob-ring-inner">预测概率</div></div></div>
<div class="secondary-grid"><div class="stat-card"><div class="label">模型预测类别</div>
<div class="value">{class_text.replace('<', '&lt;')}</div></div>
<div class="stat-card purple"><div class="label">SHAP 参考概率</div>
<div class="value">{result['baseline']:.2%}</div></div></div>''', unsafe_allow_html=True)
        if result["out_of_range"]:
            st.warning("以下指标超出训练数据范围：" + "、".join(result["out_of_range"]))

        chart_tab, table_tab, notes_tab = st.tabs(["贡献瀑布图", "指标明细", "如何阅读结果"])
        contribution_table = pd.DataFrame({
            "指标": [item["label"] for item in FEATURES],
            "患者值": [result["input_values"][name] for name in FEATURE_NAMES],
            "SHAP贡献": result["shap_values"],
            "概率变化（百分点）": result["shap_values"] * 100,
        }).sort_values("SHAP贡献", key=np.abs, ascending=False)
        with chart_tab:
            st.markdown('<div class="legend"><span><i style="background:#ff0051"></i>提高预测概率</span>'
                        '<span><i style="background:#008bfb"></i>降低预测概率</span></div>', unsafe_allow_html=True)
            st.image(st.session_state["latest_plot"], width="stretch")
        with table_tab:
            st.dataframe(contribution_table.style.format({
                "患者值": "{:.4f}", "SHAP贡献": "{:+.6f}", "概率变化（百分点）": "{:+.4f}",
            }), hide_index=True, width="stretch")
        with notes_tab:
            st.write("预测概率表示模型估计的 LDAR ≥ 5.27 的可能性；5.27 是结局定义阈值。")
            st.write("模型类别取自 SVM 的判别结果。SVM 的类别判别与校准概率不一定以 50% 为共同边界。")
            st.write("E[f(X)] 为训练均值患者的参考概率。它加上当前患者的全部 SHAP 贡献，得到本次预测概率。")
            st.write("较小贡献在图中可能四舍五入为 0，详细数值可在指标明细或下载文件中查看。")
            st.caption("SHAP 描述模型中的贡献，不代表因果关系。")

        export_table = contribution_table.copy()
        export_table["预测结局"] = OUTCOME_TEXT
        export_table["预测概率"] = result["probability"]
        export_table["预测类别"] = result["predicted_class"]
        export_table["SHAP基准概率"] = result["baseline"]
        download_1, download_2 = st.columns(2)
        download_1.download_button("↓ 下载结果 CSV", export_table.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="LDAR_prediction_result.csv", mime="text/csv", width="stretch")
        download_2.download_button("↓ 下载 SHAP 图片", st.session_state["latest_plot"],
                                   file_name="LDAR_SHAP_waterfall.png", mime="image/png", width="stretch")

st.markdown('<div class="footer"><span>LDAR · 风险预测与个体解释</span>'
            '<span>研究展示工具 · 预测结果需结合临床信息解读</span></div>', unsafe_allow_html=True)
