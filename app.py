"""Streamlit GUI for SSR survey simulation: LLM personas + Semantic Similarity Rating.

Lets a PM configure a survey statement and a list of respondent personas,
run them through Gemini + ResponseRater, and see per-respondent and
aggregated Likert-scale results without touching code.

Setup:
    pip install -e ".[app]"
    Put GOOGLE_API_KEY="your-key" in a .env file (never hardcode keys)

Run:
    streamlit run app.py
"""

import hmac
import os
import time
import uuid

import numpy as np
import plotly.graph_objects as go
import polars as po
import streamlit as st
from dotenv import load_dotenv
from google import genai

from semantic_similarity_rating import ResponseRater

load_dotenv()

DEFAULT_MODEL_NAME = "gemini-flash-lite-latest"

# The models.list endpoint also returns TTS / image / music / robotics models
# that support generateContent but can't do plain text survey answers.
NON_TEXT_MODEL_KEYWORDS = (
    "tts",
    "image",
    "banana",
    "lyria",
    "veo",
    "robotics",
    "computer-use",
    "antigravity",
    "deep-research",
    "omni",
    "live",
    "audio",
    "embedding",
)

# Diverging blue<->red palette for the 5-pt Likert polarity axis (disagree <-> agree).
# Neutral midpoint uses the more visible baseline-gray role rather than the
# near-surface-white gray, since each step must still read as a bar fill.
LIKERT_COLORS = ["#e34948", "#ee8f8e", "#c3c2b7", "#86b6ef", "#2a78d6"]

# --- Reference Likert anchor sets (same as quick_start.py) ---
REFERENCE_SETS = {
    "plain": [
        "Strongly disagree",
        "Disagree",
        "Neutral / No opinion",
        "Agree",
        "Strongly agree",
    ],
    "conversational": [
        "No, I completely disagree with that.",
        "I mostly disagree with that.",
        "I'm not sure; I don't have a strong opinion either way.",
        "I mostly agree with that.",
        "Yes, I completely agree with that.",
    ],
    "first_person": [
        "That is definitely not true for me.",
        "That is mostly not true for me.",
        "That is only partly true for me.",
        "That is mostly true for me.",
        "That is definitely true for me.",
    ],
}

DEFAULT_QUESTION = (
    "I believe working remotely significantly increases my daily productivity."
)

DEFAULT_PERSONAS = [
    "You are a 25-year-old software engineer who loves cutting-edge tech and "
    "thoroughly enjoys the flexibility of working from home.",
    "You are a 50-year-old traditional banking executive who values face-to-face "
    "communication and worries about management difficulties caused by remote work.",
    "You are a 30-year-old marketing planner who feels that while remote work "
    "offers freedom, it makes team brainstorming sessions highly inefficient.",
]


# --- Cached resources ---
@st.cache_resource(show_spinner="Loading embedding model (first run can take ~30s)...")
def get_rater() -> ResponseRater:
    # No arguments -> one shared instance for the whole server process.
    # Fine for local/single-user; reconsider if this app is ever exposed to
    # multiple concurrent PMs at once.
    df_reference = po.DataFrame(
        {
            "id": [set_id for set_id, s in REFERENCE_SETS.items() for _ in s],
            "int_response": [1, 2, 3, 4, 5] * len(REFERENCE_SETS),
            "sentence": [sent for s in REFERENCE_SETS.values() for sent in s],
        }
    )
    return ResponseRater(df_reference)


@st.cache_resource(show_spinner=False)
def get_genai_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


@st.cache_data(show_spinner=False, ttl=3600)
def list_text_models(api_key: str) -> list[str]:
    """Text-capable Gemini/Gemma models available to this API key.

    Falls back to just the default model if the listing call fails
    (e.g. network issue or restricted key) so the app stays usable.
    """
    try:
        client = get_genai_client(api_key)
        names = []
        for m in client.models.list():
            if "generateContent" not in (m.supported_actions or []):
                continue
            name = (m.name or "").removeprefix("models/")
            if any(kw in name for kw in NON_TEXT_MODEL_KEYWORDS):
                continue
            names.append(name)
        return sorted(names) or [DEFAULT_MODEL_NAME]
    except Exception:
        return [DEFAULT_MODEL_NAME]


def get_llm_free_text_response(
    client: genai.Client,
    model_name: str,
    persona: str,
    question: str,
    max_retries: int = 3,
) -> str:
    prompt = f"""
Background Persona: {persona}

Survey Statement: "{question}"

Task: What is your realistic opinion on this statement based on your background?
Please role-play the persona and express your true feelings in exactly one
sentence (around 15-30 words).

Constraint: Output the direct response text only. Do not include any numerical
ratings, labels, or quotes. Must respond in English.
"""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text.strip()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)


# --- Persona list state ---
def _add_persona() -> None:
    pid = uuid.uuid4().hex
    st.session_state.persona_order.append(pid)
    st.session_state[f"persona_text::{pid}"] = ""


def _remove_persona(pid: str) -> None:
    st.session_state.persona_order.remove(pid)
    st.session_state.pop(f"persona_text::{pid}", None)


def _init_state() -> None:
    if "persona_order" not in st.session_state:
        st.session_state.persona_order = []
        for text in DEFAULT_PERSONAS:
            pid = uuid.uuid4().hex
            st.session_state.persona_order.append(pid)
            st.session_state[f"persona_text::{pid}"] = text
    if "survey_question" not in st.session_state:
        st.session_state.survey_question = DEFAULT_QUESTION
    if "results" not in st.session_state:
        st.session_state.results = None


def make_likert_bar_chart(labels: list[str], pmf: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[f"{i + 1}. {label}" for i, label in enumerate(labels)],
            y=pmf,
            marker_color=LIKERT_COLORS,
            marker_line_color="#fcfcfb",
            marker_line_width=2,
            text=[f"{p * 100:.1f}%" for p in pmf],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        yaxis_tickformat=".0%",
        yaxis_range=[0, max(pmf) * 1.25 if len(pmf) else 1],
        template="simple_white",
        margin=dict(t=40, b=10, l=10, r=10),
        height=320,
        showlegend=False,
    )
    return fig


def check_password() -> bool:
    """Gate the app behind APP_PASSWORD when it is set in the environment.

    No APP_PASSWORD configured -> app stays open (local/trusted-LAN use).
    Set it in .env before exposing the app through a public tunnel.
    """
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected:
        return True
    if st.session_state.get("password_ok"):
        return True

    def _verify() -> None:
        entered = st.session_state.pop("app_password_input", "")
        st.session_state.password_ok = hmac.compare_digest(entered, expected)

    st.title("AI 模擬問卷調查系統")
    st.text_input(
        "請輸入使用密碼",
        type="password",
        key="app_password_input",
        on_change=_verify,
    )
    if st.session_state.get("password_ok") is False:
        st.error("密碼錯誤，請再試一次。")
    return False


st.set_page_config(page_title="AI 模擬問卷調查系統", layout="wide")

if not check_password():
    st.stop()

_init_state()

st.title("AI 模擬問卷調查系統")
st.caption("讓 LLM 依據不同人格模擬填答問卷，並透過語意相似度換算成 Likert 量表機率分佈。")

# --- Sidebar: API key + advanced settings ---
env_api_key = os.environ.get("GOOGLE_API_KEY", "")
with st.sidebar:
    st.subheader("Gemini API Key")
    if env_api_key:
        st.success("已從 .env 載入 API Key")
        effective_api_key = env_api_key
    else:
        st.warning("未在 .env 找到 GOOGLE_API_KEY")
        effective_api_key = st.text_input(
            "貼上 Gemini API Key（僅此次 session 使用，不會被儲存）",
            type="password",
            key="api_key_override",
        )

    st.subheader("LLM 模型")
    if effective_api_key:
        model_options = list_text_models(effective_api_key)
    else:
        model_options = [DEFAULT_MODEL_NAME]
    model_name = st.selectbox(
        "選擇要模擬受訪者的模型",
        options=model_options,
        index=model_options.index(DEFAULT_MODEL_NAME)
        if DEFAULT_MODEL_NAME in model_options
        else 0,
        help="清單為你的 API Key 目前可用的文字生成模型。"
        "flash-lite 最快最便宜；pro 系列品質較高但較慢、較貴。",
        key="model_name",
    )

    rater = get_rater()
    with st.expander("進階設定"):
        reference_set_id = st.selectbox(
            "參考錨點組合 (reference set)",
            options=["mean"] + rater.available_reference_sets,
            index=0,
            help="mean = 平均多組不同措辭的錨點，較穩健，建議保留預設值。",
            key="reference_set_id",
        )
        temperature = st.slider(
            "PMF 溫度 (temperature)",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="小於 1 會讓結果分佈更集中（更武斷），大於 1 會更平均（更保留不確定性）。",
            key="temperature",
        )

# --- 1. Survey question ---
st.subheader("1. 問卷內容")
st.text_area(
    "問卷敘述句（受訪者將針對此敘述表達同意程度）",
    key="survey_question",
    height=80,
)

# --- 2. Personas ---
st.subheader("2. 受訪者設定")
st.caption(f"目前受訪者人數：{len(st.session_state.persona_order)}")

for pid in list(st.session_state.persona_order):
    col_text, col_remove = st.columns([10, 1])
    col_text.text_area(
        "受訪者人格描述",
        key=f"persona_text::{pid}",
        height=80,
        label_visibility="collapsed",
    )
    col_remove.button("移除", key=f"remove::{pid}", on_click=_remove_persona, args=(pid,))

st.button("+ 新增受訪者", on_click=_add_persona)

# --- 3. Run ---
st.subheader("3. 執行")
persona_texts = [
    st.session_state[f"persona_text::{pid}"] for pid in st.session_state.persona_order
]
has_blank = any(not t.strip() for t in persona_texts)
question_blank = not st.session_state.survey_question.strip()

if has_blank:
    st.caption(":warning: 請填寫或移除空白的受訪者欄位。")
if question_blank:
    st.caption(":warning: 請填寫問卷敘述句。")
if not effective_api_key:
    st.caption(":warning: 請先提供 Gemini API Key。")

run_clicked = st.button(
    "開始模擬問卷調查",
    type="primary",
    disabled=(not effective_api_key) or has_blank or question_blank or not persona_texts,
)

if run_clicked:
    client = get_genai_client(effective_api_key)
    responses: list[str | None] = []
    errors: dict[int, str] = {}

    with st.status("正在產生受訪者回答...", expanded=True) as status:
        for i, persona in enumerate(persona_texts):
            try:
                text = get_llm_free_text_response(
                    client, model_name, persona, st.session_state.survey_question
                )
                responses.append(text)
                st.write(f'受訪者 {i + 1}："{text}"')
            except Exception as e:
                responses.append(None)
                errors[i] = str(e)
                st.error(f"受訪者 {i + 1} 多次重試後仍失敗：{e}")
        status.update(label="完成", state="complete")

    valid_idx = [i for i, r in enumerate(responses) if r is not None]
    valid_texts = [responses[i] for i in valid_idx]

    pmfs = survey_pmf = None
    if valid_texts:
        pmfs = rater.get_response_pmfs(
            reference_set_id=reference_set_id,
            llm_responses=valid_texts,
            temperature=temperature,
            epsilon=1e-5,
        )
        survey_pmf = rater.get_survey_response_pmf(pmfs)

    st.session_state.results = {
        "personas": persona_texts,
        "responses": responses,
        "errors": errors,
        "valid_idx": valid_idx,
        "pmfs": pmfs,
        "survey_pmf": survey_pmf,
        "model_name": model_name,
        "reference_set_id": reference_set_id,
        "temperature": temperature,
        "scale_labels": rater.get_reference_sentences("plain"),
    }

# --- 4. Results ---
results = st.session_state.results
if results:
    st.subheader("4. 結果")
    if results.get("model_name"):
        st.caption(f"使用模型：{results['model_name']}")

    if results["survey_pmf"] is not None:
        st.markdown("### 整體彙總分佈 (Survey PMF)")
        st.plotly_chart(
            make_likert_bar_chart(
                results["scale_labels"], results["survey_pmf"], "整體受訪者彙總分佈"
            ),
            width='stretch',
        )
        st.dataframe(
            po.DataFrame(
                {
                    "Score": [1, 2, 3, 4, 5],
                    "Scale Anchor": results["scale_labels"],
                    "Aggregated Probability": [
                        f"{p * 100:.2f}%" for p in results["survey_pmf"]
                    ],
                }
            ),
            width='stretch',
            hide_index=True,
        )
    else:
        st.warning("所有受訪者皆執行失敗，無法計算彙總分佈。")

    st.markdown("### 個別受訪者結果")
    idx_to_row = {orig_i: row_i for row_i, orig_i in enumerate(results["valid_idx"])}
    for i, persona in enumerate(results["personas"]):
        with st.expander(f"受訪者 {i + 1}", expanded=False):
            st.caption(persona)
            if i in results["errors"]:
                st.error(f"執行失敗：{results['errors'][i]}")
                continue
            response_text = results["responses"][i]
            st.write(f'"{response_text}"')
            pmf = results["pmfs"][idx_to_row[i]]
            st.plotly_chart(
                make_likert_bar_chart(results["scale_labels"], pmf, f"受訪者 {i + 1} PMF"),
                width='stretch',
            )
            st.dataframe(
                po.DataFrame(
                    {
                        "Score": [1, 2, 3, 4, 5],
                        "Scale Anchor": results["scale_labels"],
                        "Probability": [f"{p * 100:.2f}%" for p in pmf],
                    }
                ),
                width='stretch',
                hide_index=True,
            )

    st.markdown("### 匯出")
    n = len(results["personas"])
    pmf_cols = {f"p_score_{s}": [float("nan")] * n for s in range(1, 6)}
    for row_i, orig_i in enumerate(results["valid_idx"]):
        for s in range(5):
            pmf_cols[f"p_score_{s + 1}"][orig_i] = float(results["pmfs"][row_i, s])

    df_export = po.DataFrame(
        {
            "persona": results["personas"],
            "response": results["responses"],
            "error": [results["errors"].get(i, "") for i in range(n)],
            **pmf_cols,
        }
    )
    st.download_button(
        "下載結果 CSV",
        data=df_export.write_csv(),
        file_name="ssr_results.csv",
        mime="text/csv",
    )
