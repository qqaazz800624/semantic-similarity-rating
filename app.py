"""Streamlit GUI for SSR survey simulation: LLM personas + Semantic Similarity Rating.

Lets a PM configure survey questions (single, or batch-uploaded via Excel) and
a list of respondent personas (manual or Excel), run them through Gemini +
ResponseRater, and see per-question aggregated Likert-scale results.

Setup:
    pip install -e ".[app]"

Run:
    streamlit run app.py
"""

import hmac
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as po
import streamlit as st
from dotenv import load_dotenv
from google import genai

from semantic_similarity_rating import ResponseRater

load_dotenv()


def get_config_value(name: str) -> str:
    """Read a config value from env vars (.env locally) or st.secrets (cloud).

    Streamlit Community Cloud stores secrets in st.secrets; accessing it
    locally without a secrets.toml raises, hence the broad except.
    """
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


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

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
QUESTION_XLSX_HEADER = "問卷題目"
PERSONA_XLSX_HEADER = "受訪者描述"

# Above this many personas the per-row editors become unwieldy; show a
# read-only table instead (batch edits happen by re-uploading the Excel).
PERSONA_EDIT_LIMIT = 20

MAX_PARALLEL_LLM_CALLS = 8

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

EXAMPLE_QUESTIONS = [
    DEFAULT_QUESTION,
    "I would recommend our new product to my friends and colleagues.",
    "I trust AI-generated content in my daily work.",
]

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


# --- Excel import/export helpers ---
@st.cache_data(show_spinner=False)
def make_template_xlsx(header: str, examples: tuple[str, ...]) -> bytes:
    buf = BytesIO()
    pd.DataFrame({header: list(examples)}).to_excel(buf, index=False)
    return buf.getvalue()


def parse_uploaded_xlsx(uploaded_file, expected_header: str) -> list[str]:
    """Read the first column of the first sheet; header must match the template."""
    df = pd.read_excel(uploaded_file, sheet_name=0)
    if len(df.columns) == 0 or str(df.columns[0]).strip() != expected_header:
        raise ValueError(
            f"Excel 第一欄的標題必須是「{expected_header}」。"
            "請下載範本檔，對照格式後再上傳。"
        )
    texts = [str(v).strip() for v in df.iloc[:, 0].dropna().tolist()]
    return [t for t in texts if t]


def get_llm_free_text_response(
    client: genai.Client,
    model_name: str,
    persona: str,
    question: str,
    max_retries: int = 4,
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
            time.sleep(2**attempt)  # 1s, 2s, 4s — survives brief rate-limit bursts


def run_survey(
    client: genai.Client,
    model_name: str,
    questions: list[str],
    personas: list[str],
    rater: ResponseRater,
    reference_set_id: str,
    temperature: float,
    progress_cb=None,
) -> dict:
    """Run every (question, persona) pair through the LLM + SSR pipeline.

    LLM calls run in a thread pool (each takes ~1-2s; a 50x100 batch would be
    hours if serial). Embedding/SSR runs once over all collected texts.
    Returns a results dict; failed calls are recorded per-cell in `errors`
    and excluded from the PMFs rather than aborting the batch.
    """
    n_q, n_p = len(questions), len(personas)
    responses: list[list[str | None]] = [[None] * n_p for _ in range(n_q)]
    errors: dict[tuple[int, int], str] = {}

    tasks = [(qi, pi) for qi in range(n_q) for pi in range(n_p)]
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_LLM_CALLS, len(tasks))) as ex:
        futures = {
            ex.submit(
                get_llm_free_text_response,
                client,
                model_name,
                personas[pi],
                questions[qi],
            ): (qi, pi)
            for qi, pi in tasks
        }
        for done, fut in enumerate(as_completed(futures), 1):
            qi, pi = futures[fut]
            try:
                responses[qi][pi] = fut.result()
            except Exception as e:
                errors[(qi, pi)] = str(e)
            if progress_cb:
                progress_cb(done, len(tasks))

    flat_texts: list[str] = []
    flat_keys: list[tuple[int, int]] = []
    for qi in range(n_q):
        for pi in range(n_p):
            if responses[qi][pi] is not None:
                flat_texts.append(responses[qi][pi])
                flat_keys.append((qi, pi))

    pmf_map: dict[tuple[int, int], np.ndarray] = {}
    per_question_pmf: list[np.ndarray | None] = [None] * n_q
    if flat_texts:
        pmfs = rater.get_response_pmfs(
            reference_set_id=reference_set_id,
            llm_responses=flat_texts,
            temperature=temperature,
            epsilon=1e-5,
        )
        pmf_map = {key: pmfs[i] for i, key in enumerate(flat_keys)}
        for qi in range(n_q):
            rows = [pmf_map[(qi, pi)] for pi in range(n_p) if (qi, pi) in pmf_map]
            if rows:
                per_question_pmf[qi] = np.array(rows).mean(axis=0)

    return {
        "questions": questions,
        "personas": personas,
        "responses": responses,
        "errors": errors,
        "pmf_map": pmf_map,
        "per_question_pmf": per_question_pmf,
        "reference_set_id": reference_set_id,
        "temperature": temperature,
    }


def expected_score(pmf: np.ndarray) -> float:
    return float((pmf * np.arange(1, 6)).sum())


# --- Persona / question state ---
# Texts live in plain session values (persona_store / question_store), NOT in
# the widget keys: Streamlit deletes widget-backed session values after any
# rerun where the widget isn't rendered (e.g. the login screen after logout,
# or the question text_area while in Excel mode), which wiped the examples.
# Plain session values survive; widgets re-seed from them via value= and
# write back via on_change.
def _sync_persona(pid: str) -> None:
    st.session_state.persona_store[pid] = st.session_state.get(
        f"persona_text::{pid}", ""
    )


def _sync_question() -> None:
    st.session_state.question_store = st.session_state.get("survey_question", "")


def _add_persona() -> None:
    pid = uuid.uuid4().hex
    st.session_state.persona_order.append(pid)
    st.session_state.persona_store[pid] = ""


def _remove_persona(pid: str) -> None:
    st.session_state.persona_order.remove(pid)
    st.session_state.persona_store.pop(pid, None)
    st.session_state.pop(f"persona_text::{pid}", None)


def _replace_personas(texts: list[str]) -> None:
    for pid in st.session_state.persona_order:
        st.session_state.pop(f"persona_text::{pid}", None)
    st.session_state.persona_order = []
    st.session_state.persona_store = {}
    for text in texts:
        pid = uuid.uuid4().hex
        st.session_state.persona_order.append(pid)
        st.session_state.persona_store[pid] = text


def _init_state() -> None:
    if "persona_order" not in st.session_state:
        st.session_state.persona_order = []
        st.session_state.persona_store = {}
        for text in DEFAULT_PERSONAS:
            pid = uuid.uuid4().hex
            st.session_state.persona_order.append(pid)
            st.session_state.persona_store[pid] = text
    if "question_store" not in st.session_state:
        st.session_state.question_store = DEFAULT_QUESTION
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
    expected = get_config_value("APP_PASSWORD")
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


def _validate_api_key(api_key: str) -> bool:
    try:
        client = genai.Client(api_key=api_key)
        next(iter(client.models.list()), None)
        return True
    except Exception:
        return False


def check_login() -> bool:
    """Require each user to provide their own Google account + Gemini API key.

    The key is kept in this browser session only — never written to disk —
    and all LLM calls run on the user's own quota.
    """
    if st.session_state.get("user_api_key"):
        return True

    st.title("AI 模擬問卷調查系統")
    st.markdown("請先填入你的 Google 帳號與 Gemini API Key，才能開始使用。")
    with st.form("login_form"):
        email = st.text_input("Google 帳號（email）")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="可到 https://aistudio.google.com/apikey 免費申請。"
            "Key 僅保存在你這次的瀏覽器工作階段，不會被儲存。",
        )
        submitted = st.form_submit_button("開始使用", type="primary")

    if submitted:
        email = (email or "").strip()
        api_key = (api_key or "").strip()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            st.error("請輸入有效的 Google 帳號（email 格式）。")
        elif not api_key:
            st.error("請輸入 Gemini API Key。")
        else:
            with st.spinner("驗證 API Key 中..."):
                if _validate_api_key(api_key):
                    st.session_state.user_email = email
                    st.session_state.user_api_key = api_key
                    st.rerun()
                else:
                    st.error(
                        "API Key 驗證失敗，請確認 Key 是否正確、"
                        "或到 https://aistudio.google.com/apikey 重新產生。"
                    )
    return False


def _logout() -> None:
    st.session_state.pop("user_email", None)
    st.session_state.pop("user_api_key", None)


st.set_page_config(page_title="AI 模擬問卷調查系統", layout="wide")

if not check_password():
    st.stop()

if not check_login():
    st.stop()

_init_state()

st.title("AI 模擬問卷調查系統")
st.caption("讓 LLM 依據不同人格模擬填答問卷，並透過語意相似度換算成 Likert 量表機率分佈。")

# --- Sidebar: user info + advanced settings ---
effective_api_key = st.session_state.user_api_key
with st.sidebar:
    st.subheader("使用者")
    st.write(st.session_state.user_email)
    st.caption("API Key 僅保存在此瀏覽器工作階段，使用你自己的額度。")
    st.button("登出", on_click=_logout)

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

# --- 1. Survey questions ---
st.subheader("1. 問卷內容")
question_mode = st.radio(
    "題目輸入方式",
    ["手動輸入單題", "上傳 Excel（多題）"],
    horizontal=True,
    key="question_mode",
)

questions: list[str] = []
if question_mode == "手動輸入單題":
    st.text_area(
        "問卷敘述句（受訪者將針對此敘述表達同意程度）",
        value=st.session_state.question_store,
        key="survey_question",
        on_change=_sync_question,
        height=80,
    )
    if st.session_state.question_store.strip():
        questions = [st.session_state.question_store.strip()]
else:
    st.download_button(
        "下載題目 Excel 範本",
        data=make_template_xlsx(QUESTION_XLSX_HEADER, tuple(EXAMPLE_QUESTIONS)),
        file_name="questions_template.xlsx",
        mime=XLSX_MIME,
    )
    question_file = st.file_uploader(
        f"上傳題目 Excel（第一欄標題須為「{QUESTION_XLSX_HEADER}」，每列一題，"
        "建議用英文撰寫題目）",
        type=["xlsx"],
        key="question_file",
    )
    if question_file is not None:
        try:
            questions = parse_uploaded_xlsx(question_file, QUESTION_XLSX_HEADER)
            st.success(f"已載入 {len(questions)} 題")
            with st.expander("預覽題目"):
                st.dataframe(
                    pd.DataFrame(
                        {"#": range(1, len(questions) + 1), "題目": questions}
                    ),
                    hide_index=True,
                    width="stretch",
                )
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"無法讀取 Excel 檔案：{e}")

# --- 2. Personas ---
st.subheader("2. 受訪者設定")

with st.expander("批次匯入受訪者（Excel 上傳）"):
    st.download_button(
        "下載受訪者 Excel 範本",
        data=make_template_xlsx(PERSONA_XLSX_HEADER, tuple(DEFAULT_PERSONAS)),
        file_name="personas_template.xlsx",
        mime=XLSX_MIME,
    )
    persona_file = st.file_uploader(
        f"上傳受訪者 Excel（第一欄標題須為「{PERSONA_XLSX_HEADER}」，每列一位受訪者，"
        "建議用英文描述其背景與個性）",
        type=["xlsx"],
        key="persona_file",
    )
    if persona_file is not None:
        try:
            personas_from_file = parse_uploaded_xlsx(persona_file, PERSONA_XLSX_HEADER)
            st.caption(f"檔案內含 {len(personas_from_file)} 位受訪者")
            if st.button(
                f"匯入並取代目前 {len(st.session_state.persona_order)} 位受訪者",
                key="import_personas",
            ):
                _replace_personas(personas_from_file)
                st.rerun()
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"無法讀取 Excel 檔案：{e}")

st.caption(f"目前受訪者人數：{len(st.session_state.persona_order)}")

if len(st.session_state.persona_order) <= PERSONA_EDIT_LIMIT:
    for pid in list(st.session_state.persona_order):
        col_text, col_remove = st.columns([10, 1])
        col_text.text_area(
            "受訪者人格描述",
            value=st.session_state.persona_store[pid],
            key=f"persona_text::{pid}",
            on_change=_sync_persona,
            args=(pid,),
            height=80,
            label_visibility="collapsed",
            placeholder=(
                "描述這位受訪者的背景與個性（建議用英文，回答品質較穩定），例如："
                "You are a 35-year-old nurse who works night shifts and is "
                "skeptical of new technology."
            ),
        )
        col_remove.button(
            "移除", key=f"remove::{pid}", on_click=_remove_persona, args=(pid,)
        )
    st.button("+ 新增受訪者", on_click=_add_persona)
else:
    st.dataframe(
        pd.DataFrame(
            {
                "#": range(1, len(st.session_state.persona_order) + 1),
                "受訪者描述": [
                    st.session_state.persona_store[pid]
                    for pid in st.session_state.persona_order
                ],
            }
        ),
        hide_index=True,
        width="stretch",
        height=350,
    )
    st.caption(
        f"受訪者超過 {PERSONA_EDIT_LIMIT} 位，改以唯讀清單顯示；"
        "如需調整內容請修改 Excel 後重新上傳。"
    )
    st.button("清空全部受訪者", on_click=_replace_personas, args=([],))

# --- 3. Run ---
st.subheader("3. 執行")
persona_texts = [
    st.session_state.persona_store[pid] for pid in st.session_state.persona_order
]
has_blank = any(not t.strip() for t in persona_texts)
total_calls = len(questions) * len(persona_texts)

if has_blank:
    st.caption(":warning: 請填寫或移除空白的受訪者欄位。")
if not questions:
    st.caption(":warning: 請填寫問卷敘述句或上傳題目 Excel。")

if total_calls:
    st.caption(
        f"共 {len(questions)} 題 × {len(persona_texts)} 位受訪者"
        f" = 預計 {total_calls} 次 LLM 呼叫"
    )
    if total_calls > 200:
        st.warning(
            "執行量較大，可能需要數分鐘以上，並消耗較多 API 額度。"
            "免費方案的 API Key 有每分鐘請求數限制，量大建議使用已啟用計費的 Key。"
        )

run_clicked = st.button(
    "開始模擬問卷調查",
    type="primary",
    disabled=(not effective_api_key)
    or has_blank
    or not questions
    or not persona_texts,
)

if run_clicked:
    client = get_genai_client(effective_api_key)
    progress = st.progress(0.0, text="正在產生受訪者回答...")

    def _update_progress(done: int, total: int) -> None:
        progress.progress(done / total, text=f"正在產生受訪者回答...（{done}/{total}）")

    survey_results = run_survey(
        client=client,
        model_name=model_name,
        questions=questions,
        personas=persona_texts,
        rater=rater,
        reference_set_id=reference_set_id,
        temperature=temperature,
        progress_cb=_update_progress,
    )
    progress.empty()

    survey_results["model_name"] = model_name
    survey_results["scale_labels"] = rater.get_reference_sentences("plain")
    st.session_state.results = survey_results

# --- 4. Results ---
results = st.session_state.results
if results:
    st.subheader("4. 結果")
    if results.get("model_name"):
        st.caption(f"使用模型：{results['model_name']}")

    r_questions = results["questions"]
    r_personas = results["personas"]
    labels = results["scale_labels"]
    n_p = len(r_personas)

    if not results["pmf_map"]:
        st.error("所有 LLM 呼叫皆失敗，無法計算結果。請檢查 API 額度或稍後再試。")
    else:
        # --- Per-question summary table ---
        st.markdown("### 各題彙總")
        summary_rows = []
        for qi, q in enumerate(r_questions):
            pmf = results["per_question_pmf"][qi]
            n_valid = sum(1 for pi in range(n_p) if (qi, pi) in results["pmf_map"])
            row = {"題目": q, "有效回覆": f"{n_valid}/{n_p}"}
            if pmf is not None:
                row["平均分數 (1-5)"] = round(expected_score(pmf), 2)
                for s in range(5):
                    row[f"{s + 1} 分機率"] = f"{pmf[s] * 100:.1f}%"
            summary_rows.append(row)
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

        # --- Per-question detail ---
        st.markdown("### 各題明細")
        for qi, q in enumerate(r_questions):
            with st.expander(
                f"題目 {qi + 1}：{q}", expanded=(len(r_questions) == 1)
            ):
                pmf = results["per_question_pmf"][qi]
                if pmf is None:
                    st.warning("此題所有受訪者皆執行失敗。")
                    continue
                st.plotly_chart(
                    make_likert_bar_chart(labels, pmf, "本題彙總分佈"),
                    width="stretch",
                    key=f"chart_q{qi}",
                )
                detail_rows = []
                for pi, persona in enumerate(r_personas):
                    p = results["pmf_map"].get((qi, pi))
                    detail_rows.append(
                        {
                            "受訪者": pi + 1,
                            "人格描述": persona,
                            "回答": results["responses"][qi][pi] or "",
                            "平均分數": round(expected_score(p), 2)
                            if p is not None
                            else None,
                            "錯誤": results["errors"].get((qi, pi), ""),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(detail_rows),
                    hide_index=True,
                    width="stretch",
                    height=min(38 * (n_p + 1) + 2, 400),
                )
                # Small single-question runs keep the per-respondent charts
                # PMs are used to from the original layout.
                if len(r_questions) == 1 and n_p <= 10:
                    for pi in range(n_p):
                        p = results["pmf_map"].get((qi, pi))
                        if p is not None:
                            st.plotly_chart(
                                make_likert_bar_chart(
                                    labels, p, f"受訪者 {pi + 1} PMF"
                                ),
                                width="stretch",
                                key=f"chart_q{qi}_p{pi}",
                            )

        # --- Export ---
        st.markdown("### 匯出")
        export_rows = []
        for qi, q in enumerate(r_questions):
            for pi, persona in enumerate(r_personas):
                p = results["pmf_map"].get((qi, pi))
                export_rows.append(
                    {
                        "question": q,
                        "persona": persona,
                        "response": results["responses"][qi][pi] or "",
                        "error": results["errors"].get((qi, pi), ""),
                        **{
                            f"p_score_{s + 1}": (
                                float(p[s]) if p is not None else None
                            )
                            for s in range(5)
                        },
                    }
                )
        df_export = po.DataFrame(export_rows)
        st.download_button(
            "下載結果 CSV",
            data=df_export.write_csv(),
            file_name="ssr_results.csv",
            mime="text/csv",
        )
