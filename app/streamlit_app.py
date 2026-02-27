import os
import json
from datetime import datetime, timezone
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

from src.prompts import build_system_instruction
from src.schemas import make_response_schema, BatchResponse, LabeledRequest

st.set_page_config(page_title="AI Maintenance Ticket Assistant Demo", layout="centered")
st.title("AI Maintenance Ticket Assistant Demo")
st.info(
    "Residents pick urgency/category, then a constrained LLM second opinion to re-triage for routing accuracy and generates a safe, reassuring resident message."
)

# Config / env
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

MODEL = "gemini-2.5-flash"
DEMO_EXAMPLES_PATH = Path("data/outputs/demo_examples.csv")
PREDICTIONS_PATH = Path("data/outputs/predictions.csv")
MODE_TYPED = "Type my own request"
MODE_EXAMPLES = "Choose from 30 real examples"

client = genai.Client(api_key=api_key) if api_key else None

# Helpers
def now_iso_local() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def now_human_local() -> str:
    now = datetime.now(timezone.utc).astimezone()
    tz_name = now.tzname() or "Local Time"
    return now.strftime("%A, %B %d, %Y at %I:%M %p") + f" ({tz_name})"

def word_count(text: str) -> int:
    return len([w for w in (text or "").strip().split() if w])

@st.cache_data(show_spinner=False)
def load_demo_examples() -> pd.DataFrame | None:
    if not DEMO_EXAMPLES_PATH.exists():
        return None

    demo_df = pd.read_csv(DEMO_EXAMPLES_PATH)
    required_cols = [
        "id",
        "resident_selected_priority",
        "resident_selected_category",
        "comment",
        "ai_priority",
        "ai_service_category",
        "suggested_actions",
    ]
    if not set(required_cols).issubset(demo_df.columns):
        return None

    demo_df = demo_df[required_cols].copy()
    demo_df["id"] = pd.to_numeric(demo_df["id"], errors="coerce")
    demo_df = demo_df.dropna(subset=["id"]).copy()
    demo_df["id"] = demo_df["id"].astype(int)
    for col in required_cols[1:]:
        demo_df[col] = demo_df[col].fillna("").astype(str).str.strip()
    demo_df = demo_df.drop_duplicates(subset=["id"], keep="last")
    return demo_df

@st.cache_data(show_spinner=False)
def load_allowed_labels() -> tuple[list[str], list[str]] | None:
    if not PREDICTIONS_PATH.exists():
        return None

    pred_df = pd.read_csv(PREDICTIONS_PATH)
    required_cols = ["Priority", "Service_Category"]
    if not set(required_cols).issubset(pred_df.columns):
        return None

    priorities = sorted(
        pred_df["Priority"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
    )
    categories = sorted(
        pred_df["Service_Category"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
    )
    if not priorities or not categories:
        return None
    return priorities, categories

def make_example_label(row: pd.Series) -> str:
    snippet = row["comment"]
    snippet = (snippet[:90] + "…") if len(snippet) > 90 else snippet
    return (
        f'#{int(row["id"])} | {row["resident_selected_priority"]} | '
        f'{row["resident_selected_category"]} | {snippet}'
    )

# Load deploy-time artifacts for labels and examples
demo_examples_df = load_demo_examples()
demo_by_id = demo_examples_df.set_index("id") if demo_examples_df is not None else None

label_source = load_allowed_labels()
if label_source is not None:
    allowed_priorities, allowed_categories = label_source
elif demo_examples_df is not None and not demo_examples_df.empty:
    allowed_priorities = sorted(
        demo_examples_df["resident_selected_priority"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    allowed_categories = sorted(
        demo_examples_df["resident_selected_category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
else:
    st.error(
        "Missing label metadata. Provide `data/outputs/predictions.csv` "
        "or `data/outputs/demo_examples.csv`."
    )
    st.stop()


# UI: portal form
st.subheader("Submit a Service Request")
request_time_iso = now_iso_local()
request_time_display = now_human_local()
st.text_input("Request time", value=request_time_display, disabled=True)

mode = st.radio(
    "How would you like to submit the request?",
    [MODE_TYPED, MODE_EXAMPLES],
    index=1,
    horizontal=True,
)

selected_priority = None
selected_category = None
comment_text = ""
selected_example_id = None

is_example_mode = mode == MODE_EXAMPLES

if mode == MODE_TYPED:
    if client is None:
        st.info("Set GEMINI_API_KEY in `.env` to submit typed requests.")
    col1, col2 = st.columns(2)
    with col1:
        selected_priority = st.selectbox("Priority (resident-selected)", allowed_priorities, index=allowed_priorities.index("03-Routine") if "03-Routine" in allowed_priorities else 0)
    with col2:
        selected_category = st.selectbox("Service Category (resident-selected)", allowed_categories, index=0)

    comment_text = st.text_area(
        "Service comment (max 100 words)",
        height=140,
        placeholder="Describe the issue briefly...",
    )

    wc = word_count(comment_text)
    st.caption(f"Word count: {wc} / 100")
    too_long = wc > 100
    if too_long:
        st.error("Please keep the comment under 100 words.")

else:
    if demo_examples_df is None or demo_examples_df.empty:
        st.warning(
            "Missing `data/outputs/demo_examples.csv`. "
            "Run `python -m src.build_demo_examples` first."
        )
    else:
        labels = [make_example_label(r) for _, r in demo_examples_df.iterrows()]
        label_to_row = {labels[i]: demo_examples_df.iloc[i] for i in range(len(labels))}

        chosen = st.selectbox("Pick a real example", labels, index=0)
        row = label_to_row[chosen]

        selected_priority = str(row["resident_selected_priority"])
        selected_category = str(row["resident_selected_category"])
        comment_text = str(row["comment"])
        selected_example_id = int(row["id"])

        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            st.markdown("**Resident-selected Priority**")
            st.success(selected_priority)
        with ex_col2:
            st.markdown("**Resident-selected Service Category**")
            st.info(selected_category)

        st.markdown("**Resident comment:**")
        st.write(comment_text)

# Submit button
can_submit = bool(comment_text.strip()) and (word_count(comment_text) <= 100)
if mode == MODE_TYPED:
    can_submit = can_submit and (client is not None)
else:
    can_submit = can_submit and (demo_by_id is not None)

st.divider()
if st.button("Submit & Let AI Re-triage", disabled=not can_submit):
    out = None
    if is_example_mode:
        if demo_by_id is None:
            st.error(
                "Missing or invalid `data/outputs/demo_examples.csv` for example mode. "
                "Run `python -m src.build_demo_examples` first."
            )
            st.stop()
        if selected_example_id is None or selected_example_id not in demo_by_id.index:
            st.error(f"No cached prediction found for example id #{selected_example_id}.")
            st.stop()

        cached = demo_by_id.loc[selected_example_id]
        out = LabeledRequest(
            id=selected_example_id,
            Priority=str(cached["ai_priority"]),
            Service_Category=str(cached["ai_service_category"]),
            Suggested_Actions=str(cached["suggested_actions"]),
        )
    else:
        # Build input payload
        schema = make_response_schema(allowed_priorities, allowed_categories)
        system_instruction = build_system_instruction(allowed_priorities, allowed_categories)
        payload = [{
            "id": 1,  # demo id
            "ts": request_time_iso,
            "resident_selected_priority": selected_priority,
            "resident_selected_category": selected_category,
            "comment": comment_text.strip(),
        }]

        contents = system_instruction + "\n\n" + "INPUT_ITEMS_JSON:\n" + json.dumps(payload, ensure_ascii=False)

        with st.spinner("Calling Gemini..."):
            resp = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": schema,
                    "temperature": 0,
                },
            )
            out = BatchResponse.model_validate_json(resp.text).results[0]

    st.subheader("AI Re-triage Result")
    before_col, after_col = st.columns(2)
    with before_col:
        st.markdown("**Resident selection**")
        st.markdown(f"- Priority: `{selected_priority}`")
        st.markdown(f"- Category: `{selected_category}`")
    with after_col:
        st.markdown("**AI re-triage**")
        st.markdown(f"- Priority: `{out.Priority}`")
        st.markdown(f"- Category: `{out.Service_Category}`")

    st.subheader("Message to Resident")
    st.info(out.Suggested_Actions)
