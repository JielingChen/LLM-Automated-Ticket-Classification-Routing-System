from __future__ import annotations
import os
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai

from .utils import load_csv, clean_text, ensure_dir, chunked, RateLimiter, read_done_ids, append_jsonl
from .schemas import BatchResponse, make_response_schema
from .prompts import build_system_instruction, build_user_contents

DATA = Path("data/processed/service_requests_sample_1000.csv")
OUT_DIR = Path("data/outputs")
PRED_CSV = OUT_DIR / "predictions.csv"
PRED_JSONL = OUT_DIR / "predictions.jsonl"

RPM_SLEEP_SECONDS = 13.0
DAILY_CALL_CAP = 20
BATCH_SIZE = 50
MODEL = "gemini-2.5-flash"

def gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)

def label_one_batch(client: genai.Client, batch_items: list[dict], schema: dict,
                    system_instruction: str) -> BatchResponse:
    user_contents = build_user_contents(json.dumps(batch_items, ensure_ascii=False))
    contents = system_instruction + "\n\n" + user_contents

    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": schema,
            "temperature": 0,
        },
    )
    return BatchResponse.model_validate_json(resp.text)

def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in the .env file.")

    ensure_dir(str(OUT_DIR))

    df = clean_text(load_csv(str(DATA)))
    allowed_priorities = sorted(df["Priority"].dropna().unique().tolist())
    allowed_categories = sorted(df["Service Category"].dropna().unique().tolist())

    schema = make_response_schema(allowed_priorities, allowed_categories)
    system_instruction = build_system_instruction(allowed_priorities, allowed_categories)

    client = gemini_client(api_key)
    limiter = RateLimiter(min_seconds_between_calls=RPM_SLEEP_SECONDS)

    done_ids = read_done_ids(str(PRED_CSV))
    todo = df[~df["id"].astype(int).isin(done_ids)].copy().reset_index(drop=True)

    if len(todo) == 0:
        print("Predictions.csv already contains all ids.")
        return

    payload_rows = [
        {"id": int(r["id"]), "ts": str(r["SR start date/time"]), "comment": str(r["Service Comments"])}
        for _, r in todo.iterrows()
    ]

    calls_made = 0
    all_records = []

    for batch in tqdm(list(chunked(payload_rows, BATCH_SIZE)), desc="Labeling batches"):
        if calls_made >= DAILY_CALL_CAP:
            print(f"Reached daily call cap ({DAILY_CALL_CAP}). Stop and resume tomorrow.")
            break

        out = label_one_batch(client, batch, schema, system_instruction)
        records = [x.model_dump() for x in out.results]

        append_jsonl(str(PRED_JSONL), records)
        all_records.extend(records)

        calls_made += 1
        limiter.wait()

    pred_df = pd.DataFrame(all_records)
    if pred_df.empty:
        print("No new predictions generated.")
        return

    if PRED_CSV.exists():
        existing = pd.read_csv(PRED_CSV)
        combined = pd.concat([existing, pred_df], ignore_index=True).drop_duplicates(subset=["id"])
    else:
        combined = pred_df

    combined.to_csv(PRED_CSV, index=False)
    print(f"Saved predictions to {PRED_CSV} (rows: {len(combined)})")

if __name__ == "__main__":
    main()
