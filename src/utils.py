from __future__ import annotations
import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List
import pandas as pd

@dataclass
class RateLimiter:
    """Simple RPM limiter."""
    min_seconds_between_calls: float
    def wait(self):
        time.sleep(self.min_seconds_between_calls)

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Service Comments" in df.columns:
        df["Service Comments"] = df["Service Comments"].fillna("").astype(str).str.strip()
    if "Priority" in df.columns:
        df["Priority"] = df["Priority"].fillna("").astype(str).str.strip()
    if "Service Category" in df.columns:
        df["Service Category"] = df["Service Category"].fillna("").astype(str).str.strip()
    return df

def chunked(seq: List[Any], n: int) -> Iterator[List[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_done_ids(pred_csv_path: str) -> set[int]:
    if not os.path.exists(pred_csv_path):
        return set()
    try:
        df = pd.read_csv(pred_csv_path)
        return set(df["id"].astype(int).tolist()) if "id" in df.columns else set()
    except Exception:
        return set()

def append_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
