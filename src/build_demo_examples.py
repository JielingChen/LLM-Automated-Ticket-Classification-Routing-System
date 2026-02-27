from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


SAMPLE_PATH = Path("data/processed/service_requests_sample_1000.csv")
PREDICTIONS_PATH = Path("data/outputs/predictions.csv")
OUT_PATH = Path("data/outputs/demo_examples.csv")

N_EXAMPLES = 30
SEED = 42


def _compute_priority_targets(
    df: pd.DataFrame, n: int, priority_col: str = "resident_selected_priority"
) -> dict[str, int]:
    """
    Compute per-priority quotas that are as balanced as possible first.
    If some priorities lack enough rows, redistribute the remainder.
    """
    counts = (
        df[priority_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .value_counts()
        .to_dict()
    )
    priorities = sorted(counts.keys())
    if not priorities:
        return {}

    n = min(n, int(sum(counts.values())))

    base = n // len(priorities)
    remainder = n % len(priorities)

    # Start from a near-equal split.
    targets = {
        p: base + (1 if i < remainder else 0)
        for i, p in enumerate(priorities)
    }

    # Cap by availability and keep track of what could not be assigned.
    deficit = 0
    for p in priorities:
        if targets[p] > counts[p]:
            deficit += targets[p] - counts[p]
            targets[p] = counts[p]

    # Reassign leftover slots while staying as balanced as possible.
    while deficit > 0:
        progressed = False
        for p in sorted(priorities, key=lambda x: (targets[x], x)):
            spare = counts[p] - targets[p]
            if spare <= 0:
                continue
            targets[p] += 1
            deficit -= 1
            progressed = True
            if deficit == 0:
                break
        if not progressed:
            break

    return targets


def build_example_set(df: pd.DataFrame, n: int = N_EXAMPLES, seed: int = SEED) -> pd.DataFrame:
    """
    Build n examples with balanced priority first, then maximize category diversity.
    """
    if df.empty:
        return df

    n = min(n, len(df))
    rng = df.sample(frac=1, random_state=seed).reset_index(drop=True).copy()
    rng["_order"] = rng.index

    targets = _compute_priority_targets(rng, n, "resident_selected_priority")
    priorities = sorted(targets.keys())

    selected_rows = []
    selected_ids: set[int] = set()
    global_cat_counts: Counter[str] = Counter()
    prio_cat_counts: dict[str, Counter[str]] = defaultdict(Counter)
    seen_categories: set[str] = set()
    remaining = targets.copy()

    # Round-robin by priority to enforce quota balance.
    while sum(remaining.values()) > 0:
        made_pick = False
        for priority in priorities:
            if remaining.get(priority, 0) <= 0:
                continue

            candidates = rng[
                (rng["resident_selected_priority"] == priority)
                & (~rng["id"].isin(selected_ids))
            ].copy()
            if candidates.empty:
                remaining[priority] = 0
                continue

            # Diversity objective inside fixed priority quota:
            # 1) prefer unseen categories
            # 2) then categories seen least globally
            # 3) then categories seen least within this priority
            candidates["_unseen_rank"] = candidates["resident_selected_category"].map(
                lambda c: 0 if c not in seen_categories else 1
            )
            candidates["_global_cat_rank"] = candidates["resident_selected_category"].map(
                lambda c: global_cat_counts.get(c, 0)
            )
            candidates["_prio_cat_rank"] = candidates["resident_selected_category"].map(
                lambda c: prio_cat_counts[priority].get(c, 0)
            )

            row = candidates.sort_values(
                ["_unseen_rank", "_global_cat_rank", "_prio_cat_rank", "_order"]
            ).iloc[0]

            selected_rows.append(row.drop(labels=["_unseen_rank", "_global_cat_rank", "_prio_cat_rank"]))
            selected_id = int(row["id"])
            selected_category = str(row["resident_selected_category"])

            selected_ids.add(selected_id)
            remaining[priority] -= 1
            global_cat_counts[selected_category] += 1
            prio_cat_counts[priority][selected_category] += 1
            seen_categories.add(selected_category)
            made_pick = True

        if not made_pick:
            break

    picked = pd.DataFrame(selected_rows).drop(columns=["_order"], errors="ignore")

    # Safety fallback: if quotas couldn't fill all rows, top up by category diversity.
    if len(picked) < n:
        need = n - len(picked)
        remaining_rows = rng[~rng["id"].isin(selected_ids)].copy()
        if not remaining_rows.empty:
            remaining_rows["_global_cat_rank"] = remaining_rows["resident_selected_category"].map(
                lambda c: global_cat_counts.get(c, 0)
            )
            top_up = remaining_rows.sort_values(
                ["_global_cat_rank", "_order"]
            ).head(need).drop(columns=["_global_cat_rank", "_order"], errors="ignore")
            picked = pd.concat([picked, top_up], ignore_index=True)

    picked = picked.sample(frac=1, random_state=seed).reset_index(drop=True)
    return picked.head(n)


def main() -> None:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Missing sample file: {SAMPLE_PATH}")
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions file: {PREDICTIONS_PATH}")

    sample_df = pd.read_csv(SAMPLE_PATH)
    predictions_df = pd.read_csv(PREDICTIONS_PATH)

    sample_required = {"id", "Priority", "Service Category", "Service Comments"}
    pred_required = {"id", "Priority", "Service_Category", "Suggested_Actions"}
    if not sample_required.issubset(sample_df.columns):
        raise ValueError(f"Sample CSV is missing columns: {sample_required - set(sample_df.columns)}")
    if not pred_required.issubset(predictions_df.columns):
        raise ValueError(
            f"Predictions CSV is missing columns: {pred_required - set(predictions_df.columns)}"
        )

    sample_df = sample_df.rename(
        columns={
            "Priority": "resident_selected_priority",
            "Service Category": "resident_selected_category",
            "Service Comments": "comment",
        }
    )[["id", "resident_selected_priority", "resident_selected_category", "comment"]].copy()

    sample_df["id"] = pd.to_numeric(sample_df["id"], errors="coerce")
    sample_df = sample_df.dropna(subset=["id"]).copy()
    sample_df["id"] = sample_df["id"].astype(int)
    sample_df["resident_selected_priority"] = (
        sample_df["resident_selected_priority"].fillna("").astype(str).str.strip()
    )
    sample_df["resident_selected_category"] = (
        sample_df["resident_selected_category"].fillna("").astype(str).str.strip()
    )
    sample_df["comment"] = sample_df["comment"].fillna("").astype(str).str.strip()

    predictions_df = predictions_df.rename(
        columns={
            "Priority": "ai_priority",
            "Service_Category": "ai_service_category",
            "Suggested_Actions": "suggested_actions",
        }
    )[["id", "ai_priority", "ai_service_category", "suggested_actions"]].copy()

    predictions_df["id"] = pd.to_numeric(predictions_df["id"], errors="coerce")
    predictions_df = predictions_df.dropna(subset=["id"]).copy()
    predictions_df["id"] = predictions_df["id"].astype(int)
    predictions_df["ai_priority"] = predictions_df["ai_priority"].fillna("").astype(str).str.strip()
    predictions_df["ai_service_category"] = (
        predictions_df["ai_service_category"].fillna("").astype(str).str.strip()
    )
    predictions_df["suggested_actions"] = (
        predictions_df["suggested_actions"].fillna("").astype(str).str.strip()
    )
    predictions_df = predictions_df.drop_duplicates(subset=["id"], keep="last")

    merged = sample_df.merge(predictions_df, on="id", how="inner")
    merged = merged.drop_duplicates(subset=["id"], keep="first")
    if len(merged) < N_EXAMPLES:
        raise ValueError(
            f"Need at least {N_EXAMPLES} joined rows, found {len(merged)}. "
            "Regenerate predictions.csv first."
        )

    demo_df = build_example_set(merged, n=N_EXAMPLES, seed=SEED)[
        [
            "id",
            "resident_selected_priority",
            "resident_selected_category",
            "comment",
            "ai_priority",
            "ai_service_category",
            "suggested_actions",
        ]
    ].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    demo_df.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(demo_df)} demo examples to {OUT_PATH}")
    print("Unique resident priorities:", demo_df["resident_selected_priority"].nunique())
    print("Unique resident categories:", demo_df["resident_selected_category"].nunique())


if __name__ == "__main__":
    main()
