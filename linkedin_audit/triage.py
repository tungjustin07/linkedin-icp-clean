"""
Tier-1 AI triage: classify LinkedIn connections with Claude Haiku.

Processes profiles in batches of 20 to minimize API calls. Results are cached
to cache/triage.parquet so partial runs are fully recoverable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from linkedin_audit._api import HAIKU_MODEL, call_with_backoff, get_client

log = logging.getLogger("linkedin_audit.triage")

BATCH_SIZE = 20
DEFAULT_CACHE_PATH = Path("cache/triage.parquet")

CATEGORIES = ["ICP_CORE", "ICP_ADJACENT", "PARTNER", "VENDOR", "PERSONAL", "SPAM", "UNKNOWN"]

_TRIAGE_SCHEMA = {
    "profile_id": "str",
    "category": "str",
    "triage_pass": "bool",
    "triage_confidence": "float64",
    "triage_reason": "str",
}

_TOOL = {
    "name": "classify_profiles",
    "description": "Return classification results for all profiles in the batch.",
    "input_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index":       {"type": "integer", "description": "0-based index from the batch"},
                        "category":    {"type": "string", "enum": CATEGORIES},
                        "triage_pass": {"type": "boolean"},
                        "confidence":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reason":      {"type": "string"},
                    },
                    "required": ["index", "category", "triage_pass", "confidence", "reason"],
                },
            }
        },
        "required": ["results"],
    },
}

_SYSTEM = (
    "You are a B2B sales analyst classifying LinkedIn connections for ICP relevance. "
    "You will receive a batch of LinkedIn profiles and an ICP definition. "
    "You must classify each profile and return results using the classify_profiles tool. "
    "Be decisive — use the profile data available, do not hedge."
)


# ── Public API ────────────────────────────────────────────────────────────────


def run_triage(
    connections: pd.DataFrame,
    icp_config: dict,
    messages: "pd.DataFrame | None" = None,
    cache_path: Path = DEFAULT_CACHE_PATH,
    dry_run: bool = False,
    dry_run_sample: int = 50,
) -> pd.DataFrame:
    """
    Run Haiku triage on all connections not already in the Parquet cache.

    Args:
        messages: Optional per-profile message engagement stats from ingest.load_messages().
                  When provided, each profile line in the prompt includes DM history.

    Returns a DataFrame with columns:
      profile_id, category, triage_pass, triage_confidence, triage_reason
    """
    cache = _load_triage_cache(cache_path)
    cached_ids = set(cache["profile_id"].tolist()) if not cache.empty else set()

    working = connections.copy()
    if dry_run:
        working = working.head(dry_run_sample)
        log.info("Dry-run: processing %d profiles (no cache write)", len(working))

    uncached = working[~working["profile_id"].isin(cached_ids)]
    log.info(
        "%d total | %d cached | %d to triage",
        len(working), len(working) - len(uncached), len(uncached),
    )

    if uncached.empty:
        log.info("All profiles already triaged — loading from cache.")
        return cache[cache["profile_id"].isin(working["profile_id"].tolist())].reset_index(drop=True)

    client = get_client()
    icp = icp_config["icp"]
    new_rows: list[dict] = []

    # Index message stats by profile_id for O(1) lookup
    msg_index: dict[str, dict] = {}
    if messages is not None and not messages.empty:
        msg_index = messages.set_index("profile_id").to_dict("index")
        log.info("Message signals available for %d profiles", len(msg_index))

    # Build list of profile dicts for batching
    profiles = [
        {
            "index": i,
            "profile_id": row["profile_id"],
            "full_name": row["full_name"] or "Unknown",
            "position": row["position"] or "Unknown",
            "company": row["company"] or "Unknown",
            "connected_on": (
                row["connected_on"].strftime("%b %Y")
                if pd.notna(row["connected_on"])
                else "Unknown"
            ),
            "msg_summary": _format_msg_summary(msg_index.get(row["profile_id"])),
        }
        for i, (_, row) in enumerate(uncached.iterrows())
    ]

    batches = [profiles[i:i + BATCH_SIZE] for i in range(0, len(profiles), BATCH_SIZE)]

    for batch in tqdm(batches, desc="Triage (Haiku)", unit="batch"):
        results = _classify_batch(batch, icp, client)
        for r in results:
            pid = batch[r["index"]]["profile_id"]
            new_rows.append({
                "profile_id": pid,
                "category": r.get("category", "UNKNOWN"),
                "triage_pass": bool(r.get("triage_pass", False)),
                "triage_confidence": float(r.get("confidence", 0.0)),
                "triage_reason": r.get("reason", ""),
            })

        if not dry_run:
            batch_df = pd.DataFrame(new_rows)
            _save_triage_cache(batch_df, cache_path)

    new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame(columns=list(_TRIAGE_SCHEMA))
    combined = pd.concat([cache, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="profile_id", keep="last")
    return combined[combined["profile_id"].isin(working["profile_id"].tolist())].reset_index(drop=True)


# ── Cache helpers ─────────────────────────────────────────────────────────────


def load_triage_cache(cache_path: Path = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    """Public wrapper — load triage results from Parquet cache."""
    return _load_triage_cache(cache_path)


def _load_triage_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(columns=list(_TRIAGE_SCHEMA))
    try:
        df = pd.read_parquet(cache_path)
        log.debug("Loaded %d triage records from cache", len(df))
        return df
    except Exception as e:
        log.warning("Triage cache corrupt (%s) — starting fresh", e)
        return pd.DataFrame(columns=list(_TRIAGE_SCHEMA))


def _save_triage_cache(new_rows: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_triage_cache(cache_path)
    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset="profile_id", keep="last")
    merged.to_parquet(cache_path, index=False)


# ── Haiku classification ──────────────────────────────────────────────────────


def _classify_batch(batch: list[dict], icp: dict, client) -> list[dict]:
    """Send one batch of profiles to Haiku via tool_use, return parsed results."""
    prompt = _build_triage_prompt(batch, icp)

    msg = call_with_backoff(
        client=client,
        model=HAIKU_MODEL,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        tools=[_TOOL],
        max_tokens=2000,
    )

    # Extract tool_use block
    tool_block = next((b for b in msg.content if b.type == "tool_use"), None)
    if tool_block is None:
        log.warning("Haiku returned no tool_use block for batch — defaulting all to UNKNOWN")
        return _default_results(batch)

    try:
        raw_results: list[dict] = tool_block.input.get("results", [])
    except Exception as e:
        log.warning("Failed to parse tool_use input: %s — defaulting batch", e)
        return _default_results(batch)

    # Align by index, fill missing batch slots
    result_by_idx = {r["index"]: r for r in raw_results if "index" in r}
    output = []
    for item in batch:
        idx = item["index"]
        if idx in result_by_idx:
            output.append(result_by_idx[idx])
        else:
            log.warning("Missing triage result for batch index %d (%s)", idx, item["profile_id"])
            output.append({
                "index": idx,
                "category": "UNKNOWN",
                "triage_pass": False,
                "confidence": 0.0,
                "reason": "parse_error",
            })
    return output


def _build_triage_prompt(batch: list[dict], icp: dict) -> str:
    titles = ", ".join(icp.get("target_titles", []))
    industries = ", ".join(icp.get("target_industries", []))
    anti = ", ".join(icp.get("anti_patterns", []))
    n = len(batch)

    lines = [
        "## ICP Definition",
        icp.get("description", ""),
        "",
        f"Target titles (partial match is fine): {titles}",
        f"Target industries: {industries}",
        f"Anti-patterns (auto-fail regardless of company): {anti}",
        "",
        f"## Profiles to Classify (batch of {n})",
        "",
    ]
    for item in batch:
        line = (
            f"[{item['index']}] {item['full_name']} | "
            f"{item['position']} at {item['company']} | "
            f"Connected: {item['connected_on']}"
        )
        if item.get("msg_summary"):
            line += f" | DMs: {item['msg_summary']}"
        lines.append(line)

    lines += [
        "",
        "## Task",
        f"For each of the {n} profiles above, determine:",
        "1. category: ICP_CORE (strong ICP signal), ICP_ADJACENT (partial fit), PARTNER (complementary business),",
        "   VENDOR (selling to you), PERSONAL (friend/family), SPAM (irrelevant/bot), UNKNOWN (no data)",
        "   NOTE: A profile with active DM history should score at least ICP_ADJACENT unless clearly irrelevant.",
        "2. triage_pass: true if worth deep scoring (ICP_CORE, ICP_ADJACENT, or PARTNER), false otherwise",
        "   NOTE: Any profile with DM history (DMs field present) should almost always pass triage.",
        "3. confidence: 0.0–1.0 float",
        "4. reason: one sentence explaining your classification",
        "",
        f"Use the classify_profiles tool to return all {n} results.",
    ]
    return "\n".join(lines)


def _format_msg_summary(stats: dict | None) -> str:
    """Format message engagement stats into a compact string for the triage prompt."""
    if not stats:
        return ""
    sent = stats.get("messages_sent", 0)
    received = stats.get("messages_received", 0)
    total = sent + received
    if total == 0:
        return ""
    threads = stats.get("thread_count", 1)
    days = stats.get("days_since_contact", 9999)
    if days < 30:
        recency = "active (<30d)"
    elif days < 180:
        recency = f"{days}d ago"
    elif days < 365:
        recency = f"{days // 30}mo ago"
    else:
        recency = f"{days // 365}y ago"
    initiated = "you initiated" if stats.get("initiated_first") else "they initiated"
    return f"{total} msgs ({sent} sent/{received} recv) • {threads} thread(s) • {recency} • {initiated}"


def _default_results(batch: list[dict]) -> list[dict]:
    return [
        {"index": item["index"], "category": "UNKNOWN",
         "triage_pass": False, "confidence": 0.0, "reason": "api_error"}
        for item in batch
    ]
