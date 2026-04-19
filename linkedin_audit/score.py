"""
Tier-2 AI scoring: deep ICP analysis with Claude Sonnet.

Runs one profile at a time (Sonnet is expensive — quality over throughput).
Results are cached to cache/scores.parquet after every single profile so
a crash mid-run never loses completed work.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from linkedin_audit._api import SONNET_MODEL, call_with_backoff, get_client

log = logging.getLogger("linkedin_audit.score")

DEFAULT_CACHE_PATH = Path("cache/scores.parquet")
MAX_CUSTOMER_EXAMPLES = 5

_SCORE_SCHEMA = {
    "profile_id": "str",
    "icp_score": "int64",
    "recommendation": "str",
    "reasoning": "str",
    "action_items": "object",
}

_TOOL = {
    "name": "score_profile",
    "description": "Return ICP scoring analysis for a single LinkedIn profile.",
    "input_schema": {
        "type": "object",
        "properties": {
            "icp_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "0–100 ICP fit score",
            },
            "recommendation": {
                "type": "string",
                "enum": ["KEEP_ENGAGE", "KEEP_NURTURE", "DEPRIORITIZE", "REMOVE"],
                "description": (
                    "KEEP_ENGAGE=70+, active outreach warranted. "
                    "KEEP_NURTURE=45-69, keep but passive. "
                    "DEPRIORITIZE=30-44, unlikely to convert. "
                    "REMOVE=below 30, not ICP."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "2–4 sentences grounded in the profile data and ICP definition.",
            },
            "action_items": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
                "description": "Specific next steps (empty list for REMOVE/DEPRIORITIZE).",
            },
        },
        "required": ["icp_score", "recommendation", "reasoning", "action_items"],
    },
}

_SYSTEM = (
    "You are a senior B2B sales strategist scoring LinkedIn connections for ICP fit. "
    "Your job is to produce a precise, evidence-based ICP score for a single profile. "
    "You will be given: (1) the profile, (2) the ICP definition, (3) example ICP customers. "
    "Score the profile on a 0–100 scale where: "
    "0–30=Poor fit, 31–60=Partial fit, 61–80=Good fit, 81–100=Excellent fit. "
    "Return your analysis using the score_profile tool."
)


# ── Public API ────────────────────────────────────────────────────────────────


def run_scoring(
    connections: pd.DataFrame,
    triage_results: pd.DataFrame,
    icp_config: dict,
    customers: pd.DataFrame,
    messages: "pd.DataFrame | None" = None,
    cache_path: Path = DEFAULT_CACHE_PATH,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Run Sonnet deep scoring on all profiles where triage_results.triage_pass == True.

    Args:
        messages: Optional per-profile message engagement stats from ingest.load_messages().
                  When provided, message history is included in the scoring prompt and
                  significantly boosts scores for profiles with active DM history.

    Returns a DataFrame with columns:
      profile_id, icp_score, recommendation, reasoning, action_items
    """
    cache = _load_score_cache(cache_path)
    cached_ids = set(cache["profile_id"].tolist()) if not cache.empty else set()

    passed = triage_results[triage_results["triage_pass"] == True]  # noqa: E712
    passed_ids = set(passed["profile_id"].tolist())
    to_score_ids = passed_ids - cached_ids

    log.info(
        "%d passed triage | %d cached | %d to score with Sonnet",
        len(passed_ids), len(passed_ids) - len(to_score_ids), len(to_score_ids),
    )

    if not to_score_ids:
        log.info("All passing profiles already scored — loading from cache.")
        return cache[cache["profile_id"].isin(passed_ids)].reset_index(drop=True)

    # Build merged DataFrame: connections + triage results for context
    merged = connections.merge(
        triage_results[["profile_id", "category", "triage_reason"]],
        on="profile_id",
        how="inner",
    )
    to_score = merged[merged["profile_id"].isin(to_score_ids)]

    client = get_client()
    icp = icp_config["icp"]
    customer_examples = _format_customer_examples(customers)

    # Index message stats by profile_id for O(1) lookup
    msg_index: dict[str, dict] = {}
    if messages is not None and not messages.empty:
        msg_index = messages.set_index("profile_id").to_dict("index")
        log.info("Message signals available for %d profiles", len(msg_index))

    new_rows: list[dict] = []

    for _, row in tqdm(to_score.iterrows(), total=len(to_score), desc="Scoring (Sonnet)", unit="profile"):
        profile = {
            "profile_id": row["profile_id"],
            "full_name": row["full_name"] or "Unknown",
            "position": row["position"] or "Unknown",
            "company": row["company"] or "Unknown",
            "connected_on": (
                row["connected_on"].strftime("%b %Y")
                if pd.notna(row.get("connected_on"))
                else "Unknown"
            ),
            "url": row.get("url", ""),
            "category": row.get("category", "UNKNOWN"),
            "triage_reason": row.get("triage_reason", ""),
            "msg_stats": msg_index.get(row["profile_id"]),
        }

        result = _score_profile(profile, icp, customer_examples, client)
        new_rows.append(result)

        if not dry_run:
            row_df = pd.DataFrame([result])
            _save_score_cache(row_df, cache_path)

    new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame(columns=list(_SCORE_SCHEMA))
    combined = pd.concat([cache, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="profile_id", keep="last")
    return combined[combined["profile_id"].isin(passed_ids)].reset_index(drop=True)


# ── Cache helpers ─────────────────────────────────────────────────────────────


def load_score_cache(cache_path: Path = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    """Public wrapper — load score results from Parquet cache."""
    return _load_score_cache(cache_path)


def _load_score_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(columns=list(_SCORE_SCHEMA))
    try:
        df = pd.read_parquet(cache_path)
        # action_items may be serialized as strings in some Parquet readers
        if "action_items" in df.columns:
            df["action_items"] = df["action_items"].apply(_parse_list)
        log.debug("Loaded %d score records from cache", len(df))
        return df
    except Exception as e:
        log.warning("Score cache corrupt (%s) — starting fresh", e)
        return pd.DataFrame(columns=list(_SCORE_SCHEMA))


def _save_score_cache(new_rows: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_score_cache(cache_path)
    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset="profile_id", keep="last")
    merged.to_parquet(cache_path, index=False)


def _parse_list(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return []


# ── Sonnet scoring ────────────────────────────────────────────────────────────


def _score_profile(
    profile: dict,
    icp: dict,
    customer_examples: list[dict],
    client,
) -> dict:
    """Call Sonnet for a single profile. Returns a scored result dict."""
    prompt = _build_score_prompt(profile, icp, customer_examples)

    msg = call_with_backoff(
        client=client,
        model=SONNET_MODEL,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        tools=[_TOOL],
        max_tokens=1000,
    )

    tool_block = next((b for b in msg.content if b.type == "tool_use"), None)
    if tool_block is None:
        log.warning("Sonnet returned no tool_use for %s — defaulting to score 0", profile["profile_id"])
        return _default_score(profile["profile_id"])

    try:
        inp = tool_block.input
        return {
            "profile_id": profile["profile_id"],
            "icp_score": int(inp.get("icp_score", 0)),
            "recommendation": inp.get("recommendation", "REMOVE"),
            "reasoning": inp.get("reasoning", ""),
            "action_items": inp.get("action_items", []),
        }
    except Exception as e:
        log.warning("Failed to parse Sonnet output for %s: %s", profile["profile_id"], e)
        return _default_score(profile["profile_id"])


def _build_score_prompt(profile: dict, icp: dict, customer_examples: list[dict]) -> str:
    weights = icp.get("scoring_weights", {})
    titles = ", ".join(icp.get("target_titles", []))
    industries = ", ".join(icp.get("target_industries", []))

    if customer_examples:
        examples_block = "\n".join(
            f"[{i+1}] {ex['full_name']} — {ex['position']} at {ex['company']}"
            for i, ex in enumerate(customer_examples)
        )
    else:
        examples_block = "No example customers provided — rely on ICP definition alone."

    return f"""## ICP Definition
{icp.get("description", "")}

Target titles: {titles}
Target industries: {industries}
Company sizes: {", ".join(icp.get("company_sizes", []))}

Scoring weights:
- Title/role match: {weights.get("title_match", 0.4)}
- Industry/company type: {weights.get("industry_match", 0.25)}
- Company size: {weights.get("company_size_match", 0.2)}
- Recency of connection: {weights.get("recency", 0.1)}
- Engagement potential: {weights.get("engagement_potential", 0.05)}

## Example ICP Customers (few-shot reference)
{examples_block}

## Profile to Score
Name:             {profile["full_name"]}
Position:         {profile["position"]}
Company:          {profile["company"]}
Connected:        {profile["connected_on"]}
LinkedIn:         {profile.get("url", "N/A")}
Triage Category:  {profile.get("category", "UNKNOWN")}
Triage Reason:    {profile.get("triage_reason", "")}
{_format_msg_block(profile.get("msg_stats"))}
## Task
Score this profile 0–100 for ICP fit and return your analysis using the score_profile tool.
Recommendations:
  KEEP_ENGAGE    = 70+ score, active outreach warranted
  KEEP_NURTURE   = 45–69 score, worth keeping but passive
  DEPRIORITIZE   = 30–44 score, keep but unlikely to convert
  REMOVE         = below 30, not ICP, no value maintaining connection

IMPORTANT: Active DM history is a strong positive signal. A profile with real back-and-forth
message threads should almost never score below 45, regardless of title/company fit.
"""


def _format_msg_block(stats: dict | None) -> str:
    """Format message engagement stats as a prompt section (empty string if no history)."""
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
        recency = "within the last month"
    elif days < 180:
        recency = f"{days} days ago"
    elif days < 365:
        recency = f"{days // 30} months ago"
    else:
        recency = f"{days // 365} year(s) ago"
    initiated = "user initiated first contact" if stats.get("initiated_first") else "contact initiated first"
    return (
        f"DM History:       {total} messages across {threads} thread(s) "
        f"({sent} sent / {received} received) — last contact {recency} — {initiated}\n"
    )


def _format_customer_examples(customers: pd.DataFrame) -> list[dict]:
    """Select up to MAX_CUSTOMER_EXAMPLES rows from customers, prefer fully-populated rows."""
    if customers.empty:
        return []
    complete = customers[
        customers["full_name"].str.strip().astype(bool) &
        customers["company"].str.strip().astype(bool) &
        customers["position"].str.strip().astype(bool)
    ]
    sample = complete.head(MAX_CUSTOMER_EXAMPLES) if len(complete) >= MAX_CUSTOMER_EXAMPLES else customers.head(MAX_CUSTOMER_EXAMPLES)
    return sample[["full_name", "position", "company"]].to_dict("records")


def _default_score(profile_id: str) -> dict:
    return {
        "profile_id": profile_id,
        "icp_score": 0,
        "recommendation": "REMOVE",
        "reasoning": "Scoring failed — API error or malformed response.",
        "action_items": [],
    }
