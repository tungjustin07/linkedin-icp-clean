"""
Tier-2 AI scoring: deep ICP analysis with Claude Sonnet.

Batched + prompt-cached for cost efficiency:
  - SCORE_BATCH_SIZE profiles per API call (mirrors triage.py's batching pattern)
  - Static context (role preamble + ICP definition + customer examples + rubric)
    is sent as a cache_control=ephemeral `system` block so the Anthropic API
    caches it for 5 minutes, cutting repeated input cost ~90% on cached tokens.

Results are persisted to cache/scores.parquet after every batch so a crash
mid-run never loses completed work.
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
SCORE_BATCH_SIZE = 5  # Sonnet is heavier than Haiku — keep batches small

_SCORE_SCHEMA = {
    "profile_id": "str",
    "icp_score": "int64",
    "recommendation": "str",
    "reasoning": "str",
    "action_items": "object",
}

_TOOL = {
    "name": "score_profiles",
    "description": "Return ICP scoring analysis for a batch of LinkedIn profiles.",
    "input_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "0-based index matching the input batch order",
                        },
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
                    "required": ["index", "icp_score", "recommendation", "reasoning", "action_items"],
                },
            }
        },
        "required": ["results"],
    },
}

_ROLE_PREAMBLE = (
    "You are a senior B2B sales strategist scoring LinkedIn connections for ICP fit. "
    "Your job is to produce precise, evidence-based ICP scores for a batch of profiles. "
    "You will be given: (1) the ICP definition, (2) example ICP customers, (3) a batch of "
    "LinkedIn profiles. Score each profile on a 0–100 scale where: "
    "0–30=Poor fit, 31–60=Partial fit, 61–80=Good fit, 81–100=Excellent fit. "
    "Return your analysis using the score_profiles tool."
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
    dry_run_sample: int = 50,
) -> pd.DataFrame:
    """
    Run Sonnet deep scoring on all profiles where triage_results.triage_pass == True.
    Profiles are grouped into batches of SCORE_BATCH_SIZE and scored in a single
    Sonnet call per batch.

    Args:
        messages: Optional per-profile message engagement stats from ingest.load_messages().
                  When provided, message history is included in the scoring prompt and
                  significantly boosts scores for profiles with active DM history.
        dry_run: When True, cap input to dry_run_sample profiles and skip cache writes.
        dry_run_sample: Max profiles to score in dry_run mode (default 50).

    Returns a DataFrame with columns:
      profile_id, icp_score, recommendation, reasoning, action_items
    """
    cache = _load_score_cache(cache_path)
    cached_ids = set(cache["profile_id"].tolist()) if not cache.empty else set()

    passed = triage_results[triage_results["triage_pass"] == True]  # noqa: E712

    if dry_run:
        passed = passed.head(dry_run_sample)
        log.info("Dry-run: processing up to %d profiles (no cache write)", len(passed))

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

    # Build the cached system block ONCE per run — stable across all batches.
    # cache_control=ephemeral tells the API to cache this content for 5 minutes.
    system_blocks = _build_system_blocks(icp, customer_examples)

    # Index message stats by profile_id for O(1) lookup
    msg_index: dict[str, dict] = {}
    if messages is not None and not messages.empty:
        msg_index = messages.set_index("profile_id").to_dict("index")
        log.info("Message signals available for %d profiles", len(msg_index))

    # Build list of profile dicts to score
    profiles: list[dict] = []
    for _, row in to_score.iterrows():
        profiles.append({
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
        })

    batches = [
        profiles[i:i + SCORE_BATCH_SIZE]
        for i in range(0, len(profiles), SCORE_BATCH_SIZE)
    ]

    new_rows: list[dict] = []
    for batch_idx, batch in enumerate(
        tqdm(batches, desc="Scoring (Sonnet)", unit="batch")
    ):
        results = _score_batch(
            batch, system_blocks, client, log_usage=(batch_idx < 2)
        )
        new_rows.extend(results)

        if not dry_run:
            batch_df = pd.DataFrame(results)
            _save_score_cache(batch_df, cache_path)

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


# ── Sonnet batch scoring ──────────────────────────────────────────────────────


def _score_batch(
    batch: list[dict],
    system_blocks: list[dict],
    client,
    log_usage: bool = False,
) -> list[dict]:
    """Call Sonnet once for a batch of N profiles, return N scored result dicts."""
    prompt = _build_batch_prompt(batch)

    msg = call_with_backoff(
        client=client,
        model=SONNET_MODEL,
        system=system_blocks,
        messages=[{"role": "user", "content": prompt}],
        tools=[_TOOL],
        max_tokens=2500,
    )

    if log_usage:
        usage = getattr(msg, "usage", None)
        if usage is not None:
            log.info(
                "Sonnet batch usage — in:%s out:%s cache_read:%s cache_create:%s",
                getattr(usage, "input_tokens", "?"),
                getattr(usage, "output_tokens", "?"),
                getattr(usage, "cache_read_input_tokens", "?"),
                getattr(usage, "cache_creation_input_tokens", "?"),
            )

    tool_block = next((b for b in msg.content if b.type == "tool_use"), None)
    if tool_block is None:
        log.warning("Sonnet returned no tool_use for batch — defaulting all to score 0")
        return [_default_score(item["profile_id"]) for item in batch]

    try:
        raw_results: list[dict] = tool_block.input.get("results", [])
    except Exception as e:
        log.warning("Failed to parse tool_use input: %s — defaulting batch", e)
        return [_default_score(item["profile_id"]) for item in batch]

    # Align by index, fill missing slots with _default_score
    result_by_idx = {
        r["index"]: r
        for r in raw_results
        if isinstance(r, dict) and "index" in r
    }
    output: list[dict] = []
    for idx, item in enumerate(batch):
        r = result_by_idx.get(idx)
        if r is None:
            log.warning(
                "Missing score for batch index %d (%s) — defaulting", idx, item["profile_id"]
            )
            output.append(_default_score(item["profile_id"]))
            continue
        try:
            output.append({
                "profile_id": item["profile_id"],
                "icp_score": int(r.get("icp_score", 0)),
                "recommendation": r.get("recommendation", "REMOVE"),
                "reasoning": r.get("reasoning", ""),
                "action_items": list(r.get("action_items") or []),
            })
        except Exception as e:
            log.warning("Failed to parse result for %s: %s", item["profile_id"], e)
            output.append(_default_score(item["profile_id"]))
    return output


def _build_system_blocks(icp: dict, customer_examples: list[dict]) -> list[dict]:
    """
    Assemble the cached system content: role preamble + ICP definition +
    customer examples + scoring rubric. Marked with cache_control=ephemeral
    so the Anthropic API caches it for 5 minutes across batches.

    NOTE: tool-level cache_control is silently ignored by this API surface
    (verified empirically — in:~3000 cr_read:0 cr_create:0). System-level
    cache_control works. Keep the marker here, not on the tool definition.
    """
    weights = icp.get("scoring_weights", {})
    titles = ", ".join(icp.get("target_titles", []))
    industries = ", ".join(icp.get("target_industries", []))
    sizes = ", ".join(icp.get("company_sizes", []))

    if customer_examples:
        examples_block = "\n".join(
            f"[{i+1}] {ex['full_name']} — {ex['position']} at {ex['company']}"
            for i, ex in enumerate(customer_examples)
        )
    else:
        examples_block = "No example customers provided — rely on ICP definition alone."

    text = f"""{_ROLE_PREAMBLE}

## ICP Definition
{icp.get("description", "")}

Target titles: {titles}
Target industries: {industries}
Company sizes: {sizes}

Scoring weights:
- Title/role match: {weights.get("title_match", 0.4)}
- Industry/company type: {weights.get("industry_match", 0.25)}
- Company size: {weights.get("company_size_match", 0.2)}
- Recency of connection: {weights.get("recency", 0.1)}
- Engagement potential: {weights.get("engagement_potential", 0.05)}

## Example ICP Customers (few-shot reference)
{examples_block}

## Scoring Rubric
For each profile in the batch, return:
  icp_score       : 0–100 integer (0–30 Poor, 31–60 Partial, 61–80 Good, 81–100 Excellent)
  recommendation  : KEEP_ENGAGE (70+, active outreach) | KEEP_NURTURE (45–69, passive) |
                    DEPRIORITIZE (30–44, unlikely to convert) | REMOVE (<30, not ICP)
  reasoning       : 2–4 sentences grounded in the profile data and ICP definition
  action_items    : up to 3 specific next steps (empty list for REMOVE/DEPRIORITIZE)

IMPORTANT: Active DM history is a strong positive signal. A profile with real
back-and-forth message threads should almost never score below 45, regardless of
title/company fit.

## Worked Scoring Examples (calibration reference)

The following examples illustrate how the rubric above should be applied. Use them
to anchor your own scoring — if your score for a new profile feels far from
these anchors, reconsider.

[Example A] Sarah Chen — VP Revenue Operations at Stripe — connected 6 months ago,
no DM history yet. Stripe is a flagship modern B2B FinTech SaaS at target scale.
Title is an exact Tier-1 match (VP RevOps), and company archetype is textbook
sweet-spot (guaranteed SFDC + billing + enrichment complexity). No DM history
is a neutral, not negative — active outreach is warranted.
  → icp_score 90, recommendation KEEP_ENGAGE.
  Reasoning: Exact Tier-1 buyer title at an iconic modern-stack B2B SaaS
  company. Stripe's GTM sophistication and complexity-wall signals are
  unambiguous; outreach should be aggressive.

[Example B] John Smith — Staff GTM Engineer at Ramp — connected 3 years ago,
15 DMs over 6 months with Justin initiating first, last message within 30 days.
Staff-level technical IC in the GTM Systems / Growth Engineering track at a
prototypical modern-stack B2B SaaS (Ramp is the canonical exemplar from the
ICP definition). Warm DM thread elevates score above pure title-based value.
  → icp_score 82, recommendation KEEP_ENGAGE.
  Reasoning: Tier-2 technical buyer/referral source at a gold-standard modern
  B2B SaaS. Staff GTM Engineer is the exact Track 2 persona. Active DM history
  compounds the signal — this is a live relationship worth deepening.

[Example C] Jane Doe — Director of Product at HubSpot — connected 2 years ago,
no DM history. Director-level but wrong function (Product, not RevOps/GTM
Systems). HubSpot is a target company, but Product leadership does not own the
GTM system-of-record. Not an ICP buyer; possibly a referral source at best.
  → icp_score 32, recommendation DEPRIORITIZE.
  Reasoning: Right company, wrong function. Product leadership has no budget
  or decision authority for RevOps / GTM Systems engagements. Keep as a passive
  network contact but not a priority outreach target.

[Example D] Alex Brown — Founder & CEO at Acme Consulting — connected 1 year
ago, 3 DMs (they initiated). Founder status is an explicit anti-pattern — they
run a services/consulting firm, not a target B2B SaaS buyer. DM history came
from them selling or networking, not buying.
  → icp_score 15, recommendation REMOVE.
  Reasoning: Founder anti-pattern. Consulting-firm owner is a peer/competitor,
  not a buyer. Even with DM history, the sell-side dynamic disqualifies them
  from the ICP. This is a clear unfollow candidate.

[Example E] Priya Patel — Revenue Operations Manager at a pre-Series B startup
(15 employees) — connected 4 months ago, no DMs. Right title and function,
but the company is explicitly below the target size band (pre-RevOps-maturity
stage per the ICP). Her remit is likely to be firefighting rather than the
complexity-wall problems the ICP targets.
  → icp_score 40, recommendation DEPRIORITIZE.
  Reasoning: Right function, wrong company stage. Pre-Series-B companies don't
  have the systems complexity that justifies the engagement profile. Keep but
  do not prioritize — may become ICP as company scales.

[Example F] Marcus Williams — Salesforce Architect at Datadog — connected 2
years ago, no DMs. Exact Track 2 title match (Salesforce Architect) at a
flagship modern B2B SaaS company. Datadog's SFDC architecture is known to
be complex and actively evolving. Even without DM history, title + company
is enough for an 80+ score.
  → icp_score 83, recommendation KEEP_ENGAGE.
  Reasoning: Exact technical-architect match at a gold-standard B2B SaaS.
  Salesforce Architect is the single most literal match for the SFDC-rebuild
  sweet spot. Reach out with a peer-level technical framing.

Return all results in one call via the score_profiles tool, each tagged with its
0-based batch index."""

    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _build_batch_prompt(batch: list[dict]) -> str:
    """Build the per-batch user message — only the profile rows, nothing shared."""
    n = len(batch)
    lines = [f"## Profiles to Score (batch of {n})", ""]
    for i, item in enumerate(batch):
        lines.append(f"[{i}] {item['full_name']}")
        lines.append(f"    Position:        {item['position']}")
        lines.append(f"    Company:         {item['company']}")
        lines.append(f"    Connected:       {item['connected_on']}")
        lines.append(f"    LinkedIn:        {item.get('url', 'N/A')}")
        lines.append(f"    Triage Category: {item.get('category', 'UNKNOWN')}")
        lines.append(f"    Triage Reason:   {item.get('triage_reason', '')}")
        msg_block = _format_msg_block(item.get("msg_stats"))
        if msg_block:
            lines.append(f"    {msg_block}")
        lines.append("")
    lines.append(
        f"Score all {n} profiles above and return results via the score_profiles tool, "
        f"each keyed by its 0-based index."
    )
    return "\n".join(lines)


def _format_msg_block(stats: dict | None) -> str:
    """Format message engagement stats as a one-line prompt field (empty if no history)."""
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
    initiated = (
        "user initiated first contact"
        if stats.get("initiated_first")
        else "contact initiated first"
    )
    return (
        f"DM History:      {total} messages across {threads} thread(s) "
        f"({sent} sent / {received} received) — last contact {recency} — {initiated}"
    )


def _format_customer_examples(customers: pd.DataFrame) -> list[dict]:
    """Select up to MAX_CUSTOMER_EXAMPLES rows, preferring fully-populated rows."""
    if customers.empty:
        return []
    complete = customers[
        customers["full_name"].str.strip().astype(bool) &
        customers["company"].str.strip().astype(bool) &
        customers["position"].str.strip().astype(bool)
    ]
    sample = (
        complete.head(MAX_CUSTOMER_EXAMPLES)
        if len(complete) >= MAX_CUSTOMER_EXAMPLES
        else customers.head(MAX_CUSTOMER_EXAMPLES)
    )
    return sample[["full_name", "position", "company"]].to_dict("records")


def _default_score(profile_id: str) -> dict:
    return {
        "profile_id": profile_id,
        "icp_score": 0,
        "recommendation": "REMOVE",
        "reasoning": "Scoring failed — API error or malformed response.",
        "action_items": [],
    }
