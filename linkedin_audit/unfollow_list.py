"""
Post-processing: produce a worst-first unfollow/disconnect review list.

The built-in output/remove.csv already includes:
  - Prefilter drops (empty profiles, 2yr+ cold, spam patterns)
  - Triage failures (anti-patterns: founders, SDRs, recruiters, etc.)
  - Sonnet REMOVE verdicts (scored <30)

This script adds Sonnet DEPRIORITIZE profiles (scored 30–44) — which
currently fall through between keep.csv and remove.csv — and produces a
single sorted-worst-first CSV that is optimized for manual review.

Run after the main pipeline finishes:
    python -m linkedin_audit.unfollow_list
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger("linkedin_audit.unfollow_list")

# Tier labels — sorted ascending so "1_" comes first in any natural ordering.
TIER_PREFILTER = "1_PREFILTER"        # empty / cold / spam — bulk action, no thought
TIER_ANTI_PATTERN = "2_ANTI_PATTERN"  # founders, SDRs, recruiters — bulk action
TIER_REMOVE = "3_REMOVE"              # Sonnet REMOVE (score <30) — skim reasoning
TIER_DEPRIORITIZE = "4_DEPRIORITIZE"  # Sonnet DEPRIORITIZE (30–44) — judgment zone


# ── Rescue-hint patterns ──────────────────────────────────────────────────────
# Post-hoc annotations surfaced in the `rescue_hint` column. The pipeline's
# core scoring is ICP-centric, but some non-ICP patterns are still connections
# worth keeping for relationship reasons (VCs as introducers, IT leaders as
# potential hirers/peers, etc.). These hints let the user filter the unfollow
# list for "review don't auto-remove" without re-scoring.

# VC / investor — good introducers even though they don't buy directly.
#
# Three-tier matching:
#   1. Title-level VC signal (strong, unambiguous): General Partner, Managing
#      Partner, Venture Partner, Angel Investor, etc. → flag regardless of company.
#   2. Company-level strong VC signal: "Ventures" / "Capital" in the company
#      name → almost always an investment firm (VC, PE, hedge fund) → flag
#      regardless of title, because even non-partner investor titles
#      (Principal, Managing Director, Investment Director) belong here.
#   3. Company-level WEAK VC signal: "Partners" in the company name → ambiguous
#      (law firms, consulting, recruitment, VC) — require an investor-ish
#      title ("Partner", "Investment", "Portfolio", "Fund") to flag, else
#      skip to avoid false positives on Recruiters / HR Business Partners.
VC_PATTERNS = [
    r"\bventure\b", r"\bvc\b", r"\binvestor\b", r"general partner",
    r"managing partner", r"founding partner", r"venture partner",
    r"\bangel\b.*\binvestor\b", r"limited partner", r"growth equity",
]
VC_STRONG_COMPANY_PATTERNS = [r"\bventures?\b", r"\bcapital\b"]
VC_WEAK_COMPANY_PATTERNS = [r"\bpartners\b"]
VC_ADJACENT_TITLE_PATTERNS = [
    r"\bpartner\b", r"\binvestment\b", r"\bportfolio\b", r"\bfund\b",
]

# IT leadership — secondary buyer persona (can hire or get hired, peer to RevOps).
IT_LEADER_PATTERNS = [
    r"vp.*information technology", r"vp.*\bit\b", r"head of (information|it)\b",
    r"director of it\b", r"director.*information technology",
    r"chief information officer", r"\bcio\b", r"chief technology officer", r"\bcto\b",
]

# Senior seniority markers — for CS/PMM/generalist ops roles that are only
# interesting at seniority level or with engagement.
SENIOR_PATTERNS = [
    r"\bvp\b", r"vice president", r"head of\b", r"\bdirector\b", r"\bprincipal\b",
    r"\bstaff\b", r"\blead\b", r"\bchief\b",
]


def _compute_rescue_hint(row: pd.Series, msg_index: dict[str, int]) -> str:
    """
    Return a rescue_hint string if this profile matches a "don't auto-remove"
    pattern, else "". Multiple matches are concatenated with "+" for a single
    readable string, e.g. "VC+HAS_ENGAGEMENT".

    Rules (additive):
      - VC_INTRODUCER: VC / investor role or company — good intro source.
      - IT_LEADER: VP/Head/Director of IT, CIO, CTO — secondary buyer / hirer.
      - HAS_ENGAGEMENT: any DM history on this profile — a real relationship.
      - SENIOR_TITLE: VP/Head/Director/Principal/Staff/Lead/Chief — worth a
        second look even if function is off-ICP.
    """
    pos = str(row.get("position", "") or "").lower()
    comp = str(row.get("company", "") or "").lower()
    pid = str(row.get("profile_id", "") or "")

    hints: list[str] = []

    # VC signal — three tiers (see pattern comments above)
    if any(re.search(p, pos) for p in VC_PATTERNS):
        hints.append("VC_INTRODUCER")
    elif any(re.search(p, comp) for p in VC_STRONG_COMPANY_PATTERNS):
        # "Ventures" / "Capital" in company = investment firm, any title is fine
        hints.append("VC_INTRODUCER")
    elif any(re.search(p, comp) for p in VC_WEAK_COMPANY_PATTERNS) and any(
        re.search(p, pos) for p in VC_ADJACENT_TITLE_PATTERNS
    ):
        # "Partners" in company is ambiguous — require investor-ish title
        hints.append("VC_INTRODUCER")

    # IT leadership
    if any(re.search(p, pos) for p in IT_LEADER_PATTERNS):
        hints.append("IT_LEADER")

    # Engagement signal from DM history
    if pid and msg_index.get(pid, 0) > 0:
        hints.append("HAS_ENGAGEMENT")

    # Senior title (only report if NOT already flagged — redundant otherwise)
    if not hints and any(re.search(p, pos) for p in SENIOR_PATTERNS):
        hints.append("SENIOR_TITLE")

    return "+".join(hints)


def build_unfollow_list(
    connections_path: Path = Path("data/Connections.csv"),
    messages_path: Path = Path("data/messages.csv"),
    remove_csv_path: Path = Path("output/remove.csv"),
    scores_parquet_path: Path = Path("cache/scores.parquet"),
    output_path: Path = Path("output/unfollow_ranked.csv"),
) -> pd.DataFrame:
    """
    Produce a worst-first ranked unfollow review list.

    Combines the base remove.csv with DEPRIORITIZE profiles from the Sonnet
    score cache (which report.py currently leaves orphaned between keep/remove).

    Adds a `tier` column so the user can stop reviewing at whatever confidence
    threshold they want — tiers 1–2 are bulk-remove candidates, tier 3 is
    quick-skim, tier 4 is "read the reasoning before deciding."
    """
    from linkedin_audit import ingest  # local import — avoids import cycles

    if not remove_csv_path.exists():
        raise FileNotFoundError(
            f"{remove_csv_path} not found — run the main pipeline first "
            "(python -m linkedin_audit.run_audit ...)"
        )

    log.info("Loading %s", remove_csv_path)
    remove = pd.read_csv(remove_csv_path)

    # ── Assign tier to each row based on category + icp_score ─────────────
    def _tier(row: pd.Series) -> str:
        cat = str(row.get("category", "")).upper()
        score = row.get("icp_score", -1)
        if cat == "PREFILTERED":
            return TIER_PREFILTER
        # score == -1 means triage failed (no Sonnet score was assigned)
        if score == -1 or cat == "SPAM":
            return TIER_ANTI_PATTERN
        # Sonnet scored this profile as REMOVE
        if score < 30:
            return TIER_REMOVE
        return TIER_DEPRIORITIZE

    remove["tier"] = remove.apply(_tier, axis=1)

    # ── Augment with DEPRIORITIZE profiles from the scores cache ─────────
    if scores_parquet_path.exists():
        log.info("Loading %s for DEPRIORITIZE profiles", scores_parquet_path)
        scores = pd.read_parquet(scores_parquet_path)
        deprioritized = scores[scores["recommendation"] == "DEPRIORITIZE"]
        # Exclude any profile_ids already captured in remove.csv (shouldn't be any,
        # but a belt-and-suspenders defense against schema drift).
        already_in_remove = set(remove["profile_id"].astype(str).tolist())
        new_dp = deprioritized[
            ~deprioritized["profile_id"].astype(str).isin(already_in_remove)
        ].copy()

        if not new_dp.empty:
            log.info("Loading connections for DEPRIORITIZE profile details")
            connections = ingest.load_connections(connections_path)
            enriched = new_dp.merge(
                connections[["profile_id", "full_name", "company", "position", "url", "connected_on"]],
                on="profile_id",
                how="left",
            )
            enriched["category"] = "DEPRIORITIZE"
            enriched["triage_reason"] = ""
            enriched["tier"] = TIER_DEPRIORITIZE
            # Match the remove.csv column order + our added 'tier' column
            cols = [
                "profile_id", "full_name", "company", "position", "url",
                "category", "triage_reason", "icp_score", "reasoning",
                "connected_on", "tier",
            ]
            enriched = enriched[[c for c in cols if c in enriched.columns]]
            remove = pd.concat([remove, enriched], ignore_index=True, sort=False)
            log.info("Added %d DEPRIORITIZE profiles", len(new_dp))
    else:
        log.warning("Score cache %s not found — DEPRIORITIZE tier will be empty", scores_parquet_path)

    # ── Compute rescue_hint for every row ────────────────────────────────
    # Load DM history so "HAS_ENGAGEMENT" hint is accurate.
    msg_index: dict[str, int] = {}
    if messages_path.exists():
        log.info("Loading %s for engagement hints", messages_path)
        messages = ingest.load_messages(messages_path)
        if not messages.empty:
            msg_index = dict(zip(
                messages["profile_id"].astype(str),
                (messages["messages_sent"] + messages["messages_received"]).astype(int),
            ))
    else:
        log.warning("%s not found — HAS_ENGAGEMENT hint will be blank", messages_path)

    log.info("Computing rescue hints (VC / IT leader / engagement / seniority)...")
    remove["rescue_hint"] = remove.apply(
        lambda r: _compute_rescue_hint(r, msg_index),
        axis=1,
    )

    # ── Sort: tier ascending, rescue_hint presence DESC (flagged to top of
    # each tier), then icp_score ascending within those groups. This surfaces
    # "review before removing" rows at the top of every tier block. ──────
    remove["_has_hint"] = (remove["rescue_hint"] != "").astype(int)
    remove = remove.sort_values(
        ["tier", "_has_hint", "icp_score", "full_name"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    remove = remove.drop(columns=["_has_hint"])

    # Reorder columns for review readability
    preferred_cols = [
        "tier", "rescue_hint", "icp_score", "full_name", "position", "company",
        "url", "category", "triage_reason", "reasoning", "connected_on",
        "profile_id",
    ]
    cols = [c for c in preferred_cols if c in remove.columns]
    remove = remove[cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    remove.to_csv(output_path, index=False)
    log.info("Wrote %d unfollow candidates to %s", len(remove), output_path)

    # Print tier breakdown for the user
    breakdown = remove["tier"].value_counts().sort_index()
    log.info("Tier breakdown:")
    for tier, count in breakdown.items():
        log.info("  %-17s %5d", tier, count)

    # Rescue-hint breakdown
    hinted = remove[remove["rescue_hint"] != ""]
    if not hinted.empty:
        log.info("Rescue-hint breakdown (review before removing):")
        hint_counts: dict[str, int] = {}
        for h in hinted["rescue_hint"]:
            for part in str(h).split("+"):
                hint_counts[part] = hint_counts.get(part, 0) + 1
        for h, count in sorted(hint_counts.items(), key=lambda x: -x[1]):
            log.info("  %-20s %5d", h, count)

    return remove


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    build_unfollow_list()


if __name__ == "__main__":
    main()
