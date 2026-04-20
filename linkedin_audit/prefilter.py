"""
Tier-0 deterministic pre-filter: drop obvious dead-weight connections at $0
before any Claude API call.

Three conservative rules — a profile is dropped ONLY if:
  EMPTY_PROFILE  — position AND company are both blank (no data to score on).
  COLD_STALE     — no message history, connected >N days ago, AND no ICP-adjacent
                   keyword in position/company (safety net: "VP RevOps at Stripe"
                   stays even if never DMed).
  SPAM_PATTERN   — full_name looks like a bot entry (URL, email, all-caps >3 words,
                   or spam keyword).

Anyone with DM history OR an ICP keyword hit OR a recent connection passes through
untouched. The goal is to remove the obvious zero-value rows before Haiku triage,
not to make scoring decisions.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

log = logging.getLogger("linkedin_audit.prefilter")

# Drop-reason tags written to the dropped DataFrame
DROP_EMPTY_PROFILE = "EMPTY_PROFILE"
DROP_COLD_STALE = "COLD_STALE"
DROP_SPAM_PATTERN = "SPAM_PATTERN"

# Obvious bot / spam markers in the full_name field
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_SPAM_KEYWORDS = (
    "click here",
    "buy now",
    "discount",
    "free trial",
    "limited offer",
    "follow me",
    "dm me",
    "whatsapp",
)

# Rescue regex patterns — profiles whose title/company matches any of these
# are exempt from COLD_STALE even if they don't match an ICP title/industry.
# These are connections worth keeping beyond strict ICP fit (VCs as good
# introducers, IT leaders as potential hirers/peers). Defined here rather
# than in icp_config.yaml because they are NOT buyer signals — we don't want
# Sonnet to score them as primary ICP, just don't want prefilter to bulk-drop
# them. All patterns are matched against a lowercased "position company" string.
_RESCUE_PATTERNS = (
    # VC / investor titles
    r"\bventure\b", r"\bventures\b", r"\bcapital\b", r"\binvestor\b",
    r"\bgeneral partner\b", r"\bmanaging partner\b", r"\bfounding partner\b",
    r"\bventure partner\b", r"\bangel investor\b", r"\bgrowth equity\b",
    r"\blimited partner\b",
    # IT leadership titles (secondary buyer / potential hirer)
    r"\bvp of it\b", r"\bvp information technology\b", r"\bvp, it\b",
    r"\bhead of it\b", r"\bhead of information\b",
    r"\bdirector of it\b", r"\bdirector of information\b",
    r"\bchief information officer\b", r"\bcio\b",
    r"\bchief technology officer\b", r"\bcto\b",
)
# Pre-compile once at import time for speed on ~11k-row DataFrames.
_RESCUE_REGEX = re.compile("|".join(_RESCUE_PATTERNS), re.IGNORECASE)


def run_prefilter(
    connections: pd.DataFrame,
    messages: "pd.DataFrame | None" = None,
    icp_config: "dict | None" = None,
    min_connection_age_days: int = 730,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Apply deterministic drop rules before the AI pipeline.

    Args:
        connections: DataFrame with at least profile_id, full_name, position,
                     company, connected_on columns (i.e. CONNECTIONS_COLUMNS from ingest).
        messages: Optional per-profile engagement stats DataFrame. When provided,
                  profiles with any DM history are exempt from COLD_STALE.
        icp_config: Optional parsed icp_config.yaml dict. When provided, profiles
                    whose position/company substring-matches any target_titles or
                    target_industries keyword are exempt from COLD_STALE.
        min_connection_age_days: COLD_STALE threshold (default 730 = 2 years).

    Returns:
        (kept_df, dropped_df) — kept_df has same schema as `connections`.
        dropped_df has same schema + a `drop_reason` column.
    """
    if connections.empty:
        log.info("Pre-filter: empty connections — nothing to do")
        return connections.copy(), _empty_dropped_df(connections)

    df = connections.copy()

    # Messaged profile IDs (for COLD_STALE exemption)
    msged_ids: set[str] = set()
    if messages is not None and not messages.empty:
        msged_ids = set(messages["profile_id"].astype(str).tolist())

    # ICP keyword haystack (lowercased substrings; for COLD_STALE exemption)
    icp_keywords: list[str] = []
    if icp_config:
        icp = icp_config.get("icp", {})
        icp_keywords = [
            str(kw).lower().strip()
            for kw in (icp.get("target_titles", []) + icp.get("target_industries", []))
            if kw
        ]
        icp_keywords = [kw for kw in icp_keywords if kw]

    now = pd.Timestamp.now(tz="UTC")
    stale_cutoff = now - pd.Timedelta(days=min_connection_age_days)

    # ── Rule 1: EMPTY_PROFILE ────────────────────────────────────────────────
    empty_mask = (
        df["position"].fillna("").astype(str).str.strip().eq("")
        & df["company"].fillna("").astype(str).str.strip().eq("")
    )

    # ── Rule 3: SPAM_PATTERN ─────────────────────────────────────────────────
    spam_mask = df["full_name"].fillna("").astype(str).apply(_is_spam_name)

    # ── Rule 2: COLD_STALE ───────────────────────────────────────────────────
    haystack = (
        df["position"].fillna("").astype(str).str.lower()
        + " "
        + df["company"].fillna("").astype(str).str.lower()
    )
    if icp_keywords:
        icp_hit = haystack.apply(lambda s: any(kw in s for kw in icp_keywords))
    else:
        icp_hit = pd.Series(False, index=df.index)

    # Also exempt profiles that match a rescue pattern (VC / IT leader /
    # etc.) — these are worth keeping beyond strict ICP fit.
    rescue_hit = haystack.str.contains(_RESCUE_REGEX, regex=True, na=False)
    keyword_hit = icp_hit | rescue_hit

    connected_on = pd.to_datetime(df["connected_on"], errors="coerce", utc=True)
    stale_connection = connected_on.notna() & (connected_on < stale_cutoff)
    no_msg_history = ~df["profile_id"].astype(str).isin(msged_ids)

    cold_mask = no_msg_history & stale_connection & ~keyword_hit

    # Reason priority: EMPTY > SPAM > COLD. An empty profile is "more dropped"
    # than a cold one — assign the strongest reason last so it overrides.
    reasons = pd.Series("", index=df.index, dtype=object)
    reasons[cold_mask] = DROP_COLD_STALE
    reasons[spam_mask] = DROP_SPAM_PATTERN
    reasons[empty_mask] = DROP_EMPTY_PROFILE

    drop_mask = reasons != ""

    kept = df[~drop_mask].reset_index(drop=True)
    dropped = df[drop_mask].copy()
    dropped["drop_reason"] = reasons[drop_mask].to_numpy()
    dropped = dropped.reset_index(drop=True)

    counts = dropped["drop_reason"].value_counts().to_dict() if not dropped.empty else {}
    log.info(
        "Pre-filter: kept %d / %d connections | dropped %d (%s=%d, %s=%d, %s=%d)",
        len(kept),
        len(df),
        len(dropped),
        DROP_EMPTY_PROFILE,
        counts.get(DROP_EMPTY_PROFILE, 0),
        DROP_COLD_STALE,
        counts.get(DROP_COLD_STALE, 0),
        DROP_SPAM_PATTERN,
        counts.get(DROP_SPAM_PATTERN, 0),
    )

    return kept, dropped


# ── Internal helpers ──────────────────────────────────────────────────────────


def _empty_dropped_df(connections: pd.DataFrame) -> pd.DataFrame:
    """Return an empty dropped_df with the right schema."""
    cols = list(connections.columns) + ["drop_reason"]
    return pd.DataFrame(columns=cols)


def _is_spam_name(name: str) -> bool:
    """Return True if the full_name field looks like a bot/spam entry."""
    if not name:
        return False
    s = name.strip()
    if not s:
        return False
    if _URL_RE.search(s):
        return True
    if _EMAIL_RE.search(s):
        return True
    # All-caps with >3 words (e.g., "BUY NOW LIMITED TIME OFFER")
    if s.isupper() and len(s.split()) > 3:
        return True
    lower = s.lower()
    if any(kw in lower for kw in _SPAM_KEYWORDS):
        return True
    return False
