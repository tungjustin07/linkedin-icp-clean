"""
Data ingestion and normalization for the LinkedIn audit pipeline.

Handles LinkedIn's quirky CSV export format, normalizes field names,
derives stable profile IDs, and validates the ICP config.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

log = logging.getLogger("linkedin_audit.ingest")

# ── Expected columns after normalization ─────────────────────────────────────

CONNECTIONS_COLUMNS = [
    "profile_id",
    "first_name",
    "last_name",
    "full_name",
    "url",
    "email",
    "company",
    "position",
    "connected_on",
]

CUSTOMERS_COLUMNS = [
    "profile_id",
    "full_name",
    "company",
    "position",
    "url",
    "notes",
]

TARGETS_COLUMNS = [
    "profile_id",
    "full_name",
    "company",
    "position",
    "url",
    "priority_note",
]

MESSAGES_COLUMNS = [
    "profile_id",
    "messages_sent",       # messages the user sent to this person
    "messages_received",   # messages this person sent to the user
    "thread_count",        # distinct conversation threads
    "last_contact_date",   # most recent message date (datetime)
    "days_since_contact",  # integer days since last message
    "initiated_first",     # True if user sent the first message
]

# Map of LinkedIn export header names → internal snake_case
_LINKEDIN_COL_MAP = {
    "first name": "first_name",
    "last name": "last_name",
    "url": "url",
    "email address": "email",
    "company": "company",
    "position": "position",
    "connected on": "connected_on",
}


# ── Public loaders ────────────────────────────────────────────────────────────


def load_connections(path: str | Path) -> pd.DataFrame:
    """
    Load a LinkedIn connections export CSV.

    LinkedIn's export may begin with a 3-line preamble ("Notes:", blank, blank)
    before the real header row. Both formats are handled automatically.

    Returns a DataFrame with CONNECTIONS_COLUMNS.
    Duplicates (by profile_id) are dropped, keeping the most recent connection.
    """
    path = Path(path)
    log.info("Loading connections from %s", path)

    raw = _read_linkedin_csv(path)
    df = _normalize_linkedin_csv(raw)

    # Derive full_name and profile_id
    df["full_name"] = (
        df["first_name"].fillna("") + " " + df["last_name"].fillna("")
    ).str.strip()
    df["profile_id"] = df.apply(
        lambda r: _extract_profile_id(r["url"]) if r.get("url") else _stable_id_from_row(r),
        axis=1,
    )

    # Parse connected_on — LinkedIn format: "15 Jan 2024" or "2024-01-15"
    df["connected_on"] = pd.to_datetime(df["connected_on"], dayfirst=True, errors="coerce")

    # Fill sparse fields
    df["company"] = df["company"].fillna("").str.strip()
    df["position"] = df["position"].fillna("").str.strip()
    df["email"] = df["email"].fillna("").str.strip()

    # Dedup: keep the most recently connected row per profile_id
    df = df.sort_values("connected_on", ascending=False, na_position="last")
    df = df.drop_duplicates(subset="profile_id", keep="first")

    df = df[CONNECTIONS_COLUMNS].reset_index(drop=True)
    log.info("Loaded %d unique connections", len(df))
    return df


def load_customers(path: Optional[str | Path]) -> pd.DataFrame:
    """
    Load known-good customer / ICP-example CSV.

    Accepts the same LinkedIn export format or any CSV with at least
    Name/Company/Position columns. Missing columns are filled with "".
    Returns empty DataFrame with CUSTOMERS_COLUMNS if path is None.
    """
    if path is None:
        return pd.DataFrame(columns=CUSTOMERS_COLUMNS)

    path = Path(path)
    if not path.exists():
        log.warning("Customers file not found: %s — proceeding without examples", path)
        return pd.DataFrame(columns=CUSTOMERS_COLUMNS)

    log.info("Loading customers from %s", path)
    raw = _read_linkedin_csv(path)
    df = _normalize_linkedin_csv(raw)

    # Handle minimal CSVs (just name/company/position)
    if "first_name" in df.columns and "last_name" in df.columns:
        df["full_name"] = (
            df["first_name"].fillna("") + " " + df["last_name"].fillna("")
        ).str.strip()
    elif "full_name" not in df.columns:
        # Try common alternative column names
        for col in ("name", "contact name", "full name"):
            if col in df.columns:
                df["full_name"] = df[col]
                break
        else:
            df["full_name"] = ""

    df["profile_id"] = df.apply(
        lambda r: _extract_profile_id(r.get("url", "")) or _stable_id_from_row(r),
        axis=1,
    )

    for col in ("company", "position", "url", "notes"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").str.strip()

    df = df[CUSTOMERS_COLUMNS].drop_duplicates(subset="profile_id").reset_index(drop=True)
    log.info("Loaded %d customer examples", len(df))
    return df


def load_targets(path: Optional[str | Path]) -> pd.DataFrame:
    """
    Load optional targets CSV (connections to make).

    Returns empty DataFrame with TARGETS_COLUMNS if path is None or missing.
    """
    if path is None:
        return pd.DataFrame(columns=TARGETS_COLUMNS)

    path = Path(path)
    if not path.exists():
        log.warning("Targets file not found: %s — no targets will be scored", path)
        return pd.DataFrame(columns=TARGETS_COLUMNS)

    log.info("Loading targets from %s", path)
    raw = _read_linkedin_csv(path)
    df = _normalize_linkedin_csv(raw)

    if "first_name" in df.columns and "last_name" in df.columns:
        df["full_name"] = (
            df["first_name"].fillna("") + " " + df["last_name"].fillna("")
        ).str.strip()
    elif "full_name" not in df.columns:
        for col in ("name", "contact name", "full name"):
            if col in df.columns:
                df["full_name"] = df[col]
                break
        else:
            df["full_name"] = ""

    df["profile_id"] = df.apply(
        lambda r: _extract_profile_id(r.get("url", "")) or _stable_id_from_row(r),
        axis=1,
    )

    for col in ("company", "position", "url", "priority_note"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").str.strip()

    df = df[TARGETS_COLUMNS].drop_duplicates(subset="profile_id").reset_index(drop=True)
    log.info("Loaded %d targets", len(df))
    return df


def load_messages(path: Optional[str | Path]) -> pd.DataFrame:
    """
    Parse LinkedIn messages.csv export into per-profile engagement stats.

    LinkedIn messages.csv columns:
      CONVERSATION ID, CONVERSATION TITLE, FROM, SENDER PROFILE URL,
      TO, RECIPIENT PROFILE URLS, DATE, SUBJECT, CONTENT, FOLDER

    Detects the user's own profile URL as the most-frequent sender across all
    messages (heuristic — works well for active LinkedIn users).

    Returns a DataFrame with MESSAGES_COLUMNS, one row per contact profile_id.
    Profiles with no message history are not included (left-join in callers).
    """
    if path is None:
        return pd.DataFrame(columns=MESSAGES_COLUMNS)

    path = Path(path)
    if not path.exists():
        log.warning("Messages file not found: %s — proceeding without message signals", path)
        return pd.DataFrame(columns=MESSAGES_COLUMNS)

    log.info("Loading messages from %s", path)
    df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")

    # Normalize column names
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

    # Required columns — bail gracefully if format is unexpected
    required = {"SENDER_PROFILE_URL", "RECIPIENT_PROFILE_URLS", "DATE", "CONVERSATION_ID"}
    missing = required - set(df.columns)
    if missing:
        log.warning("messages.csv missing columns %s — skipping message signals", missing)
        return pd.DataFrame(columns=MESSAGES_COLUMNS)

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", utc=True)
    df = df.dropna(subset=["SENDER_PROFILE_URL"])

    # Detect the user's own URL: most frequent sender
    sender_counts = df["SENDER_PROFILE_URL"].value_counts()
    my_url = sender_counts.index[0] if not sender_counts.empty else ""
    my_slug = _extract_profile_id(my_url)
    log.info("Detected user's LinkedIn URL as: %s", my_url)

    records: dict[str, dict] = {}  # profile_id → stats

    for _, row in df.iterrows():
        sender_url = str(row.get("SENDER_PROFILE_URL", "")).strip()
        sender_slug = _extract_profile_id(sender_url)
        conv_id = str(row.get("CONVERSATION_ID", ""))
        date = row.get("DATE")

        # Parse recipient URLs (may be comma-separated for group chats)
        raw_recipients = str(row.get("RECIPIENT_PROFILE_URLS", ""))
        recipient_slugs = [
            _extract_profile_id(u.strip())
            for u in raw_recipients.split(",")
            if u.strip()
        ]

        is_from_me = (sender_slug == my_slug or sender_slug == "")

        # Determine the "other person" in this message exchange
        if is_from_me:
            others = [s for s in recipient_slugs if s and s != my_slug]
        else:
            others = [sender_slug] if sender_slug and sender_slug != my_slug else []

        for other_slug in others:
            if not other_slug:
                continue
            if other_slug not in records:
                records[other_slug] = {
                    "profile_id": other_slug,
                    "messages_sent": 0,
                    "messages_received": 0,
                    "threads": set(),
                    "last_contact_date": None,
                    "first_message_from_me": None,
                    "first_message_date": None,
                }
            r = records[other_slug]
            if is_from_me:
                r["messages_sent"] += 1
            else:
                r["messages_received"] += 1
            r["threads"].add(conv_id)
            if date and pd.notna(date):
                if r["last_contact_date"] is None or date > r["last_contact_date"]:
                    r["last_contact_date"] = date
                if r["first_message_date"] is None or date < r["first_message_date"]:
                    r["first_message_date"] = date
                    r["first_message_from_me"] = is_from_me

    if not records:
        log.info("No message engagement data found")
        return pd.DataFrame(columns=MESSAGES_COLUMNS)

    today = pd.Timestamp.now(tz="UTC")
    rows = []
    for slug, r in records.items():
        last = r["last_contact_date"]
        days_since = int((today - last).days) if last else 9999
        rows.append({
            "profile_id": slug,
            "messages_sent": r["messages_sent"],
            "messages_received": r["messages_received"],
            "thread_count": len(r["threads"]),
            "last_contact_date": last,
            "days_since_contact": days_since,
            "initiated_first": bool(r.get("first_message_from_me")),
        })

    result = pd.DataFrame(rows)
    log.info(
        "Message engagement: %d profiles with DM history (%d total messages)",
        len(result),
        result["messages_sent"].sum() + result["messages_received"].sum(),
    )
    return result


def load_icp_config(path: str | Path) -> dict:
    """
    Load and validate icp_config.yaml.

    Raises ValueError with a descriptive message if required keys are missing.
    """
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f)

    icp = config.get("icp")
    if not icp:
        raise ValueError(f"icp_config.yaml must have a top-level 'icp' key: {path}")

    required = ["description", "target_titles", "target_industries", "scoring_weights",
                "triage_pass_threshold", "keep_score_threshold"]
    missing = [k for k in required if k not in icp]
    if missing:
        raise ValueError(f"icp_config.yaml missing required keys under 'icp': {missing}")

    log.info(
        "ICP config loaded: %d titles, %d industries, threshold=%s",
        len(icp.get("target_titles", [])),
        len(icp.get("target_industries", [])),
        icp.get("keep_score_threshold"),
    )
    return config


# ── Internal helpers ──────────────────────────────────────────────────────────


def _read_linkedin_csv(path: Path) -> pd.DataFrame:
    """
    Read a LinkedIn CSV export, skipping any preamble lines before the header.

    LinkedIn sometimes prepends metadata lines like:
        Notes: To see a ...
        (blank)
        (blank)
    before the real "First Name,Last Name,..." header.
    """
    # Sniff the first few lines to detect preamble
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = [f.readline() for _ in range(5)]

    skip = 0
    for i, line in enumerate(lines):
        # The real header always starts with "First Name" or "first name"
        if re.match(r"(?i)first\s*name", line.strip().lstrip("\ufeff")):
            skip = i
            break
        # Also handle CSVs that start with "Name" or other variations
        if re.match(r"(?i)(name|full.?name)", line.strip().lstrip("\ufeff")):
            skip = i
            break

    return pd.read_csv(path, skiprows=skip, encoding="utf-8-sig", on_bad_lines="skip")


def _normalize_linkedin_csv(raw: pd.DataFrame) -> pd.DataFrame:
    """Map LinkedIn export column names → internal snake_case names."""
    col_map = {col: _LINKEDIN_COL_MAP.get(col.lower().strip(), col.lower().strip().replace(" ", "_"))
               for col in raw.columns}
    return raw.rename(columns=col_map)


def _extract_profile_id(url: str) -> str:
    """
    Extract LinkedIn profile slug from a URL.

    https://www.linkedin.com/in/john-doe-abc123/ → "john-doe-abc123"
    Falls back to md5(url) for non-standard URLs.
    Returns "" if url is empty/None.
    """
    if not url or not isinstance(url, str):
        return ""
    url = url.strip().rstrip("/")
    match = re.search(r"linkedin\.com/in/([^/?#]+)", url, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Non-standard URL — hash it for a stable ID
    return "url_" + hashlib.md5(url.encode()).hexdigest()[:12]


def _stable_id_from_row(row: pd.Series) -> str:
    """
    Fallback profile_id when URL is absent: md5(full_name + company).
    """
    key = (str(row.get("full_name", "")) + str(row.get("company", ""))).strip().lower()
    if not key:
        key = str(row.get("first_name", "")) + str(row.get("last_name", ""))
    return "hash_" + hashlib.md5(key.encode()).hexdigest()[:12]
