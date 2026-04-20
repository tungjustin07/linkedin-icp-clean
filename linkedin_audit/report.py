"""
Output generation for the LinkedIn audit pipeline.

Merges triage + score results, writes keep/remove/targets CSVs,
and prints a console summary report in the style of the original post.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

log = logging.getLogger("linkedin_audit.report")

OUTPUT_DIR = Path("output")


# ── Public API ────────────────────────────────────────────────────────────────


def generate_outputs(
    connections: pd.DataFrame,
    triage_results: pd.DataFrame,
    score_results: pd.DataFrame,
    targets: pd.DataFrame,
    icp_config: dict,
    output_dir: Path = OUTPUT_DIR,
    dry_run: bool = False,
    prefilter_dropped: "pd.DataFrame | None" = None,
) -> dict:
    """
    Merge all results, write CSVs, print summary.

    `prefilter_dropped` (optional): rows dropped by the deterministic pre-filter
    before any AI call. They are appended to remove.csv with category="PREFILTERED"
    and triage_reason=drop_reason so the final unfollow list remains complete.

    Returns a summary dict with audit metrics.
    """
    icp = icp_config["icp"]
    keep_threshold = icp.get("keep_score_threshold", 65)

    # Build output DataFrames
    keep_df = _build_keep_df(connections, triage_results, score_results, keep_threshold)
    remove_df = _build_remove_df(
        connections, triage_results, score_results, keep_threshold, prefilter_dropped
    )
    targets_df = _score_targets(targets, icp) if not targets.empty else pd.DataFrame()

    # Write outputs
    _write_csv(keep_df, output_dir / "keep.csv", dry_run)
    _write_csv(remove_df, output_dir / "remove.csv", dry_run)
    if not targets_df.empty:
        _write_csv(targets_df, output_dir / "targets_prioritized.csv", dry_run)

    # Build summary stats
    total = len(connections)
    removal_count = len(remove_df)
    keep_count = len(keep_df)
    pct_removed = (removal_count / total * 100) if total > 0 else 0.0

    # ICP density: % of kept connections scoring >= threshold
    if not score_results.empty and keep_count > 0:
        kept_ids = set(keep_df["profile_id"].tolist()) if "profile_id" in keep_df.columns else set()
        scored_kept = score_results[score_results["profile_id"].isin(kept_ids)]
        icp_count = len(scored_kept[scored_kept["icp_score"] >= keep_threshold])
        icp_density = (icp_count / keep_count * 100) if keep_count > 0 else 0.0
    else:
        icp_density = 0.0

    # Category breakdown
    category_breakdown = {}
    if not triage_results.empty:
        cats = triage_results["category"].value_counts()
        category_breakdown = cats.to_dict()

    # Top 10 ICP connections (by score)
    top_icp: list[dict] = []
    if not score_results.empty and not connections.empty:
        merged_top = connections.merge(score_results, on="profile_id", how="inner")
        merged_top = merged_top.sort_values("icp_score", ascending=False).head(10)
        for _, r in merged_top.iterrows():
            top_icp.append({
                "full_name": r.get("full_name", ""),
                "position": r.get("position", ""),
                "company": r.get("company", ""),
                "icp_score": r.get("icp_score", 0),
                "recommendation": r.get("recommendation", ""),
            })

    summary = {
        "total_connections": total,
        "keep_count": keep_count,
        "removal_count": removal_count,
        "pct_removed": pct_removed,
        "icp_density": icp_density,
        "keep_score_threshold": keep_threshold,
        "category_breakdown": category_breakdown,
        "top_icp_profiles": top_icp,
        "targets_scored_count": len(targets_df) if not targets_df.empty else 0,
    }

    _print_summary(summary, output_dir, dry_run)
    return summary


# ── DataFrame builders ────────────────────────────────────────────────────────


def _build_keep_df(
    connections: pd.DataFrame,
    triage_results: pd.DataFrame,
    score_results: pd.DataFrame,
    keep_threshold: int,
) -> pd.DataFrame:
    """
    Build keep.csv.
    Includes profiles where icp_score >= threshold OR recommendation in KEEP_*.
    Profiles that passed triage but have no Sonnet score (shouldn't happen normally)
    are included if their category is ICP_CORE or ICP_ADJACENT.
    """
    # Start with all connections, join triage and score
    df = connections.merge(
        triage_results[["profile_id", "category", "triage_pass", "triage_reason"]],
        on="profile_id", how="left",
    )
    df = df.merge(
        score_results[["profile_id", "icp_score", "recommendation", "reasoning", "action_items"]]
        if not score_results.empty else pd.DataFrame(columns=["profile_id", "icp_score", "recommendation", "reasoning", "action_items"]),
        on="profile_id", how="left",
    )

    keep_mask = (
        (df["icp_score"] >= keep_threshold) |
        (df["recommendation"].isin(["KEEP_ENGAGE", "KEEP_NURTURE"])) |
        # No score yet but strong triage category
        (df["icp_score"].isna() & df["category"].isin(["ICP_CORE", "ICP_ADJACENT", "PARTNER"]))
    )

    keep = df[keep_mask].copy()
    keep["icp_score"] = keep["icp_score"].fillna(-1).astype(int)
    keep["recommendation"] = keep["recommendation"].fillna("UNSCORED")
    keep["reasoning"] = keep["reasoning"].fillna("")
    keep["action_items"] = keep["action_items"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else (x or "")
    )
    keep["category"] = keep["category"].fillna("UNKNOWN")

    cols = ["profile_id", "full_name", "company", "position", "url",
            "category", "icp_score", "recommendation", "reasoning", "action_items", "connected_on"]
    cols = [c for c in cols if c in keep.columns]
    return keep[cols].sort_values("icp_score", ascending=False).reset_index(drop=True)


def _build_remove_df(
    connections: pd.DataFrame,
    triage_results: pd.DataFrame,
    score_results: pd.DataFrame,
    keep_threshold: int,
    prefilter_dropped: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """
    Build remove.csv.
    Includes: pre-filtered drops (if any) + triage failures + Sonnet REMOVE recommendations.
    """
    df = connections.merge(
        triage_results[["profile_id", "category", "triage_pass", "triage_reason"]],
        on="profile_id", how="left",
    )
    df = df.merge(
        score_results[["profile_id", "icp_score", "recommendation", "reasoning"]]
        if not score_results.empty else pd.DataFrame(columns=["profile_id", "icp_score", "recommendation", "reasoning"]),
        on="profile_id", how="left",
    )

    remove_mask = (
        (df["triage_pass"] == False) |  # noqa: E712
        (df["recommendation"] == "REMOVE")
    )

    remove = df[remove_mask].copy()
    remove["icp_score"] = remove["icp_score"].fillna(-1).astype(int)
    remove["recommendation"] = remove["recommendation"].fillna("FAILED_TRIAGE")
    remove["reasoning"] = remove["reasoning"].fillna(remove["triage_reason"].fillna(""))
    remove["category"] = remove["category"].fillna("UNKNOWN")

    # Append pre-filter drops so the unfollow list is complete.
    if prefilter_dropped is not None and not prefilter_dropped.empty:
        pf = prefilter_dropped.copy()
        drop_reason = pf["drop_reason"] if "drop_reason" in pf.columns else ""
        pf["category"] = "PREFILTERED"
        pf["triage_pass"] = False
        pf["triage_reason"] = drop_reason
        pf["icp_score"] = -1
        pf["recommendation"] = "PREFILTER_DROP"
        pf["reasoning"] = drop_reason
        remove = pd.concat([remove, pf], ignore_index=True, sort=False)
        remove["icp_score"] = remove["icp_score"].fillna(-1).astype(int)

    cols = ["profile_id", "full_name", "company", "position", "url",
            "category", "triage_reason", "icp_score", "reasoning", "connected_on"]
    cols = [c for c in cols if c in remove.columns]
    return remove[cols].reset_index(drop=True)


def _score_targets(targets: pd.DataFrame, icp: dict) -> pd.DataFrame:
    """
    Rank targets_prioritized.csv using keyword matching — no API calls.
    Returns targets sorted by priority_score DESC.
    """
    titles = [t.lower() for t in icp.get("target_titles", [])]
    industries = [i.lower() for i in icp.get("target_industries", [])]
    anti = [a.lower() for a in icp.get("anti_patterns", [])]

    def _score_row(row) -> tuple[int, str]:
        position = str(row.get("position", "")).lower()
        company = str(row.get("company", "")).lower()
        reasons = []
        score = 0

        # Anti-patterns — hard zero
        if any(a in position for a in anti):
            return 0, "anti-pattern match"

        # Title match
        title_hits = [t for t in titles if t in position]
        if title_hits:
            score += 50
            reasons.append(f"title match: {title_hits[0]}")

        # Industry / company keyword match
        industry_hits = [i for i in industries if i in company or i in position]
        if industry_hits:
            score += 30
            reasons.append(f"industry signal: {industry_hits[0]}")

        # Priority note present
        if str(row.get("priority_note", "")).strip():
            score += 10
            reasons.append("has priority note")

        return score, "; ".join(reasons) if reasons else "no strong signal"

    scores, reasons = zip(*targets.apply(_score_row, axis=1)) if len(targets) > 0 else ([], [])
    result = targets.copy()
    result["priority_score"] = list(scores)
    result["match_reason"] = list(reasons)

    cols = ["full_name", "company", "position", "url", "priority_score", "match_reason", "priority_note"]
    cols = [c for c in cols if c in result.columns]
    return result[cols].sort_values("priority_score", ascending=False).reset_index(drop=True)


# ── Console output ────────────────────────────────────────────────────────────


def _print_summary(summary: dict, output_dir: Path, dry_run: bool) -> None:
    total = summary["total_connections"]
    removal = summary["removal_count"]
    keep = summary["keep_count"]
    pct_removed = summary["pct_removed"]
    icp_density = summary["icp_density"]
    threshold = summary["keep_score_threshold"]
    cats = summary["category_breakdown"]
    top = summary["top_icp_profiles"]
    targets_n = summary["targets_scored_count"]

    prefix = "[DRY-RUN] " if dry_run else ""
    out_label = "(not written — dry-run)" if dry_run else str(output_dir)

    width = 62
    print()
    print("=" * width)
    print(f"  {prefix}LINKEDIN NETWORK AUDIT — {date.today()}")
    print("=" * width)
    print(f"  Total connections analyzed : {total:,}")
    print(f"  Connections to KEEP        : {keep:,}")
    print(f"  Connections to REMOVE      : {removal:,}  ({pct_removed:.0f}%)")
    print(f"  ICP density (scored ≥{threshold}) : {icp_density:.0f}%")
    print()

    if cats:
        print("  Category Breakdown:")
        cat_order = ["ICP_CORE", "ICP_ADJACENT", "PARTNER", "VENDOR", "PERSONAL", "SPAM", "UNKNOWN"]
        for cat in cat_order:
            n = cats.get(cat, 0)
            if n:
                pct = n / total * 100
                bar = "█" * max(1, int(pct / 3))
                print(f"    {cat:<14} {bar:<20} {n:>5}  ({pct:.0f}%)")
        print()

    if top:
        print("  Top ICP Connections:")
        for i, p in enumerate(top, 1):
            print(f"    {i:>2}. {p['full_name']} — {p['position']} at {p['company']}  [score: {p['icp_score']}]")
        print()

    if targets_n:
        print(f"  Targets prioritized: {targets_n} written to {out_label}/targets_prioritized.csv")

    print(f"  Output: {out_label}/")
    print("=" * width)
    print()


def _write_csv(df: pd.DataFrame, path: Path, dry_run: bool) -> None:
    if dry_run:
        log.info("[DRY-RUN] Would write %d rows to %s — showing first 5:", len(df), path.name)
        print(df.head(5).to_string(index=False))
        print()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Wrote %d rows → %s", len(df), path)
