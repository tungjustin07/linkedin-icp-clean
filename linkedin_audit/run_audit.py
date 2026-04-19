"""
LinkedIn Network Audit — CLI orchestrator.

Usage:
    python -m linkedin_audit.run_audit \\
        --connections data/connections.csv \\
        --customers   data/customers.csv \\
        --targets     data/targets.csv \\
        --icp-config  icp_config.yaml \\
        [--dry-run]
        [--skip-triage]
        [--skip-scoring]
        [--output-dir output/]
        [--cache-dir  cache/]

Pipeline steps:
    1. Ingest: load all CSVs + icp_config
    2. Triage: Haiku batch classification (20 profiles/call)
    3. Score:  Sonnet deep ICP scoring (profiles that passed triage)
    4. Report: write keep.csv / remove.csv / targets_prioritized.csv + console summary
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("linkedin_audit.run_audit")


def main() -> None:
    args = _parse_args()
    _validate_inputs(args)

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)

    from linkedin_audit import ingest, triage, score, report

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    log.info("Step 1/4: Loading data...")
    try:
        connections = ingest.load_connections(args.connections)
        customers = ingest.load_customers(args.customers)
        targets = ingest.load_targets(args.targets)
        messages = ingest.load_messages(args.messages)
        icp_config = ingest.load_icp_config(args.icp_config)
    except (FileNotFoundError, ValueError) as e:
        log.error("Data loading failed: %s", e)
        sys.exit(1)

    log.info(
        "  → %d connections | %d customers | %d targets | %d profiles with DM history",
        len(connections), len(customers), len(targets),
        len(messages) if not messages.empty else 0,
    )

    # ── Step 2: Triage ────────────────────────────────────────────────────────
    log.info("Step 2/4: Triage with Haiku...")
    if args.skip_triage:
        triage_results = triage.load_triage_cache(cache_dir / "triage.parquet")
        if triage_results.empty:
            log.error("--skip-triage set but no triage cache found at %s", cache_dir / "triage.parquet")
            sys.exit(1)
        log.info("  → Loaded %d triage results from cache", len(triage_results))
    else:
        try:
            triage_results = triage.run_triage(
                connections,
                icp_config,
                messages=messages if not messages.empty else None,
                cache_path=cache_dir / "triage.parquet",
                dry_run=args.dry_run,
                dry_run_sample=50,
            )
        except Exception as e:
            log.error("Triage failed: %s", e, exc_info=True)
            sys.exit(3)

    passed_count = triage_results["triage_pass"].sum() if not triage_results.empty else 0
    log.info("  → %d/%d profiles passed triage", passed_count, len(triage_results))

    # ── Step 3: Score ─────────────────────────────────────────────────────────
    log.info("Step 3/4: Deep scoring with Sonnet...")
    if args.skip_scoring:
        score_results = score.load_score_cache(cache_dir / "scores.parquet")
        if score_results.empty:
            log.error("--skip-scoring set but no score cache found at %s", cache_dir / "scores.parquet")
            sys.exit(1)
        log.info("  → Loaded %d score results from cache", len(score_results))
    else:
        try:
            score_results = score.run_scoring(
                connections,
                triage_results,
                icp_config,
                customers,
                messages=messages if not messages.empty else None,
                cache_path=cache_dir / "scores.parquet",
                dry_run=args.dry_run,
            )
        except Exception as e:
            log.error("Scoring failed: %s", e, exc_info=True)
            sys.exit(3)

    log.info("  → %d profiles scored", len(score_results))

    # ── Step 4: Report ────────────────────────────────────────────────────────
    log.info("Step 4/4: Generating outputs...")
    try:
        report.generate_outputs(
            connections,
            triage_results,
            score_results,
            targets,
            icp_config,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )
    except Exception as e:
        log.error("Report generation failed: %s", e, exc_info=True)
        sys.exit(3)

    log.info("Done.%s", " (dry-run — no files written)" if args.dry_run else "")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LinkedIn Network Audit — 2-tier AI pipeline to score and clean your network.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--connections", required=True,
        help="Path to LinkedIn connections CSV export (required)",
    )
    parser.add_argument(
        "--customers", default=None,
        help="Path to known-good customer/ICP examples CSV (optional but improves scoring)",
    )
    parser.add_argument(
        "--targets", default=None,
        help="Path to target connections CSV — people you want to connect with (optional)",
    )
    parser.add_argument(
        "--messages", default=None,
        help="Path to LinkedIn messages.csv export — DM history is a strong engagement signal (optional)",
    )
    parser.add_argument(
        "--icp-config", default="icp_config.yaml",
        help="Path to ICP config YAML (default: icp_config.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Score a sample of 50 profiles only — no files written, no cache updated",
    )
    parser.add_argument(
        "--skip-triage", action="store_true",
        help="Load triage results from cache only (skip Haiku API calls)",
    )
    parser.add_argument(
        "--skip-scoring", action="store_true",
        help="Load score results from cache only (skip Sonnet API calls)",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory for output CSVs (default: output/)",
    )
    parser.add_argument(
        "--cache-dir", default="cache",
        help="Directory for Parquet cache files (default: cache/)",
    )
    return parser.parse_args()


def _validate_inputs(args: argparse.Namespace) -> None:
    connections_path = Path(args.connections)
    if not connections_path.exists():
        log.error("Connections file not found: %s", connections_path)
        sys.exit(1)

    icp_path = Path(args.icp_config)
    if not icp_path.exists():
        log.error("ICP config not found: %s — copy icp_config.yaml and fill in your ICP", icp_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
