# linkedin-icp-clean

LinkedIn 1st-degree connection scorer for a GTM systems / RevOps consulting practice. Takes a LinkedIn data export, classifies each connection against a two-track ICP, and outputs a ranked outreach list with reasoning.

## What it does

1. **Ingest.** Reads a LinkedIn `Connections.csv` export plus auxiliary signals (post-engagement history, mutual context, prior coworker overlap).
2. **Pre-filter.** Drops connections that are obviously out of ICP (founders/CEOs of pre-RevOps companies, pure BI sales, students, recruiters) before paying for an LLM call.
3. **Score with Claude.** Each surviving lead is scored against the ICP defined in `icp_config.yaml` — two parallel buyer tracks:
   - **Track 1:** Traditional RevOps leadership (VP/Head/Director Revenue Ops or Sales Ops)
   - **Track 2:** Technical GTM Systems leadership (GTM Engineering Lead, RevOps Architect, Manager Business Systems, etc.)
   Past coworkers and ops/systems alumni from technically sophisticated B2B SaaS are scored highest regardless of current title.
4. **Rescue passes.** Targeted re-scoring for VC introducer, engineering leader, and ICP-edge cases that would otherwise be misclassified.
5. **Cache + cost-control.** Sonnet scoring uses prompt caching; results cached to disk so re-runs only re-score deltas.
6. **Output.** Ranked CSV/Parquet with score, reasoning, and outreach angle. Optional Supabase + Voyage embedding layer for semantic search across the corpus.

## Layout

```
icp_config.yaml          # ICP definition + scoring rubric (the one file to tune)
linkedin_audit/          # core package: scoring, rescue passes, cache, prefilters
data/                    # LinkedIn export + auxiliary signals (gitignored)
cache/                   # per-row LLM result cache
output/                  # ranked outreach CSVs
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # ANTHROPIC_API_KEY, SUPABASE_URL, SUPABASE_KEY, VOYAGE_API_KEY (optional)
```

Drop your LinkedIn export into `data/Connections.csv` and any engagement export alongside it.

## Running

```bash
python -m linkedin_audit.run            # full pipeline
python -m linkedin_audit.run --rescue   # re-score the rescue cohort only
python -m linkedin_audit.run --since 2026-04-01   # delta run
```

## Notes

- The ICP in `icp_config.yaml` is opinionated and personal. It explicitly **excludes** founder/CEO sales, pre-RevOps companies, and pure BI/analytics deals. Edit before reusing.
- Prefilter rules are intentionally conservative — when in doubt, send to scoring. Cheap to score, expensive to miss a good lead.
- VC firm Principals were misclassified as VC_INTRODUCER in early versions; the rescue pass now handles this. If you fork, keep the rescue pass.
