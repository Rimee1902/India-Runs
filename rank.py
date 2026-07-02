"""
rank.py

End-to-end ranking pipeline. Streams candidates.jsonl line by line (never
holds all 100K parsed records in memory at once - only a bounded top-K
heap), scores each candidate with fully local/offline logic (no network,
no GPU - see jd_config.py, features.py, scoring.py), and writes the
top-100 CSV in the exact format required by submission_spec.md.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Runtime target: well under the 5-minute / 16GB / CPU-only / no-network
budget - this is pure Python string/dict logic over JSON lines, no model
inference at all.
"""

import argparse
import csv
import heapq
import json
import sys
import time
from datetime import date

try:
    import resource  # Unix-only; used for memory reporting only, not required to run
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False  # e.g. Windows

import features as feat
import scoring as sc
from reasoning import generate_reasoning

TOP_K = 100
# Keep a slightly larger internal pool before final honeypot-aware sort,
# purely so ties / near-ties don't get lost to floating point ordering.
INTERNAL_POOL = 150


def load_candidates(path: str):
    """Generator - yields one parsed candidate dict at a time."""
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def score_candidate(candidate: dict, reference_date: date):
    features = feat.extract_all_features(candidate, reference_date)
    breakdown = sc.compute_score(features)
    return features, breakdown


def build_row(candidate: dict, features: dict, breakdown: dict, rank: int):
    return {
        "candidate_id": candidate["candidate_id"],
        "rank": rank,
        "score": breakdown["final_score"],
        "reasoning": generate_reasoning(candidate, features, breakdown, rank),
    }


def run(candidates_path: str, out_path: str, reference_date: date):
    start = time.time()
    heap = []  # min-heap of (score, tie_break_key, candidate_id, candidate, features, breakdown)
    counter = 0
    n_processed = 0
    n_honeypots_seen = 0

    for candidate in load_candidates(candidates_path):
        n_processed += 1
        features, breakdown = score_candidate(candidate, reference_date)
        if breakdown["is_honeypot"]:
            n_honeypots_seen += 1

        score = breakdown["final_score"]
        counter += 1
        entry = (score, -counter, candidate["candidate_id"], candidate, features, breakdown)

        if len(heap) < INTERNAL_POOL:
            heapq.heappush(heap, entry)
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, entry)

    elapsed = time.time() - start
    print(f"Processed {n_processed} candidates in {elapsed:.1f}s "
          f"({n_honeypots_seen} honeypot-flagged encountered).", file=sys.stderr)

    # Sort descending by score, tie-break by candidate_id ascending
    # (matches validate_submission.py's tie-break rule).
    ranked = sorted(heap, key=lambda e: (-e[0], e[2]))[:TOP_K]

    honeypots_in_top100 = sum(1 for e in ranked if e[5]["is_honeypot"])
    if honeypots_in_top100 > 0:
        print(f"WARNING: {honeypots_in_top100} honeypot(s) still in top 100 "
              f"after penalty - review scoring.py penalties.", file=sys.stderr)

    rows = []
    for i, (score, _, cid, candidate, features, breakdown) in enumerate(ranked, start=1):
        rows.append(build_row(candidate, features, breakdown, i))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote top {len(rows)} candidates to {out_path}", file=sys.stderr)
    if HAS_RESOURCE:
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Total wall time: {time.time() - start:.1f}s | Peak RSS: {peak_mb:.0f} MB", file=sys.stderr)
    else:
        print(f"Total wall time: {time.time() - start:.1f}s "
              f"(peak-memory reporting unavailable on this OS - use Task Manager if you want to check)",
              file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument(
        "--reference-date", default=None,
        help="YYYY-MM-DD to use as 'today' for recency calculations. "
             "Defaults to today's date.",
    )
    args = parser.parse_args()

    ref_date = date.today()
    if args.reference_date:
        ref_date = date.fromisoformat(args.reference_date)

    run(args.candidates, args.out, ref_date)


if __name__ == "__main__":
    main()
