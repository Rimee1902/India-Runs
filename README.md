# Redrob Intelligent Candidate Discovery & Ranking — Submission

Ranks the top 100 candidates from a 100,000-candidate pool against Redrob's
AI Engineer job description, without keyword matching — using a fully
local, deterministic, rule-based hybrid ranker that satisfies the
hackathon's compute constraints (≤5 min, ≤16GB RAM, CPU-only, no network
during ranking).

## Reproduce the submission

```bash
pip install -r requirements.txt   # stdlib-only for ranking; streamlit is only for the sandbox demo
python src/rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
```

No pre-computation step is required — no embeddings to precompute, no
model weights to download. `rank.py` streams `candidates.jsonl` line by
line and produces the ranked CSV directly.

**Measured performance on the full 100,000-candidate pool:** ~26 seconds
wall-clock, ~16 MB peak RSS, CPU-only, zero network calls.

## Why no embeddings model / no LLM at ranking time

The compute constraints (no GPU, no network, 5-minute budget for 100K
candidates) rule out per-candidate LLM calls and make a large embedding
model impractical to guarantee reproducibly across environments. Instead
of a black-box similarity score, we hand-encoded the JD's own stated
priorities (see `src/jd_config.py`) into explicit, interpretable features
— which also directly produces the "specific facts, no hallucination"
reasoning the Stage 4 manual review is checking for.

## Architecture

```
src/
├── jd_config.py   # Structured JD requirements, hand-encoded from job_description.md
├── features.py    # Per-candidate feature extraction (skills, career evidence,
│                  #   disqualifiers, location, experience, behavioral signals, honeypot detection)
├── scoring.py      # Combines features into one composite score (weighted sum,
│                  #   multiplicative disqualifier penalties, behavioral multiplier)
├── reasoning.py    # Generates the fact-specific `reasoning` column per row
└── rank.py         # Streaming pipeline: load → score → top-100 heap → CSV
```

### Scoring components (see `scoring.py` for exact weights)

1. **Must-have skill match (30%)** — trust-weighted: a skill only counts
   if it's evidenced by tenure (`duration_months`) or endorsements, not
   just listed. This dataset heavily uses randomized skill lists as
   decoys, so an unevidenced skill contributes almost nothing.
2. **Shipped-system evidence (20%)** — keyword evidence in career
   *descriptions* (not just skills) of having actually built
   ranking/retrieval/recommendation systems — the JD explicitly says this
   is what separates a real fit from a keyword-stuffed profile.
3. **Title relevance (10%)**, **experience-band fit (10%)**,
   **location fit (10%)**, **nice-to-have skills (5%)**.
4. **Disqualifier penalties (multiplicative)** — research-only,
   consulting-only career, CV/speech/robotics-only, framework-only recent
   AI experience, non-coding senior titles, title-chasing career pattern.
   Each maps to a specific sentence in the JD's "explicitly do NOT want"
   section.
5. **Behavioral multiplier** — availability/responsiveness signals
   (`open_to_work_flag`, recency of activity, recruiter response rate,
   interview completion rate, notice period) scale the fit score down,
   per the JD's own framing: "not actually available... down-weight them
   appropriately," not a hard exclusion.
6. **Honeypot gate** — profiles with ≥4 "expert" skills at 0 months
   tenure, or where `years_of_experience` is grossly inconsistent with
   summed `career_history` duration, are scored near-zero regardless of
   any other signal.

## Honeypot handling

`features.py::honeypot_features` implements both patterns named in
`redrob_signals_doc.md` / `README.md`: (1) many "expert"-proficiency
skills with zero tenure, and (2) `years_of_experience` inconsistent with
`career_history`. On the full 100K pool this flagged 59 candidates;
0 honeypots appeared in the final top 100.

## Sandbox

`app.py` is a Streamlit app that runs the same ranking logic on a small
sample (bundled `data/sample_candidates.json`, 50 candidates) and lets
you download a ranked CSV. Run locally with `streamlit run app.py`, or
deploy to Streamlit Community Cloud / HuggingFace Spaces pointing at this
file. A `Dockerfile` is included as a self-contained fallback per Section
10.5 (`docker build -t redrob-ranker . && docker run -p 8501:8501
redrob-ranker`).

## Known limitations / honest tradeoffs

- Keyword/phrase matching (not embeddings) means genuinely novel
  phrasing for a relevant skill could be missed. We accepted this
  tradeoff for reproducibility and interview-defensibility over recall.
- Disqualifier penalties and score weights are hand-set from the JD text,
  not learned — there's no labeled ground truth available to fit against.
- "Shipped system" evidence relies on career-history descriptions
  containing certain phrases; a candidate who did the work but described
  it differently could be under-scored on that component (though other
  components can still surface them).

## Dataset

`candidates.jsonl` (100K records, ~487MB) is **not** committed to this
repo — download it from the hackathon bundle and place it at the repo
root (or pass `--candidates <path>` to `rank.py`).
