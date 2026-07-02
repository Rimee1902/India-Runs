"""
app.py - Sandbox demo (Streamlit)

Satisfies submission_spec.md Section 10.5: accepts a small candidate
sample, runs the ranking system end-to-end, produces a ranked CSV,
completes well within the CPU compute budget.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud or HuggingFace Spaces (Streamlit SDK)
by pointing at this file - no secrets or network access required.
"""

import io
import json
import sys
from datetime import date

sys.path.insert(0, "src")

import features as feat
import scoring as sc
from reasoning import generate_reasoning

import streamlit as st

st.set_page_config(page_title="Redrob Candidate Ranker - Sandbox", layout="wide")
st.title("SmartHire / Redrob Candidate Ranker — Sandbox")
st.caption(
    "Fully local, rule-based ranking (no LLM calls, no GPU, no network at "
    "ranking time). Upload a small candidate sample (JSON array or JSONL, "
    "≤100 candidates) to see it run end-to-end."
)

uploaded = st.file_uploader("Upload sample_candidates.json or a .jsonl file", type=["json", "jsonl"])

default_path = "data/sample_candidates.json"

if uploaded is not None:
    raw = uploaded.read().decode("utf-8")
    if uploaded.name.endswith(".jsonl"):
        candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        candidates = json.loads(raw)
else:
    st.info("No file uploaded — using the bundled sample_candidates.json (50 candidates).")
    try:
        with open(default_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    except FileNotFoundError:
        candidates = []
        st.warning("Bundled sample file not found. Please upload a candidate file.")

if candidates:
    st.write(f"Loaded **{len(candidates)}** candidates.")
    reference_date = date.today()

    results = []
    for c in candidates:
        features = feat.extract_all_features(c, reference_date)
        breakdown = sc.compute_score(features)
        results.append((c, features, breakdown))

    results.sort(key=lambda r: (-r[2]["final_score"], r[0]["candidate_id"]))
    top_n = min(len(results), 100)

    rows = []
    for i, (c, features, breakdown) in enumerate(results[:top_n], start=1):
        rows.append({
            "candidate_id": c["candidate_id"],
            "rank": i,
            "score": breakdown["final_score"],
            "reasoning": generate_reasoning(c, features, breakdown, i),
        })

    st.subheader(f"Ranked output (top {top_n})")
    st.dataframe(rows, use_container_width=True)

    csv_buf = io.StringIO()
    import csv as csv_module
    writer = csv_module.DictWriter(csv_buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    writer.writerows(rows)

    st.download_button(
        "Download ranked CSV",
        data=csv_buf.getvalue(),
        file_name="sandbox_ranked_output.csv",
        mime="text/csv",
    )
else:
    st.warning("No candidates loaded yet.")
