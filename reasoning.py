"""
reasoning.py

Generates the 1-2 sentence `reasoning` column. Stage 4 manual review
explicitly checks for: specific facts, JD connection, honest concerns
about gaps, no hallucination, variation across rows, and rank-consistent
tone. So this pulls only facts that are actually present in the
candidate's record (title, years, a matched skill group, a concrete
concern) rather than using one fill-in-the-blank template for everyone.
"""

import random

MATCHED_GROUP_LABELS = {
    "embeddings_retrieval": "embeddings/retrieval",
    "vector_db_hybrid_search": "vector search infra",
    "python": "Python",
    "eval_frameworks": "ranking evaluation (NDCG/MRR/MAP)",
}


def _matched_groups(skill_features: dict, threshold: float = 0.5) -> list:
    return [
        MATCHED_GROUP_LABELS[g]
        for g, score in skill_features["must_have_group_scores"].items()
        if score >= threshold
    ]


def _concern_phrase(features: dict, score_breakdown: dict) -> str:
    beh = features["behavioral"]
    loc = features["location"]
    disq = score_breakdown["triggered_disqualifiers"]

    if disq:
        readable = disq[0].replace("_", " ")
        return f"flagged for {readable}"
    if beh["days_inactive"] > 90:
        return f"inactive on-platform for {int(beh['days_inactive'])} days"
    if not beh["open_to_work"]:
        return "not currently marked open to work"
    if beh["notice_period_days"] > 90:
        return f"long notice period ({int(beh['notice_period_days'])} days)"
    if not loc["is_india"] and not loc["willing_to_relocate"]:
        return "based outside India with no visa sponsorship available"
    if beh["recruiter_response_rate"] < 0.3:
        return f"low recruiter response rate ({beh['recruiter_response_rate']:.0%})"
    return ""


def generate_reasoning(candidate: dict, features: dict, score_breakdown: dict, rank: int) -> str:
    p = candidate["profile"]
    title = p["current_title"]
    company = p["current_company"]
    yoe = p["years_of_experience"]
    location = p["location"]

    matched = _matched_groups(features["skill"])
    shipped_hits = features["career"]["shipped_system_hits"]
    concern = _concern_phrase(features, score_breakdown)

    is_top_tier = rank <= 30
    is_mid_tier = 30 < rank <= 70

    parts = []

    if score_breakdown["is_honeypot"]:
        return (
            f"{title} at {company}, {yoe} yrs — profile shows internal inconsistencies "
            f"(skill/tenure or experience/history mismatch) consistent with a honeypot; "
            f"excluded from serious consideration despite surface keyword match."
        )

    lead = f"{title} at {company}, {yoe} yrs."
    parts.append(lead)

    if matched and (is_top_tier or is_mid_tier):
        parts.append(f"Evidenced strength in {', '.join(matched)}.")
    elif not matched:
        parts.append("Skills list doesn't show trust-weighted evidence of the JD's core stack.")

    if shipped_hits >= 2:
        parts.append("Career history describes hands-on work building ranking/retrieval/recommendation systems, matching the JD's core mandate.")
    elif shipped_hits == 1:
        parts.append("Some career-history evidence of relevant systems work, though not extensive.")
    elif is_top_tier:
        parts.append("Title and skill signals are relevant but role descriptions don't explicitly describe shipping a ranking/retrieval system.")

    if location.lower() and is_top_tier:
        parts.append(f"Located in {location}.")

    if concern:
        parts.append(f"Concern: {concern}.")
    elif is_top_tier:
        parts.append("No major availability or fit concerns surfaced.")

    return " ".join(parts)
