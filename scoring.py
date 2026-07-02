"""
scoring.py

Combines the interpretable features from features.py into a single
composite score in [0, 1]. Weights are explicit constants (not learned -
we have no labeled training data) chosen to mirror the JD's own stated
priorities: must-have skills and genuine shipped-system evidence dominate;
disqualifiers apply multiplicative penalties (a hard-ish gate, not just a
subtracted constant, so a disqualified candidate can't be dragged back up
by an unrelated high score elsewhere); behavioral availability is a
multiplier on top of fit, exactly as the JD frames it ("down-weight them
appropriately", not "exclude them").
"""

WEIGHTS = {
    "must_have_skills": 0.30,
    "shipped_system_evidence": 0.20,
    "title_relevance": 0.10,
    "experience_fit": 0.10,
    "location_fit": 0.10,
    "nice_to_have": 0.05,
}

# Multiplicative penalty per disqualifier flag (applied to the fit score
# before the behavioral multiplier). Chosen to be harsh but not always
# zero, since the JD itself says these are "probably not" moves, not
# always-instant rejections (except research-only, which the JD states
# unconditionally: "we will not move forward").
DISQUALIFIER_PENALTIES = {
    "research_only": 0.05,        # JD is explicit/unconditional here
    "framework_only": 0.35,
    "consulting_only": 0.45,
    "cv_speech_only": 0.35,
    "non_coding_title": 0.55,
    "title_chaser": 0.55,
    "no_external_validation": 0.75,  # softest - "nice to have" evidence, not a hard rule
}


def _shipped_system_score(career: dict) -> float:
    hits = career["shipped_system_hits"]
    # Saturating: 0 hits -> 0, 1 hit -> 0.5, 3+ hits -> 1.0
    return min(1.0, hits / 3.0) if hits else 0.0


def _experience_fit_score(exp: dict) -> float:
    if exp["in_sweet_spot"]:
        return 1.0
    if exp["in_band"]:
        return 0.75
    # Outside band: decay gracefully rather than a hard cliff, since the
    # JD says "we'll seriously consider candidates outside the band if
    # other signals are strong."
    yoe = exp["years_of_experience"]
    dist = min(abs(yoe - 5), abs(yoe - 9))
    return max(0.2, 0.75 - 0.08 * dist)


def _location_fit_score(loc: dict) -> float:
    if loc["preferred_city"]:
        return 1.0
    if loc["tier1_city"]:
        return 0.85
    if loc["is_india"]:
        return 0.55
    # Outside India: JD says case-by-case, no visa sponsorship - heavy
    # penalty unless they've explicitly flagged willingness to relocate.
    return 0.25 if loc["willing_to_relocate"] else 0.08


def _behavioral_multiplier(beh: dict) -> float:
    """
    Multiplier in roughly [0.25, 1.05]. Not a component of the weighted
    sum - a multiplier, because the JD frames availability as gating
    ("not actually available") rather than as one input among many.
    """
    mult = 1.0

    if not beh["open_to_work"]:
        mult *= 0.55

    days = beh["days_inactive"]
    if days > 180:
        mult *= 0.5
    elif days > 90:
        mult *= 0.75
    elif days > 30:
        mult *= 0.92

    rr = beh["recruiter_response_rate"]
    mult *= 0.7 + 0.3 * rr  # scales 0.7x (never responds) to 1.0x (always responds)

    ic = beh["interview_completion_rate"]
    mult *= 0.85 + 0.15 * ic

    notice = beh["notice_period_days"]
    if notice <= 30:
        mult *= 1.03
    elif notice > 90:
        mult *= 0.9

    return mult


def compute_score(features: dict) -> dict:
    skill = features["skill"]
    career = features["career"]
    disq = features["disqualifiers"]
    loc = features["location"]
    exp = features["experience"]
    beh = features["behavioral"]
    honeypot = features["honeypot"]

    shipped_score = _shipped_system_score(career)
    title_score = 1.0 if career["title_relevant"] else 0.15
    exp_score = _experience_fit_score(exp)
    loc_score = _location_fit_score(loc)
    nice_score = min(1.0, skill["nice_to_have_count"] / 4.0)

    fit_score = (
        WEIGHTS["must_have_skills"] * skill["must_have_skill_score"]
        + WEIGHTS["shipped_system_evidence"] * shipped_score
        + WEIGHTS["title_relevance"] * title_score
        + WEIGHTS["experience_fit"] * exp_score
        + WEIGHTS["location_fit"] * loc_score
        + WEIGHTS["nice_to_have"] * nice_score
    )

    penalty_mult = 1.0
    triggered_disqualifiers = []
    for flag, penalty in DISQUALIFIER_PENALTIES.items():
        if disq[flag]:
            penalty_mult *= penalty
            triggered_disqualifiers.append(flag)

    fit_score *= penalty_mult

    beh_mult = _behavioral_multiplier(beh)
    final_score = fit_score * beh_mult

    # Honeypots: hard cap near zero regardless of anything else. The docs
    # are explicit that these are forced to relevance tier 0 in ground
    # truth, and >10% honeypot rate in the top 100 causes disqualification
    # at Stage 3 - so we treat this as a gate, not a soft penalty.
    if honeypot["is_honeypot"]:
        final_score *= 0.02

    return {
        "final_score": round(final_score, 6),
        "fit_score": round(fit_score, 6),
        "behavioral_multiplier": round(beh_mult, 4),
        "shipped_score": round(shipped_score, 4),
        "title_score": round(title_score, 4),
        "exp_score": round(exp_score, 4),
        "loc_score": round(loc_score, 4),
        "triggered_disqualifiers": triggered_disqualifiers,
        "is_honeypot": honeypot["is_honeypot"],
    }
