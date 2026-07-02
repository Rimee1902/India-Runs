"""
features.py

Pure functions that take a raw candidate dict (as loaded from a line of
candidates.jsonl) and derive interpretable features. No network, no GPU,
no external model calls - everything here is string/keyword matching and
arithmetic over fields already in the record, so it's trivially fast
(microseconds per candidate) and fully reproducible offline.
"""

from datetime import date, datetime
import jd_config as cfg


def _text_blob(candidate: dict) -> str:
    """Concatenate all free-text fields we want to keyword-search over."""
    p = candidate["profile"]
    parts = [p.get("headline", ""), p.get("summary", "")]
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    return " ".join(parts).lower()


def _skill_names_lower(candidate: dict) -> set:
    return {s["name"].strip().lower() for s in candidate.get("skills", [])}


def skill_match_features(candidate: dict) -> dict:
    """
    Trust-weighted match against the JD's must-have skill groups.
    A skill only counts if it's genuinely evidenced (duration_months > 0
    OR proficiency is advanced/expert with endorsements > 0) - this is
    the mechanism that discounts random keyword-stuffed skill lists,
    which this dataset uses heavily as decoys.
    """
    skills = candidate.get("skills", [])
    skill_lookup = {}
    for s in skills:
        name = s["name"].strip().lower()
        trust = 0.0
        if s.get("duration_months", 0) >= 12:
            trust = 1.0
        elif s.get("duration_months", 0) > 0:
            trust = 0.6
        elif s.get("proficiency") in ("advanced", "expert") and s.get("endorsements", 0) > 0:
            trust = 0.3  # some signal, but no tenure - weak
        else:
            trust = 0.05  # essentially unevidenced - likely decoy
        skill_lookup[name] = max(skill_lookup.get(name, 0.0), trust)

    group_scores = {}
    for group, keywords in cfg.MUST_HAVE_SKILL_GROUPS.items():
        best = 0.0
        for kw in keywords:
            for name, trust in skill_lookup.items():
                if kw in name or name in kw:
                    best = max(best, trust)
        group_scores[group] = best

    nice_to_have_hits = 0
    for kw in cfg.NICE_TO_HAVE_SKILLS:
        for name, trust in skill_lookup.items():
            if trust >= 0.6 and (kw in name or name in kw):
                nice_to_have_hits += 1
                break

    must_have_score = sum(group_scores.values()) / len(cfg.MUST_HAVE_SKILL_GROUPS)

    return {
        "must_have_group_scores": group_scores,
        "must_have_skill_score": must_have_score,
        "nice_to_have_count": nice_to_have_hits,
    }


def career_relevance_features(candidate: dict) -> dict:
    """
    Looks past the (noisy) skills list into title + role descriptions for
    genuine evidence of having shipped ranking/retrieval/recommendation
    systems - the JD explicitly says this is what separates a real fit
    from a keyword-stuffed decoy.
    """
    blob = _text_blob(candidate)
    title = candidate["profile"]["current_title"].lower()

    shipped_hits = sum(1 for kw in cfg.SHIPPED_SYSTEM_KEYWORDS if kw in blob)
    title_relevant = any(kw in title for kw in cfg.RELEVANT_TITLE_KEYWORDS)

    return {
        "shipped_system_hits": shipped_hits,
        "title_relevant": title_relevant,
    }


def disqualifier_features(candidate: dict) -> dict:
    """
    Flags corresponding to the JD's explicit "do NOT want" / hard
    disqualifier list. Each is a boolean/soft signal; scoring.py decides
    how heavily to penalize.
    """
    blob = _text_blob(candidate)
    title = candidate["profile"]["current_title"].lower()
    industry = candidate["profile"].get("current_industry", "").lower()
    yoe = candidate["profile"]["years_of_experience"]

    # Research-only, no production deployment
    research_only = (
        any(kw in blob for kw in cfg.RESEARCH_ONLY_KEYWORDS)
        and not any(kw in blob for kw in cfg.PRODUCTION_EVIDENCE_KEYWORDS)
    )

    # Framework-only recent AI experience (LangChain/OpenAI wrapper work)
    # without deeper pre-LLM ML/production evidence
    framework_only = (
        any(kw in blob for kw in cfg.FRAMEWORK_ONLY_KEYWORDS)
        and not any(kw in blob for kw in cfg.PRE_LLM_ML_KEYWORDS)
    )

    # Consulting-only career: EVERY role was at one of the named firms
    companies = [r.get("company", "").lower() for r in candidate.get("career_history", [])]
    consulting_only = len(companies) > 0 and all(
        any(firm in c for firm in cfg.CONSULTING_FIRMS) for c in companies
    )

    # CV/speech/robotics-primary without NLP/IR exposure
    cv_speech_signal = any(kw in blob for kw in cfg.CV_SPEECH_ROBOTICS_KEYWORDS)
    nlp_ir_signal = any(kw in blob for kw in cfg.NLP_IR_KEYWORDS)
    cv_speech_only = cv_speech_signal and not nlp_ir_signal

    # Architecture/tech-lead title, long tenure => hasn't written code recently
    non_coding_title = any(kw in title for kw in cfg.NON_CODING_TITLE_KEYWORDS)

    # Title-chaser: 3+ jobs, each < 18 months, with escalating seniority words
    short_stints = [
        r for r in candidate.get("career_history", [])
        if r.get("duration_months", 999) < 18
    ]
    title_chaser = len(candidate.get("career_history", [])) >= 3 and len(short_stints) >= 3

    # Closed-source only 5+ years: no GitHub signal, no certifications, no OSS mention
    no_external_validation = (
        yoe >= 5
        and candidate.get("redrob_signals", {}).get("github_activity_score", -1) < 0
        and len(candidate.get("certifications", []) or []) == 0
        and "open source" not in blob and "open-source" not in blob
    )

    return {
        "research_only": research_only,
        "framework_only": framework_only,
        "consulting_only": consulting_only,
        "cv_speech_only": cv_speech_only,
        "non_coding_title": non_coding_title,
        "title_chaser": title_chaser,
        "no_external_validation": no_external_validation,
    }


def location_features(candidate: dict) -> dict:
    location = candidate["profile"].get("location", "").lower()
    country = candidate["profile"].get("country", "").lower()
    is_india = country == "india"
    preferred_city = any(city in location for city in cfg.PREFERRED_CITIES)
    tier1_city = any(city in location for city in cfg.TIER1_CITIES)
    willing_to_relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)

    return {
        "is_india": is_india,
        "preferred_city": preferred_city,
        "tier1_city": tier1_city,
        "willing_to_relocate": willing_to_relocate,
    }


def experience_features(candidate: dict) -> dict:
    yoe = candidate["profile"]["years_of_experience"]
    in_sweet_spot = cfg.SWEET_SPOT_MIN_YOE <= yoe <= cfg.SWEET_SPOT_MAX_YOE
    in_band = cfg.IDEAL_MIN_YOE <= yoe <= cfg.IDEAL_MAX_YOE
    return {"years_of_experience": yoe, "in_sweet_spot": in_sweet_spot, "in_band": in_band}


def _days_since(date_str: str, reference: date) -> float:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (reference - d).days
    except (ValueError, TypeError):
        return 9999


def behavioral_features(candidate: dict, reference_date: date) -> dict:
    """
    JD: "a perfect-on-paper candidate who hasn't logged in for 6 months and
    has a 5% recruiter response rate is, for hiring purposes, not actually
    available. Down-weight them appropriately."
    """
    sig = candidate.get("redrob_signals", {})
    days_inactive = _days_since(sig.get("last_active_date", ""), reference_date)

    return {
        "open_to_work": sig.get("open_to_work_flag", False),
        "days_inactive": days_inactive,
        "recruiter_response_rate": sig.get("recruiter_response_rate", 0.0),
        "interview_completion_rate": sig.get("interview_completion_rate", 0.0),
        "notice_period_days": sig.get("notice_period_days", 90),
        "profile_completeness": sig.get("profile_completeness_score", 0.0),
    }


def honeypot_features(candidate: dict) -> dict:
    """
    Detects the two honeypot patterns explicitly named in the docs:
      1. "expert" proficiency in many skills with 0 months used.
      2. years_of_experience inconsistent with total career_history duration.
    """
    skills = candidate.get("skills", [])
    expert_zero_count = sum(
        1 for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )

    yoe = candidate["profile"]["years_of_experience"]
    total_months = sum(r.get("duration_months", 0) for r in candidate.get("career_history", []))
    ratio = total_months / (yoe * 12 + 1e-6) if yoe > 0 else 1.0
    duration_mismatch = ratio < 0.5 or ratio > 1.8

    is_honeypot = expert_zero_count >= 4 or duration_mismatch

    return {
        "expert_zero_count": expert_zero_count,
        "duration_mismatch": duration_mismatch,
        "is_honeypot": is_honeypot,
    }


def extract_all_features(candidate: dict, reference_date: date) -> dict:
    return {
        "skill": skill_match_features(candidate),
        "career": career_relevance_features(candidate),
        "disqualifiers": disqualifier_features(candidate),
        "location": location_features(candidate),
        "experience": experience_features(candidate),
        "behavioral": behavioral_features(candidate, reference_date),
        "honeypot": honeypot_features(candidate),
    }
