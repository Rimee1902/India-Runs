"""
jd_config.py

Structured representation of the Redrob AI Engineer JD, hand-encoded from
job_description.md. We do NOT call an LLM to parse the JD at ranking time
(that would violate the no-network/no-GPU ranking constraint) - the JD is
fixed and known in advance, so we encode it once as data.

All keyword lists below are deliberately traceable back to specific
sentences in the JD (see comments) so the scoring logic can be defended
in the Stage 5 interview.
"""

# ---------------------------------------------------------------------------
# "Things you absolutely need" (JD skills inventory - must-haves)
# ---------------------------------------------------------------------------
MUST_HAVE_SKILL_GROUPS = {
    "embeddings_retrieval": [
        "sentence-transformers", "sentence transformers", "openai embeddings",
        "bge", "e5", "embeddings", "dense retrieval", "semantic search",
        "text embeddings",
    ],
    "vector_db_hybrid_search": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
        "elasticsearch", "faiss", "vector database", "hybrid search",
        "vector search",
    ],
    "python": ["python"],
    "eval_frameworks": [
        "ndcg", "mrr", "map", "a/b test", "ab testing", "offline evaluation",
        "online evaluation", "evaluation framework", "ranking evaluation",
        "learning to rank", "ltr",
    ],
}

# "Things we'd like you to have but won't reject you for"
NICE_TO_HAVE_SKILLS = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuning",
    "xgboost", "learning-to-rank", "learning to rank",
    "recruiting tech", "hr-tech", "hr tech", "marketplace",
    "distributed systems", "large-scale inference", "inference optimization",
    "open source", "open-source",
]

# Career-history / description keywords indicating genuine "shipped a
# ranking/retrieval/recommendation system" evidence (used against titles
# and role descriptions, not just the skills list, since skills lists in
# this dataset are noisy/randomized decoys).
SHIPPED_SYSTEM_KEYWORDS = [
    "ranking system", "retrieval system", "recommendation system",
    "recommender system", "search system", "search relevance",
    "relevance ranking", "candidate ranking", "matching system",
    "personalization", "re-ranking", "reranking", "vector search",
    "semantic search", "hybrid retrieval", "query understanding",
    "embedding pipeline", "index refresh", "embedding drift",
]

RELEVANT_TITLE_KEYWORDS = [
    "ml engineer", "machine learning engineer", "ai engineer",
    "applied scientist", "research engineer", "ai research engineer",
    "search engineer", "ranking engineer", "recommender", "nlp engineer",
    "data scientist", "mle",
]

# ---------------------------------------------------------------------------
# Explicit disqualifiers ("Things we explicitly do NOT want" + hard
# disqualifiers under "what we mean by 5-9 years")
# ---------------------------------------------------------------------------

# Pure research / academic-only, no production deployment
RESEARCH_ONLY_KEYWORDS = [
    "research lab", "academic", "phd research", "postdoc", "postdoctoral",
    "research scientist", "university research",
]
PRODUCTION_EVIDENCE_KEYWORDS = [
    "production", "shipped", "deployed", "real users", "scale", "live system",
]

# "AI experience consists primarily of recent (<12mo) LangChain->OpenAI
# calls" without deeper pre-LLM ML production experience
FRAMEWORK_ONLY_KEYWORDS = ["langchain", "openai api", "prompt engineering"]
PRE_LLM_ML_KEYWORDS = [
    "recommendation", "search", "ranking", "retrieval", "nlp", "information retrieval",
    "machine learning", "deep learning", "classification", "regression",
]

# Consulting-only career (explicit named firms from the JD)
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
]

# CV/speech/robotics primary expertise without NLP/IR exposure
CV_SPEECH_ROBOTICS_KEYWORDS = [
    "computer vision", "image classification", "object detection",
    "speech recognition", "robotics", "autonomous", "signal processing",
]
NLP_IR_KEYWORDS = [
    "nlp", "natural language", "information retrieval", "search", "retrieval",
    "ranking", "text", "language model",
]

# "Architecture / tech lead" roles => hasn't written code recently
NON_CODING_TITLE_KEYWORDS = [
    "architect", "engineering manager", "tech lead", "head of engineering",
    "director of engineering", "vp of engineering", "vp engineering",
]

# ---------------------------------------------------------------------------
# Location (JD: "Pune/Noida-preferred but flexible... Candidates in
# Hyderabad, Pune, Mumbai, Delhi NCR welcome... Outside India: case-by-case,
# but we don't sponsor work visas.")
# ---------------------------------------------------------------------------
PREFERRED_CITIES = {"pune", "noida"}
TIER1_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "new delhi",
    "gurgaon", "gurugram", "bangalore", "bengaluru",
}

# ---------------------------------------------------------------------------
# Experience band: "5-9 years... roughly where people we've hired into this
# kind of role have landed" - soft band, not a hard cutoff.
# Ideal: "6-8 years total, of which 4-5 in applied ML/AI at product companies"
# ---------------------------------------------------------------------------
IDEAL_MIN_YOE = 5
IDEAL_MAX_YOE = 9
SWEET_SPOT_MIN_YOE = 6
SWEET_SPOT_MAX_YOE = 8

# Company size threshold used as a rough "product company" heuristic when
# combined with the consulting-firm exclusion list above.
IT_SERVICES_INDUSTRY_HINTS = ["it services", "consulting", "outsourcing"]
