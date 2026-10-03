"""All tunable policy values in one place."""
from dataclasses import dataclass, field


@dataclass
class Settings:
    provider: str = "openai"                      # "openai" | "anthropic"
    models: dict = field(default_factory=lambda: {
        "openai": {"reasoning": "gpt-5.6-terra", "cheap": "gpt-5.6-luna"},
        "anthropic": {"reasoning": "claude-sonnet-5", "cheap": "claude-haiku-4-5-20251001"},
    })
    # retrieval
    max_papers_per_query: int = 20
    max_extract_per_round: int = 10               # top-N by relevance; the rest wait, they are not dropped
    max_claims_per_paper: int = 8
    fulltext_char_limit: int = 60000
    # evidence
    min_studies_per_link: int = 2
    contradiction_threshold: float = 0.5          # contradicting share of papers that blocks support
    # portfolio search
    max_rounds: int = 5
    n_seed_hypotheses: int = 4
    max_hypotheses: int = 6
    max_path_len: int = 4
    targets_per_round: int = 3
    exploration_slots: int = 1
    ucb_beta: float = 0.5
    softmax_temperature: float = 0.2
    zero_yield_rounds_to_exhaust: int = 2
    # semantic layer
    max_candidates_per_claim: int = 8
    similarity_threshold: dict = field(default_factory=lambda: {"sentence": 0.60, "hashing": 0.30})
    # ontology: "ols" | "llm" | "hybrid" | "off"
    ontology_provider: str = "hybrid"
    ontology_timeout_s: float = 8.0
    # NCBI
    ncbi_api_key: str = ""
    ncbi_email: str = ""
    # human review of the seeded portfolio (needs a checkpointer)
    interactive: bool = False
