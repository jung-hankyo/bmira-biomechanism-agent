"""All tunable policy values in one place."""
from dataclasses import dataclass, field


@dataclass
class Settings:
    provider: str = "openai"                      # "openai" | "anthropic"
    models: dict = field(default_factory=lambda: {
        "openai": {"reasoning": "gpt-6-sol", "cheap": "gpt-6-luna"},
        "anthropic": {"reasoning": "claude-sonnet-5-5", "cheap": "claude-sonnet-5-5"},
    })
    # retrieval
    max_papers_per_query: int = 20               # coverage round
    max_papers_per_target_query: int = 5         # targeted rounds: reading capacity is the limit
    min_relevance: int = 50                      # screening score (0-100) needed for inclusion
    max_extract_per_round: int = 20               # top-N by relevance; the rest wait, they are not dropped
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
    # Sampling temperature. None = the model's default, which every model accepts; some
    # reasoning models reject any other value. Set a number only if your model allows it.
    temperature: float | None = None
    # LLM calls: transient errors (rate limits, overload, timeouts) are retried with
    # backoff that honours Retry-After; fatal ones (empty balance, bad key, unknown model,
    # unsupported parameter) stop the run at once and the session is aborted.
    llm_max_retries: int = 6
    llm_timeout_s: float = 180.0
    # Reasoning effort per task for OpenAI reasoning models (ignored for Anthropic).
    # A task missing here, or None, uses the model's default effort. Classification tasks
    # run low; extraction, planning and the report run medium (an A/B on 16 papers: low read 18% fewer claims, 23% fewer nulls).
    reasoning_effort: dict = field(default_factory=lambda: {
        "screen": "low", "entity": "low", "entities": "low", "relation": "low", "alias": "low", "pair": "low",
        "conflict": "low", "entailment": "low", "chat": "low", "preflight": "low",
        "parse": "medium", "plan": "medium", "extract": "medium", "seed": "medium",
        "expand": "medium", "synthesize": "medium"})
    # Classification tasks run on the cheap model whatever the call site asks for.
    cheap_tasks: tuple = ("screen", "entity", "entities", "relation", "alias", "pair",
                          "conflict", "entailment", "preflight")
    # Directory for caches reused across runs (entity resolutions); None = no disk cache.
    cache_dir: str | None = None
    # Soft token budget per run (input + output over all tasks), checked between rounds:
    # when reached, searching stops and the report is written from what was found.
    # None = no cap. Also set a hard spend limit at the provider as the real backstop.
    budget_tokens: int | None = None
    # Soft budget in USD per run (TE-10): the estimated spend of every LLM task and the judge, from
    # `prices`, checked between rounds like budget_tokens. Models without a price are not counted and
    # are named in the run warnings. None = no cap.
    budget_usd: float | None = None
    # USD per 1M tokens (input, output), for cost estimates in telemetry only. Edit to your
    # provider's current prices; a model missing here shows no estimate. OpenAI figures:
    # official changelog; Anthropic figures: published list prices (both checked 2026-10-03).
    prices: dict = field(default_factory=lambda: {
        "gpt-6-sol": (2.0, 10.0), "gpt-6.1-sol": (2.0, 10.0), "gpt-6-luna": (0.10, 0.50),
        "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-4-5-20251001": (1.0, 5.0),
        "jev-1.13.0": (0.042, 0.0)})                 # judge: input only, output free (TypeSafe, 2026-10-05)
    # Decision model ("judge", TypeSafe Jev) for typed judgments; questions and thresholds in
    # bmira/questions.py. "off" keeps today's executors everywhere. Shadow mode asks and logs next
    # to today's decision (state judge_log, telemetry 'judge') and changes nothing; "act" is
    # refused until a touchpoint is calibrated on gold labels. Needs TYPESAFE_API_KEY.
    judge_provider: str = "off"                  # "jev" | "off"
    judge_model: str = "jev-1.13.0"              # pinned; never an alias such as jev-latest
    judge_mode: str = "shadow"
    judge_timeout_s: float = 30.0
    judge_workers: int = 16
    judge_max_retries: int = 6
