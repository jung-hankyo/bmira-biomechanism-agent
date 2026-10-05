"""Controlled vocabularies and the data passed between nodes."""
from typing import Literal, Optional, get_args

from pydantic import BaseModel, Field

RELATIONS = Literal[
    "increases", "decreases", "modulates", "no_effect",
    "required_for", "sufficient_for",
    "associated_with", "not_associated", "predicts",
    "binds", "modifies",
    "unresolved",                                   # sentinel: never biological evidence
]
RELATION_SET = set(get_args(RELATIONS))
CAUSAL_RELATIONS = {"increases", "decreases", "modulates", "required_for", "sufficient_for"}
ASSOCIATIVE_RELATIONS = {"associated_with", "not_associated", "predicts"}
NULL_RELATIONS = {"no_effect", "not_associated"}
PHYSICAL_RELATIONS = {"binds", "modifies"}
DIRECTION = {"increases": "up", "decreases": "down"}

CLAIM_TYPES = Literal["observation", "author_interpretation", "mechanistic_speculation"]
PERTURBATIONS = Literal["none", "genetic_association", "pharmacological", "environmental",
                        "knockdown", "knockout", "overexpression", "transfer"]
SYSTEMS = Literal["human_in_vivo", "human_primary_cells", "animal_in_vivo", "animal_cells",
                  "organoid", "cell_line", "in_silico", "unclear"]
SYSTEM_GROUP = {"human_in_vivo": "human", "human_primary_cells": "human", "animal_in_vivo": "animal",
                "animal_cells": "animal", "organoid": "cell", "cell_line": "cell", "in_silico": "in_silico"}
ATTRIBUTES = Literal["none", "expression", "amount", "activity", "modification", "differentiation"]
STUDY_TYPES = Literal["meta_analysis", "human_rct", "human_cohort", "human_crosssectional",
                      "human_primary", "organoid_ipsc", "animal", "cell_line",
                      "computational_cohort", "in_silico", "review"]
GRADES = Literal["ungraded", "weak", "moderate", "strong"]
TIER = {"ungraded": 0, "weak": 1, "moderate": 2, "strong": 3}
LABEL = {0: "ungraded", 1: "weak", 2: "moderate", 3: "strong"}


# ── question and search ─────────────────────────────────────────────────────
class ParsedQuestion(BaseModel):
    population_model: str
    exposure: str
    comparator: str
    outcome: str
    mechanism_hypothesis: str
    outcome_readouts: list[str] = Field(default_factory=list, description=(
        "2-5 measurable readouts of the outcome as bare entity names, e.g. IFNG, granzyme B"))
    target_system: Literal["human", "animal", "cell", "any"] = Field(
        "any", description="Population or system the question is about")
    exposure_change: Literal["up", "down"] = Field(
        "up", description="down if the question is about a decrease, loss or deficiency of the exposure")
    expected_direction: Literal["up", "down", "none", "unknown"] = Field(
        "unknown", description="Direction the question asserts for the exposure CHANGE -> outcome")
    exposure_members: list[str] = Field(default_factory=list, description=(
        "If the exposure is a class or group, up to 6 specific members as bare names; otherwise empty"))
    in_scope: bool = Field(True, description=(
        "False if this is not a question the biomedical literature can answer as exposure -> outcome"))
    scope_note: str = Field("", description="One sentence: why the question is out of scope")


class SearchQuery(BaseModel):
    query: str
    intent: Literal["broad", "mechanism", "contradiction", "negative_result",
                    "gap_positive", "gap_alternative_terms", "gap_null"]
    target: str = Field("", description="Number of the target this query is for (T1, T2, ...); empty in round 1")


class QueryPlan(BaseModel):
    queries: list[SearchQuery]


class Paper(BaseModel):
    pmid: str
    title: str = ""
    abstract: str = ""
    journal: str = ""
    year: str = ""
    source_text: str = ""
    text_access: Literal["full_text", "abstract_only"] = "abstract_only"
    publication_types: list[str] = Field(default_factory=list)
    pubtype_study_type: Optional[STUDY_TYPES] = None
    retracted: bool = False
    screen_status: Literal["unscreened", "included", "excluded"] = "unscreened"
    relevance_score: int = 0
    relevance_reason: str = ""
    study_type: Optional[STUDY_TYPES] = None
    retrieved_for: list[str] = Field(default_factory=list)   # steps whose searches returned it
    read_for: list[str] = Field(default_factory=list)        # steps it was extracted for
    n_reads: int = 0                                          # extraction calls on this paper (re-reads included)
    chars_read: int = 0                                       # characters sent to extraction, over all reads


class Screen(BaseModel):
    relevant: bool
    reason: str
    relevance_score: int = Field(50, ge=0, le=100)
    study_type: STUDY_TYPES


# ── claims ──────────────────────────────────────────────────────────────────
class ExtractedClaim(BaseModel):
    claim_type: CLAIM_TYPES
    subject: str = Field(description="Bare entity name, without words like expression or levels")
    subject_attribute: ATTRIBUTES = "none"
    relation: str = Field(description="Surface relation, verbatim wording from the span")
    object: str = Field(description="Bare entity name, without words like expression or levels")
    object_attribute: ATTRIBUTES = "none"
    system: SYSTEMS = Field("unclear", description="Experimental system of THIS claim")
    context_tissue: str = Field("", description="Tissue or site, e.g. colon, spleen, bone marrow")
    context_model: str = ""
    context_cell_type: str = ""
    context_dose: str = ""
    context_timepoint: str = ""
    perturbation_class: PERTURBATIONS = "none"
    subject_lost: bool = Field(False, description=(
        "True when the subject is knocked out or down, deleted, depleted, inhibited or absent in this "
        "experiment ('Gpr109a-/- mice', 'X-deficient cells', 'mice lacking X') and the relation states "
        "what that LOSS did"))
    rescue_arm: bool = False
    orthogonal_validation: bool = False
    comparator_present: bool = False
    readout_is_inferred: bool = False
    span: str = Field(description="Verbatim sentence from the text")
    methods_span: str = Field("", description=(
        "Verbatim sentence describing the perturbation, rescue, validation or comparator; "
        "empty if the text does not describe them"))


class ClaimList(BaseModel):
    claims: list[ExtractedClaim]


class Claim(ExtractedClaim):
    id: str
    pmid: str
    study_type: STUDY_TYPES
    text_access: Literal["full_text", "abstract_only"]
    anchored: bool = False
    subject_concept: str = ""
    object_concept: str = ""
    subject_label: str = ""
    object_label: str = ""
    subject_category: str = "unknown"
    object_category: str = "unknown"
    subject_parents: list[str] = Field(default_factory=list)
    object_parents: list[str] = Field(default_factory=list)
    context_concept: str = ""
    round: int = 0
    method_checks: list[str] = Field(default_factory=list)   # method fields reset for lack of evidence
    drop_reason: str = ""
    relation_raw: str = ""
    relation_norm: str = ""                        # "" = pending, never sent twice once set
    relation_source: str = ""
    relation_confidence: float = 0.0
    grade: GRADES = "ungraded"
    grade_detail: dict = Field(default_factory=dict)

    @property
    def context(self) -> str:
        return (self.context_cell_type or "unspecified").strip().lower()


class RelationResolution(BaseModel):
    claim_id: str
    canonical_relation: RELATIONS
    confidence: float = Field(ge=0.0, le=1.0)


class RelationResolutionBatch(BaseModel):
    resolutions: list[RelationResolution]


class EntityResolution(BaseModel):
    normalized_label: str
    category: str
    confidence: float = Field(ge=0.0, le=1.0)


class EntityItem(BaseModel):
    surface: str = Field(description="The surface form exactly as given")
    normalized_label: str
    category: str
    confidence: float = Field(ge=0.0, le=1.0)


class EntityBatch(BaseModel):
    items: list[EntityItem]


class AliasVerdict(BaseModel):
    label_a: str
    label_b: str
    same_entity: bool


class AliasBatch(BaseModel):
    verdicts: list[AliasVerdict]


# ── semantic layer and conflicts ────────────────────────────────────────────
class PairAdjudication(BaseModel):
    pair: int = Field(description="The number n of the pair, as in '[PAIR n]'")
    same_finding: bool = Field(description="Same exposure and materially the same measured endpoint")
    same_context: bool = Field(description="Same or compatible model and cell type")


class PairBatch(BaseModel):
    pairs: list[PairAdjudication]


class Cluster(BaseModel):
    key: str
    claim_ids: list[str]
    discordant: int = 0


class Conflict(BaseModel):
    cluster_key: str
    verdict: Literal["true_conflict", "context_dependent", "not_comparable"]
    explanation: str
    discriminating_experiment: str = ""
    claim_ids: list[str] = Field(default_factory=list, description="Leave empty; filled by code")


class ConflictBatch(BaseModel):
    conflicts: list[Conflict]


# ── pathway portfolio ───────────────────────────────────────────────────────
class ProposedLink(BaseModel):
    source: str
    relation: RELATIONS
    target: str


class ProposedPathway(BaseModel):
    name: str
    rationale: str = ""
    links: list[ProposedLink]


class PathwayProposal(BaseModel):
    pathways: list[ProposedPathway]


class LinkEvidence(BaseModel):
    key: str                                       # "subject_id|relation|object_id"
    subject: str
    relation: str
    object: str
    subject_label: str = ""
    object_label: str = ""
    support_ids: list[str] = Field(default_factory=list)
    corroborating_ids: list[str] = Field(default_factory=list)
    contradicting_ids: list[str] = Field(default_factory=list)
    uncounted: dict = Field(default_factory=dict)  # claim id -> why it does not count
    n_studies: int = 0                             # independent papers: support + corroboration
    n_contra_studies: int = 0
    grade: GRADES = "ungraded"                     # best supporting claim
    contra_grade: GRADES = "ungraded"
    contradiction_share: float = 0.0
    completeness: float = 0.0
    status: Literal["supported", "contradicted", "insufficient"] = "insufficient"
    reason: str = "not found yet"
    contexts: list[str] = Field(default_factory=list)
    times_targeted: int = 0
    zero_yield_count: int = 0
    exhausted: bool = False


class Hypothesis(BaseModel):
    id: str
    name: str
    origin: Literal["llm_seed", "llm_expansion", "ledger_path", "user"]
    links: list[str]                               # LinkEvidence keys, in order
    rationale: str = ""
    status: Literal["supported", "contradicted", "insufficient"] = "insufficient"
    reason: str = ""
    open: bool = True                              # still worth searching (internal)
    score: float = 0.0
    logic_factor: float = 1.0
    logic_flags: list[str] = Field(default_factory=list)


# ── verification ────────────────────────────────────────────────────────────
class SentenceEntailment(BaseModel):
    sentence_index: int
    verdict: Literal["entailed", "partial", "unsupported"]
    rationale: str = ""


class EntailmentBatch(BaseModel):
    judgements: list[SentenceEntailment]
