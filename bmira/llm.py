"""LLM access. Every call goes through one of three methods so calls can be counted
and an offline surrogate (bmira.offline.SurrogateLLM) can stand in for the real model.

`ctx` carries the raw objects behind a prompt. The real client ignores it; the
surrogate reads it instead of parsing prompt text.
"""
import os
from collections import Counter
from functools import lru_cache

PROMPTS = {
    "parse": (
        "You are a biomedical research methodologist. Decompose the question into P-E-C-O-M "
        "(population/model, exposure/perturbation, comparator, outcome, mechanism hypothesis). "
        "Set expected_direction to the "
        "direction the question asserts for exposure -> outcome (up, down, none, unknown)."),
    "plan": (
        "Write PubMed queries with MeSH terms, gene symbols and synonyms in valid syntax.\n"
        "If NO targets are given, return exactly 4 queries, one per intent: broad, mechanism, "
        "contradiction, negative_result.\n"
        "If targets are given, return exactly 3 queries PER target, intents gap_positive, "
        "gap_alternative_terms, gap_null, and copy the target id into `target`. Do not re-run "
        "general coverage in targeted rounds."),
    "screen": (
        "Decide whether this paper can contribute evidence to the question and classify its "
        "study type from the abstract. Be inclusive of mechanism, negative and contradictory "
        "findings. relevance_score 0-100: highest for papers testing a mechanism step or the "
        "endpoint directly, low for background."),
    "extract": (
        "Extract structured claims relevant to the question.\n"
        "1. claim_type separates what was MEASURED (observation), what authors CONCLUDED from "
        "their data (author_interpretation), and what goes BEYOND it (mechanistic_speculation).\n"
        "2. span is copied VERBATIM from one place in the text; stitched or paraphrased spans "
        "are discarded automatically.\n"
        "3. relation is the surface wording ('prevents', 'is required for'); do not normalize.\n"
        "4. perturbation_class / rescue_arm / orthogonal_validation / comparator_present come "
        "from what the METHODS did, not from the conclusions. They decide the grade.\n"
        "5. readout_is_inferred = true for computationally inferred readouts.\n"
        "6. At most {max_claims} claims, closest to the question first."),
    "relation": (
        "Map each surface relation to exactly one canonical relation: increases, decreases, "
        "modulates, no_effect, required_for, sufficient_for, associated_with, not_associated, "
        "predicts, binds, modifies, unresolved. Preserve direction and causal strength; never "
        "infer stronger causality than the wording. Use unresolved when no relation is faithful."),
    "entity": (
        "Normalize a biomedical entity. Do not invent database identifiers. Return a concise "
        "standard label and a broad category (gene_or_protein, chemical, cell_type, process, "
        "phenotype, disease, anatomy, other)."),
    "alias": (
        "Decide whether two surface forms denote the SAME entity: abbreviations, spelling "
        "variants, generic/trade names, and measurement synonyms of one quantity are the same. "
        "A parent class, a pathway, a family member or a correlated endpoint is NOT the same. "
        "When unsure, answer false."),
    "pair": (
        "For each pair of claims decide two things. same_finding: same exposure and materially "
        "the same measured endpoint, regardless of direction. same_context: same or compatible "
        "model and cell type. Ignore citation quality."),
    "conflict": (
        "Triage candidate evidence conflicts. true_conflict: comparable systems disagree. "
        "context_dependent: findings differ because cell type, model, dose or timepoint differ. "
        "not_comparable: the claims do not measure the same thing, or a method artifact explains "
        "the difference. Say which in the explanation, and give a discriminating experiment. "
        "Leave claim_ids empty."),
    "seed": (
        "Propose {k} DIFFERENT candidate mechanistic pathways from exposure to outcome, each an "
        "ordered list of links (source, relation, target) using the closed relation vocabulary. "
        "Pathways must differ in their intermediate steps: parallel, convergent and "
        "context-specific routes are all welcome. Use entity names exactly as in the claim list "
        "where possible. Include a link even if no claim supports it yet."),
    "expand": (
        "The evidence ledger contains intermediates that no current pathway uses. If any of "
        "them suggests a distinct, biologically coherent route from exposure to outcome, "
        "propose at most ONE new pathway using it. Return an empty list otherwise."),
    "synthesize": (
        "Write a calibrated research synthesis in English for a ranked portfolio of candidate "
        "pathways.\nHARD CONSTRAINTS (checked automatically):\n"
        "1. Every factual sentence cites claim IDs in square brackets, e.g. [C123_0].\n"
        "2. Verb strength never exceeds the WEAKEST cited grade: weak -> association language; "
        "moderate -> promotes/inhibits/increases/reduces; strong -> causes/drives/required for.\n"
        "3. Claims with associated_with / not_associated / predicts get association language.\n"
        "4. A sentence stating ABSENCE of evidence cites [NO_EVIDENCE] instead.\n"
        "5. Give each pathway a heading containing its tag, e.g. '### [H1] ...', and mention "
        "every link by its tag, e.g. [L3].\n"
        "6. Use only these status words: Supported, Contradicted, Insufficient evidence.\n"
        "7. Run status and warnings are appended to the report by code. Do not restate them, but "
        "never describe evidence as complete or concordant when a warning says otherwise.\n"
        "Structure: summary of the leading pathway(s) and how rivals compare; each pathway with its links; "
        "conflicts with discriminating experiments; gaps framed as missing experiments."),
    "chat": (
        "You answer follow-up questions about a finished B-MiRA run. Use ONLY the run context. "
        "Cite claim IDs in square brackets, e.g. [C123_0], for every statement about evidence. "
        "Use only these verdict words: Supported, Contradicted, Insufficient evidence. Verb "
        "strength never exceeds the weakest cited grade (weak -> association language). If the "
        "context does not answer the question, say so plainly and suggest starting a new "
        "investigation with '/new <question>'. Be concise."),
    "entailment": (
        "For each numbered sentence decide whether the CITED claims entail it: entailed, "
        "partial (supports part, or joins claims into a stronger proposition), unsupported."),
}


class LangChainLLM:
    def __init__(self, settings, api_key: str | None = None):
        self.settings = settings
        self.api_key = api_key or os.environ.get(f"{settings.provider.upper()}_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(f"{settings.provider} API key missing; no silent provider switch.")
        self.calls, self.items = Counter(), Counter()

    @lru_cache(maxsize=8)
    def _model(self, role: str, temperature: float):
        name = self.settings.models[self.settings.provider][role]
        if self.settings.provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=name, temperature=temperature, api_key=self.api_key)
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=name, temperature=temperature, api_key=self.api_key)

    def structured(self, task, schema, system, user, role="reasoning", ctx=None, n_items=1):
        self.calls[task] += 1
        self.items[task] += n_items
        model = self._model(role, 0.0).with_structured_output(schema)
        return model.invoke([("system", system), ("human", user)])

    def structured_many(self, task, schema, system, users, role="cheap", ctxs=None):
        self.calls[task] += 1
        self.items[task] += len(users)
        model = self._model(role, 0.0).with_structured_output(schema)
        return model.batch([[("system", system), ("human", u)] for u in users],
                           config={"max_concurrency": 8})

    def text(self, task, system, user, role="reasoning", ctx=None):
        self.calls[task] += 1
        self.items[task] += 1
        return self._model(role, 0.2).invoke([("system", system), ("human", user)]).content
