"""LLM access. Every call goes through one of three methods so calls can be counted
and an offline surrogate (bmira.offline.SurrogateLLM) can stand in for the real model.

`ctx` carries the raw objects behind a prompt. The real client ignores it; the
surrogate reads it instead of parsing prompt text.
"""
import os
import re
import time
from collections import Counter
from functools import lru_cache

PROMPTS = {
    "parse": (
        "You are a biomedical research methodologist. Write every field in English (translate the "
        "question if it is in another language). Set in_scope false, with a one-sentence scope_note, "
        "if the biomedical literature cannot answer the question as an exposure -> outcome question "
        "(chit-chat, a non-biomedical topic, a request for a personal diagnosis or treatment decision, "
        "no identifiable exposure and outcome); still fill the other fields with your best guess. "
        "Decompose the question into P-E-C-O-M "
        "(population/model, exposure/perturbation, comparator, outcome, mechanism hypothesis). "
        "Give exposure and outcome as bare entity or phenotype names, without verbs such as "
        "induction or suppression and without tissue words (e.g. 'regulatory T cell', not "
        "'induction of colonic regulatory T cells'; 'heart failure hospitalization', not 'reduction "
        "in hospitalizations for heart failure'). If the exposure is a CLASS or group ('SGLT2 "
        "inhibitors', 'statins', 'short-chain fatty acids'), list up to 6 specific members in "
        "exposure_members (empagliflozin, dapagliflozin, ...): papers name the members. Otherwise leave it "
        "empty. If the question is about a decline, loss or "
        "deficiency of the exposure ('NAD+ decline', 'TET2 loss'), name the bare entity ('NAD+', "
        "'TET2') and set exposure_change to down; otherwise up. Set expected_direction to the "
        "direction the question asserts for the exposure CHANGE -> outcome (up, down, none, "
        "unknown): 'NAD+ decline drives inflammaging' is exposure_change down, expected_direction up. "
        "List 2-5 outcome_readouts: measurable readouts of the "
        "outcome as bare entity names: a marker, a cell population, or for a clinical outcome a component "
        "endpoint or surrogate (e.g. FOXP3, granzyme B, NT-proBNP, cardiovascular death). Set target_system "
        "to the population the question is about (human, animal, cell, any)."),
    "plan": (
        "Write PubMed queries with MeSH terms, gene symbols and synonyms in valid syntax. Use at "
        "most 3 AND-groups per query (more returns nothing) and no multi-word phrases PubMed does not "
        "index: for null results use single words or short common phrases (negative, dispensable, "
        "independent, \"no effect\"), never 'failed to induce'.\n"
        "If NO targets are given, return exactly 4 queries, one per intent: broad, mechanism, "
        "contradiction, negative_result. For a clinical or epidemiological question make the broad query "
        "ask for human evidence (randomized controlled trial or meta-analysis publication types); for a "
        "laboratory question use model-system terms. If the exposure has listed members, use them as "
        "synonyms in the broad query.\n"
        "If targets are given, return exactly 3 queries PER target, intents gap_positive, "
        "gap_alternative_terms, gap_null, and set `target` to the target's number (T1, T2, ...). Do not re-run "
        "general coverage in targeted rounds."),
    "screen": (
        "Decide whether this paper can contribute evidence to the question (or to the listed "
        "mechanism steps, if any) and classify its study type from the abstract. Include "
        "mechanism, negative and contradictory findings. relevance_score 0-100: 70+ only for "
        "papers that report an experiment or analysis on the question or a listed step; 40-69 "
        "for indirect evidence; below 40 for background, reviews of unrelated topics, or other "
        "systems. Papers below 50 are not read."),
    "extract": (
        "Extract structured claims relevant to the question.\n"
        "1. claim_type separates what was MEASURED (observation), what authors CONCLUDED from "
        "their data (author_interpretation), and what goes BEYOND it (mechanistic_speculation).\n"
        "2. span is ONE sentence copied VERBATIM from the text, and it must name both entities "
        "and state the relation. Claims whose quote does not are discarded automatically.\n"
        "3. subject and object are ONE entity each (gene, protein, metabolite, cell type, process, "
        "phenotype), written with the span's own wording. Measurement or process words go in "
        "*_attribute (expression, amount, activity, modification, differentiation); tissue or "
        "site words (colonic, splenic, bone marrow) go in context_tissue. For 'A and B' write one "
        "claim per entity. Reagents (antibodies, inhibitors, siRNA) are perturbations: name the "
        "entity they target. Never write 'not specified'; skip the claim instead. A cell type or "
        "genotype that only sets the context goes in context_cell_type, and the treatment stays "
        "the subject: 'DCs exposed to butyrate express IDO1' is butyrate increases IDO1 (cell "
        "type: dendritic cell). When a knockout abolishes an effect ('Slc5a8-null DCs do not "
        "induce IDO1 in response to butyrate'), write 'Slc5a8 required_for IDO1 induction' with "
        "perturbation_class knockout; 'mediated by X' is the same: X required_for the effect. When the "
        "subject itself is lost in the experiment ('Gpr109a-/- mice show fewer CD103+ DCs', 'X-deficient "
        "cells', 'mice lacking X'), keep it as the subject, copy the wording of what the LOSS did into "
        "relation, and set subject_lost true; code derives the subject's normal role. In a trial or "
        "cohort the treatment group is the subject and the endpoint the object ('dapagliflozin reduced "
        "heart failure hospitalization').\n"
        "4. relation is the surface wording from the span ('prevents', 'did not change').\n"
        "5. system is the experimental system of THIS claim (one paper can contain several).\n"
        "6. perturbation_class / rescue_arm / orthogonal_validation / comparator_present come "
        "from what the METHODS did. Copy the sentence that shows them into methods_span; fields "
        "without textual support are not credited.\n"
        "7. readout_is_inferred = true for computationally inferred readouts.\n"
        "8. At most {max_claims} claims. Priority: findings on any focus steps; then null or "
        "opposite findings (always include them when reported); then other findings on the "
        "question."),
    "relation": (
        "Map each surface relation to exactly one canonical relation: increases, decreases, "
        "modulates, no_effect, required_for, sufficient_for, associated_with, not_associated, "
        "predicts, binds, modifies, unresolved. Preserve direction and causal strength; never "
        "infer stronger causality than the wording. Use unresolved when no relation is faithful. "
        "Translate the span's wording as written: what the span says happened to the object is the "
        "direction ('X-deficient mice show fewer Y' is decreases). Never invert it for knockouts, "
        "deficiency or inhibition; code does that."),
    "entity": (
        "Normalize a biomedical entity. Do not invent database identifiers. Return a concise "
        "standard label and a broad category (gene_or_protein, chemical, cell_type, process, "
        "phenotype, disease, anatomy, other)."),
    "entities": (
        "Normalize each biomedical entity. Return the surface exactly as given, a concise "
        "standard singular label without tissue or measurement words, and a broad category "
        "(gene_or_protein, chemical, cell_type, process, phenotype, disease, anatomy, other). "
        "Do not invent database identifiers."),
    "alias": (
        "Decide whether two surface forms denote the SAME entity: abbreviations, spelling "
        "variants, generic/trade names, and measurement synonyms of one quantity are the same. "
        "A parent class, a pathway, a family member or a correlated endpoint is NOT the same. "
        "When unsure, answer false."),
    "pair": (
        "For each pair of claims decide two things. same_finding: same exposure and materially "
        "the same measured endpoint, regardless of direction. same_context: same or compatible "
        "model and cell type. Ignore citation quality. Return exactly one verdict per pair, in order, "
        "with `pair` set to the number n of '[PAIR n]'."),
    "conflict": (
        "Triage candidate evidence conflicts. true_conflict: comparable systems disagree. "
        "context_dependent: findings differ because cell type, model, dose or timepoint differ. "
        "not_comparable: the claims do not measure the same thing, or a method artifact explains "
        "the difference. Say which in the explanation, and give a discriminating experiment. "
        "Return one verdict per candidate, with `cluster_key` set to the number n of "
        "'[CANDIDATE n]' (just the number). Leave claim_ids empty."),
    "seed": (
        "Propose {k} DIFFERENT candidate mechanistic pathways from exposure to outcome, each an "
        "ordered list of links (source, relation, target) using the closed relation vocabulary. "
        "Pathways must differ in their intermediate steps: parallel, convergent and "
        "context-specific routes are all welcome. Use entity names exactly as in the claim list "
        "where possible. Include a link even if no claim supports it yet. Every link must be "
        "one that a primary study could test with the exposure present (for example 'butyrate "
        "increases histone H3 acetylation'), not textbook background; stop the pathway at the "
        "first readout of the outcome."),
    "expand": (
        "The evidence ledger contains intermediates that no current pathway uses. If any of "
        "them suggests a distinct, biologically coherent route from exposure to outcome, "
        "propose at most ONE new pathway using it. A pathway is an ordered chain: each link starts where "
        "the previous one ends, and the last link ends at the outcome or one of its readouts. Put side "
        "branches in the rationale, not in the links. Return an empty list otherwise."),
    "synthesize": (
        "Write a calibrated research synthesis in English for a ranked portfolio of candidate "
        "pathways.\nHARD CONSTRAINTS (checked automatically):\n"
        "1. Every factual sentence cites claim IDs in square brackets, e.g. [C123_0].\n"
        "2. Verb strength never exceeds the WEAKEST cited grade: weak -> association language; "
        "moderate -> promotes/inhibits/increases/reduces; strong -> causes/drives/required for. Never "
        "join findings of different grades in one sentence: give each weak finding its own sentence.\n"
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


class FatalLLMError(BaseException):
    """Unrecoverable provider error: empty balance, bad key, unknown model, unsupported parameter.

    Derives from BaseException on purpose. Pipeline nodes catch Exception to degrade
    gracefully (a failed batch is retried next round); an empty balance must not be
    degraded around, it must stop the run at once.
    """


FATAL = re.compile(r"insufficient_quota|credit_balance|credit balance|billing|invalid_api_key|"
                   r"incorrect api key|authentication|permission_error|model_not_found|"
                   r"not_found_error|does not exist|unsupported_value|unsupported_parameter|"
                   r"unrecognized request argument", re.I)


def is_fatal(e: BaseException) -> bool:
    return getattr(e, "status_code", None) in (401, 403, 404) or bool(FATAL.search(str(e)))


class LangChainLLM:
    """Counts calls, items, tokens (incl. reasoning), seconds and failures per task, and
    the model each task used (read by bmira.telemetry). Transient errors are retried by
    the client with backoff (honouring Retry-After); fatal errors raise FatalLLMError."""

    def __init__(self, settings, api_key: str | None = None):
        self.settings = settings
        self.api_key = api_key or os.environ.get(f"{settings.provider.upper()}_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(f"{settings.provider} API key missing; no silent provider switch.")
        self.calls, self.items, self.failures = Counter(), Counter(), Counter()
        self.tokens_in, self.tokens_out, self.seconds = Counter(), Counter(), Counter()
        self.tokens_reasoning, self.model_of = Counter(), {}

    @lru_cache(maxsize=16)
    def _model(self, role: str, effort: str | None = None):
        s = self.settings
        kw = {"model": s.models[s.provider][role], "api_key": self.api_key,
              "max_retries": s.llm_max_retries, "timeout": s.llm_timeout_s}
        if s.temperature is not None:
            kw["temperature"] = s.temperature
        if s.provider == "openai":
            from langchain_openai import ChatOpenAI
            if effort:
                kw["reasoning_effort"] = effort
            return ChatOpenAI(**kw)
        from langchain_anthropic import ChatAnthropic   # effort is an OpenAI setting
        return ChatAnthropic(**kw)

    def _for(self, task, role):
        role = "cheap" if task in self.settings.cheap_tasks else role
        self.model_of[task] = self.settings.models[self.settings.provider][role]
        return self._model(role, self.settings.reasoning_effort.get(task))

    def _run(self, task, n_items, fn):
        self.calls[task] += 1
        self.items[task] += n_items
        t0 = time.perf_counter()
        try:
            return fn()
        except Exception as e:
            self.failures[task] += 1
            if is_fatal(e):
                raise FatalLLMError(f"{task}: {type(e).__name__}: {e}") from e
            raise
        finally:
            self.seconds[task] += time.perf_counter() - t0

    def _usage(self, task, msg):
        u = getattr(msg, "usage_metadata", None) or {}
        self.tokens_in[task] += u.get("input_tokens", 0) or 0
        self.tokens_out[task] += u.get("output_tokens", 0) or 0
        self.tokens_reasoning[task] += (u.get("output_token_details") or {}).get("reasoning", 0) or 0

    def _parsed(self, task, res):
        self._usage(task, res.get("raw"))
        if res.get("parsed") is None:
            self.failures[task] += 1
            raise res.get("parsing_error") or ValueError(f"{task}: unparseable model output")
        return res["parsed"]

    def structured(self, task, schema, system, user, role="reasoning", ctx=None, n_items=1):
        model = self._for(task, role).with_structured_output(schema, include_raw=True)
        res = self._run(task, n_items, lambda: model.invoke([("system", system), ("human", user)]))
        return self._parsed(task, res)

    def structured_many(self, task, schema, system, users, role="cheap", ctxs=None):
        model = self._for(task, role).with_structured_output(schema, include_raw=True)
        res = self._run(task, len(users), lambda: model.batch(
            [[("system", system), ("human", u)] for u in users], config={"max_concurrency": 8},
            return_exceptions=True))
        out = []
        for r in res:            # one bad item costs that item (None), not the whole paid batch
            if isinstance(r, Exception):
                self.failures[task] += 1
                if is_fatal(r):
                    raise FatalLLMError(f"{task}: {type(r).__name__}: {r}") from r
                out.append(None)
                continue
            try:
                out.append(self._parsed(task, r))
            except Exception:
                out.append(None)
        return out

    def text(self, task, system, user, role="reasoning", ctx=None):
        model = self._for(task, role)
        msg = self._run(task, 1, lambda: model.invoke([("system", system), ("human", user)]))
        self._usage(task, msg)
        return msg.content if isinstance(msg.content, str) else "".join(
            b.get("text", "") for b in msg.content if isinstance(b, dict))
