"""Which claims report the same finding, and conflict triage (bmira.semantic). Offline: no keys, no network."""
from bmira.offline import HashingEmbedder
from bmira.semantic import candidate_pairs

from helpers import make_claim


# P5: negation-aware verbs; tags required.
def test_top_k_cap():
    claims = [make_claim(f"c{i}", f"p{i}", "increases") for i in range(20)]
    for c in claims:
        c.subject_label, c.object_label = "lactate", "nad"
    pairs = candidate_pairs(claims, HashingEmbedder(), 0.0, k=3)
    count = {}
    for a, b, _ in pairs:
        count[a] = count.get(a, 0) + 1
        count[b] = count.get(b, 0) + 1
    assert max(count.values()) <= 3


# K1-K10: fixes from the pilot3 live run (butyrate -> Treg).
def test_pair_verdicts_are_matched_by_position_not_echoed_ids():
    from bmira.schemas import PairAdjudication, PairBatch
    from bmira.semantic import adjudicate

    class LLM:
        def structured(self, task, schema, system, user, role="reasoning", ctx=None, n_items=1):
            assert "[PAIR 1]" in user and "[PAIR 2]" in user
            return PairBatch(pairs=[PairAdjudication(pair=i + 1, same_finding=True, same_context=True)
                                    for i in range(n_items)])
    claims = [make_claim(i, "p", "increases") for i in ("C1_0", "C2_0", "C3_0")]
    cache = {}
    failed = adjudicate(claims, [("C1_0", "C2_0", .9), ("C1_0", "C3_0", .8)], LLM(), cache)
    assert failed == 0 and set(cache) == {frozenset(("C1_0", "C2_0")), frozenset(("C1_0", "C3_0"))}


# L1-L5: problems the other seven questions would hit (found by probing the code, not yet seen live).
def test_pair_verdict_has_no_unused_rationale():
    from bmira.schemas import PairAdjudication
    assert "rationale" not in PairAdjudication.model_fields                  # ~100 output tokens per pair


# P1-P6: fixes from the pilot5 live run (butyrate -> Treg).
def test_conflict_verdicts_are_matched_by_number_not_echoed_keys():
    """Pilot5 asked 11 candidates and kept 1: the model had to echo 'SEM::C22724664_0'."""
    from bmira.schemas import Cluster, Conflict, ConflictBatch
    from bmira.semantic import triage
    claims = [make_claim(i, "p", "increases") for i in "ab"]
    cands = [Cluster(key=f"SEM::{k}", claim_ids=["a", "b"], discordant=1) for k in ("C22724664_0", "C1_0", "C2_1")]

    class LLM:
        def __init__(self, keys):
            self.keys, self.prompt = keys, ""

        def structured(self, task, schema, system, user, **kw):
            self.prompt = user
            return ConflictBatch(conflicts=[Conflict(cluster_key=k, verdict="context_dependent", explanation="e")
                                            for k in self.keys])
    llm = LLM(["1", "CANDIDATE 2", "[3]", "7", "nonsense"])
    out, status = triage(claims, cands, llm)
    assert status == "COMPLETE" and [c.cluster_key for c in out] == ["SEM::C22724664_0", "SEM::C1_0", "SEM::C2_1"]
    assert "[CANDIDATE 1]" in llm.prompt and "SEM::" not in llm.prompt          # nothing to echo
    assert [c.cluster_key for c in triage(claims, cands, LLM(["SEM::C1_0"]))[0]] == ["SEM::C1_0"]   # a scripted model

