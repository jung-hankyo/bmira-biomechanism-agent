"""Follow-up questions about a finished run, answered only from that run's evidence.

The answer is checked with the same verifier as the report, so a chat reply cannot
quietly use stronger language than its cited claims allow.
"""
from bmira import portfolio as pf
from bmira.evidence import verify_text
from bmira.llm import PROMPTS


def portfolio_rows(state) -> list[dict]:
    links = state["links"]
    return [{"id": h.id, "pathway": " → ".join([links[h.links[0]].subject_label]
                                               + [links[k].object_label for k in h.links]),
             "verdict": pf.STATUS_LABEL[h.status], "why": h.reason, "score": h.score,
             "source": pf.ORIGIN_LABEL[h.origin]} for h in state["hypotheses"]]


def run_context(state, max_claims=150) -> str:
    links, hyps = state["links"], state["hypotheses"]
    used = {i for h in hyps for k in h.links for i in links[k].support_ids
            + links[k].contradicting_ids + links[k].context_dependent_ids}
    claims = [c for c in state["claims"] if c.id in used][:max_claims]
    lines = [f"Question: {state['question']}",
             f"Search stopped: {pf.STOP_LABEL[state['gate']]} after {state['round_idx']} rounds.",
             "Warnings: " + ("; ".join(state.get("warnings", [])) or "none"), "", "Pathways:"]
    for h in hyps:
        lines.append(f"[{h.id}] {h.name}: {pf.STATUS_LABEL[h.status]} ({h.reason}), score {h.score}")
        for k in h.links:
            ln = links[k]
            lines.append(f"  step {ln.subject_label} -{ln.relation}-> {ln.object_label}: "
                         f"{pf.STATUS_LABEL[ln.status]} ({ln.reason}); support {ln.support_ids}; "
                         f"opposing {ln.contradicting_ids}; context-dependent {ln.context_dependent_ids}")
    lines += ["", "Claims:"] + [
        f"[{c.id}] ({c.grade}) {c.subject_label} {c.relation_norm} {c.object_label} | "
        f"{c.study_type} | {c.context_cell_type} | PMID {c.pmid}" for c in claims]
    return "\n".join(lines)


def answer(state, rt, question: str, history: list[dict]) -> tuple[str, list[str]]:
    """Return (reply, issues). `history` is the chat so far as {'role','content'} dicts."""
    recent = "\n".join(f"{m['role']}: {m['content'][:1500]}" for m in history[-6:])
    user = f"RUN CONTEXT\n{run_context(state)}\n\nCONVERSATION\n{recent}\n\nQUESTION\n{question}"
    reply = rt.llm.text("chat", PROMPTS["chat"], user, role="reasoning",
                        ctx={"question": question, "state": state})
    check = verify_text(reply, state["claims"], [])
    issues = [f"stronger wording than the evidence allows: {o['sentence'][:100]}"
              for o in check["overclaims"]] + [f"unknown claim id {i}" for i in check["unknown_ids"]]
    return reply, issues
