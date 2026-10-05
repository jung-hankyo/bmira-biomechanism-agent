"""Shared test helpers: a claim factory, stand-ins for the model and for OLS, and an offline
session runner."""
import json

from bmira.schemas import Claim, ParsedQuestion


def make_claim(i, pmid, rel, grade="moderate", subj="LOCAL:a", obj="LOCAL:b", ctx="cd8",
               system="animal_cells", study_type="animal", **kw):
    return Claim(id=i, pmid=pmid, claim_type="observation", subject="a", relation=rel, object="b",
                 span="x" * 30, study_type=study_type, text_access="abstract_only", system=system,
                 subject_concept=subj, object_concept=obj, relation_norm=rel, grade=grade,
                 context_cell_type=ctx, comparator_present=True, **kw)


def fail_on_call(monkeypatch, task, exc, on_call=1):
    """Offline runtime whose surrogate raises `exc` on the n-th call of `task`."""
    import bmira.offline as off
    real, seen = off.offline_runtime, {"n": 0}          # shared: the failure happens once per test

    def patched(**kw):
        rt, sc = real(**kw)
        original = rt.llm.structured

        def structured(t, *args, **kwargs):
            if t == task:
                seen["n"] += 1
                if seen["n"] == on_call:
                    raise exc
            return original(t, *args, **kwargs)
        rt.llm.structured = structured
        return rt, sc
    monkeypatch.setattr(off, "offline_runtime", patched)


class FixedLabelLLM:
    """Answers 'entity'/'entities' with one fixed label and 'alias' with 'same'."""
    def __init__(self, label="regulatory T cell"):
        self.label = label

    def structured(self, task, schema, system, user, role="reasoning", ctx=None, n_items=1):
        from bmira.schemas import AliasBatch, AliasVerdict, EntityBatch, EntityItem, EntityResolution
        if task == "entities":
            return EntityBatch(items=[EntityItem(surface=s, normalized_label=self.label, category="cell_type",
                                                 confidence=0.9) for s in ctx["surfaces"]])
        if task == "alias":
            return AliasBatch(verdicts=[AliasVerdict(label_a=a, label_b=b, same_entity=True) for a, b in ctx["pairs"]])
        return EntityResolution(normalized_label=self.label, category="cell_type", confidence=0.9)


def fake_ols(table):
    """A stand-in for OLS: lookup key -> (id, label)."""
    import bmira.normalize as nz
    return lambda n: nz.Concept(*table[nz.lookup_key(n)], "chemical", "ols", 0.9) if nz.lookup_key(n) in table else None



def run_session(tmp_path, questions="one\ntwo\n", *args, out="s.json"):
    """`python -m bmira.experiments --offline --quiet` on a questions file written to tmp_path.
    Returns the session file's contents (UTF-8; Windows defaults differ)."""
    from bmira.experiments import main
    qs = tmp_path / "q.txt"
    qs.write_text(questions, encoding="utf-8")
    path = main(["--offline", "--quiet", "--questions", str(qs), "--out", str(tmp_path / out), *args])
    return json.loads(path.read_text(encoding="utf-8"))


def parsed_question(**fields):
    """A ParsedQuestion with placeholder text in every required field not given."""
    base = {"population_model": "m", "exposure": "a", "comparator": "c", "outcome": "o", "mechanism_hypothesis": "h"}
    return ParsedQuestion(**{**base, **fields})


def stub_ols(monkeypatch, *docs):
    """Makes every OLS search answer `docs`: (obo_id, label, synonyms, ontology_name) tuples."""
    import bmira.normalize as nz

    class Response:
        def json(self):
            return {"response": {"docs": [{"obo_id": i, "label": lab, "synonym": list(syn), "ontology_name": o,
                                           "iri": ""} for i, lab, syn, o in docs]}}
    monkeypatch.setattr(nz.requests, "get", lambda *a, **k: Response())


class StubLLM:
    """Answers every structured() call with `reply` (a value, or a function of task, user, ctx and
    n_items) and keeps the prompts it was sent."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    @property
    def prompt(self):
        return self.prompts[-1] if self.prompts else ""

    def structured(self, task, schema, system, user, **kw):
        self.prompts.append(user)
        if callable(self.reply):
            return self.reply(task, user, kw.get("ctx"), kw.get("n_items", 1))
        return self.reply
