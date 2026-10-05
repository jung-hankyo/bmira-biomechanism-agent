"""Link evidence, verdicts, pathway scoring and search allocation (bmira.portfolio). Offline: no keys, no network."""
from bmira import portfolio as pf
from bmira.config import Settings
from bmira.normalize import EntityResolver
from bmira.offline import offline_runtime
from bmira.schemas import Hypothesis

from helpers import make_claim, parsed_question


def test_logic_checks():
    links = {}
    keys = [pf.link_key("A", "increases", "B"), pf.link_key("B", "increases", "C")]
    assert "sign_mismatch" in pf.logic_check(keys, links, {}, "down")[1]
    assert "disconnected" in pf.logic_check([keys[0], pf.link_key("X", "increases", "C")], links, {}, "up")[1]
    assert "cycle" in pf.logic_check([keys[0], pf.link_key("B", "increases", "A")], links, {}, "up")[1]


def test_allocation_explores():
    s = Settings(targets_per_round=2, exploration_slots=1)
    links = {k: pf.LinkEvidence(key=k, subject=k[0], relation="increases", object=k[-1],
                                completeness=c)
             for k, c in [("A|increases|B", 0.3), ("B|increases|C", 0.3), ("A|increases|Z", 0.0)]}
    hyps = [Hypothesis(id="H1", name="lead", origin="llm_seed", links=["A|increases|B", "B|increases|C"], score=0.3),
            Hypothesis(id="H2", name="alt", origin="llm_seed", links=["A|increases|Z"], score=0.0)]
    assert "A|increases|Z" in pf.allocate(hyps, links, s, 1)


def test_allocation_prefers_the_step_nearest_the_exposure():
    """Pilot4: equal-priority steps were ordered by key string, so CHEBI-keyed links beat the
    HDAC route's NCIT and LOCAL steps. The later step here sorts first alphabetically."""
    s = Settings(targets_per_round=1, exploration_slots=0)
    first, later = "Z|increases|Y", "Y|increases|A"
    links = {k: pf.LinkEvidence(key=k, subject=k[0], relation="increases", object=k[-1]) for k in (first, later)}
    hyps = [Hypothesis(id="H1", name="chain", origin="llm_seed", links=[first, later])]
    assert pf.allocate(hyps, links, s, 1) == [first]
    # progress along a route, not its weakest step, sets its weight: a started route beats an unstarted one
    started = pf.LinkEvidence(key="S|increases|T", subject="S", relation="increases", object="T", completeness=0.4)
    idle = pf.LinkEvidence(key="I|increases|J", subject="I", relation="increases", object="J")
    two = {**links, started.key: started, idle.key: idle}
    hyps = [Hypothesis(id="H2", name="started", origin="llm_seed", links=[started.key, later]),
            Hypothesis(id="H3", name="idle", origin="llm_seed", links=[idle.key, first])]
    assert pf.allocate(hyps, two, s, 1) == [started.key]     # by min-step score both routes weigh 0 and idle wins


# P4: more evidence never lowers a link; counted contradiction blocks support.
def test_link_aggregation():
    s, labels = Settings(), {}
    strong = [make_claim("c1", "p1", "increases", "strong"), make_claim("c2", "p2", "increases", "strong")]
    k = pf.link_key("LOCAL:a", "increases", "LOCAL:b")
    g1 = pf.build_links(strong, {}, {}, labels, s)[k].grade
    g2 = pf.build_links(strong + [make_claim("c3", "p3", "increases", "weak")], {}, {}, labels, s)[k].grade
    assert g1 == g2 == "strong"
    contra = strong[:1] + [make_claim("c4", "p4", "decreases", "strong"), make_claim("c5", "p5", "no_effect", "strong")]
    assert pf.build_links(contra, {}, {}, labels, s)[k].status == "contradicted"
    # R6: triaged 'context-dependent' sets opposing claims aside only if contexts truly differ...
    tri = {frozenset(("c1", "c4")), frozenset(("c1", "c5"))}
    assert pf.build_links(contra, {}, {}, labels, s, discounted=tri)[k].status == "contradicted"
    other = strong[:1] + [make_claim("c4", "p4", "decreases", "strong", ctx="tumor"),
                          make_claim("c5", "p5", "no_effect", "strong", ctx="tumor")]
    ln = pf.build_links(other, {}, {}, labels, s, discounted=tri)[k]
    assert set(ln.uncounted) == {"c4", "c5"} and ln.status != "contradicted"
    # ...and never when the opposing evidence is closer to humans than the support
    human = strong[:1] + [make_claim("c4", "p4", "decreases", "strong", ctx="tumor", system="human_primary_cells"),
                          make_claim("c5", "p5", "no_effect", "strong", ctx="tumor", system="human_primary_cells")]
    assert pf.build_links(human, {}, {}, labels, s, discounted=tri)[k].status == "contradicted"


def test_evidence_floor_and_null_asymmetry():
    s, k = Settings(), pf.link_key("LOCAL:a", "increases", "LOCAL:b")
    weak = [make_claim("c1", "p1", "increases", "weak"), make_claim("c2", "p2", "increases", "weak")]
    ln = pf.build_links(weak, {}, {}, {}, s)[k]
    assert ln.status == "insufficient" and "only weak" in ln.reason            # R8
    strong = [make_claim("c1", "p1", "increases", "strong"), make_claim("c2", "p2", "increases", "strong")]
    nulls = [make_claim("c3", "p3", "no_effect", "moderate"), make_claim("c4", "p4", "no_effect", "moderate")]
    ln = pf.build_links(strong + nulls, {}, {}, {}, s)[k]
    assert ln.status == "supported" and set(ln.uncounted) == {"c3", "c4"}     # R5
    review = [make_claim("c5", "p5", "increases", "weak", study_type="review")]
    ln = pf.build_links(strong[:1] + review, {}, {}, {}, s)[k]
    assert ln.n_studies == 1 and "c5" in ln.uncounted                          # R1


def test_findings_on_subtypes_support_the_parent_link():
    parent = pf.link_key("LOCAL:a", "increases", "CL:parent")
    claims = [make_claim("A", "p1", "increases", obj="CL:sub1"), make_claim("B", "p2", "increases", obj="CL:sub2"),
              make_claim("N", "p3", "no_effect", obj="CL:sub1")]
    anc = {"CL:sub1": ("CL:parent",), "CL:sub2": ("CL:parent",)}
    ln = pf.build_links(claims, {}, {}, {}, Settings(), extra={parent}, ancestors=anc)[parent]
    assert ln.status == "supported" and ln.n_studies == 2
    assert ln.n_contra_studies == 0                                        # a subtype null never refutes the parent
    assert pf.build_links(claims, {}, {}, {}, Settings(), extra={parent})[parent].status != "supported"


def test_null_findings_contradict_required_and_modulating_steps():
    null = make_claim("n", "p", "no_effect")
    assert pf._contradicts("required_for", null) and pf._contradicts("sufficient_for", null)
    assert pf._contradicts("modulates", null)
    assert not pf._contradicts("required_for", make_claim("i", "p", "increases"))
    assert not pf._contradicts("binds", null)
    key = pf.link_key("LOCAL:a", "required_for", "LOCAL:b")
    nulls = [make_claim("n1", "p1", "no_effect"), make_claim("n2", "p2", "no_effect")]     # moderate, with comparator
    assert pf.build_links(nulls, {}, {}, {}, Settings(), extra={key})[key].status == "contradicted"


def test_expected_sign_follows_the_exposure_change():
    assert pf.pathway_sign("up", "down") == "down" and pf.pathway_sign("down", "down") == "up"
    assert pf.pathway_sign("up", "up") == "up" and pf.pathway_sign("none", "down") == "none"
    keys = [pf.link_key("N", "decreases", "I")]                       # NAD+ -| inflammation
    assert "sign_mismatch" not in pf.logic_check(keys, {}, {}, pf.pathway_sign("up", "down"))[1]
    assert "sign_mismatch" in pf.logic_check(keys, {}, {}, "up")[1]


def test_one_direct_route_per_outcome_and_a_slot_left_for_expansion():
    """Pilot5 (after N8) had 13 pathways: seven direct routes, four to Treg differing only in relation;
    and the last slots went to routes repeating seeds, so expansion never ran."""
    from bmira.graph import portfolio
    rt, _ = offline_runtime()
    rt.settings.max_hypotheses = 3
    parsed = parsed_question(outcome="b", expected_direction="up")
    claims = [make_claim("c1", "p1", "increases"), make_claim("c2", "p2", "increases"),
              make_claim("c3", "p3", "modulates", grade="weak"), make_claim("c4", "p4", "associated_with", grade="weak"),
              make_claim("m1", "p5", "increases", subj="LOCAL:a", obj="LOCAL:m1"),
              make_claim("m2", "p5", "increases", subj="LOCAL:m1", obj="LOCAL:b"),
              make_claim("n1", "p6", "increases", subj="LOCAL:a", obj="LOCAL:m2"),
              make_claim("n2", "p6", "increases", subj="LOCAL:m2", obj="LOCAL:b"),
              make_claim("k1", "p7", "increases", subj="LOCAL:a", obj="LOCAL:m3"),
              make_claim("k2", "p7", "increases", subj="LOCAL:m3", obj="LOCAL:b")]
    state = {"claims": claims, "parsed": parsed, "exposure": "LOCAL:a", "outcome": "LOCAL:b",
             "outcome_ids": ["LOCAL:b"], "seed_status": "OK", "round_idx": 0, "hypotheses": [], "papers": []}
    out = portfolio(state, rt)["hypotheses"]
    direct = [h for h in out if pf.is_direct(h)]
    assert len(direct) == 1 and "increases" in direct[0].links[0]          # the strongest, not four relations
    assert sum(not pf.is_direct(h) for h in out) == rt.settings.max_hypotheses - 1   # one slot left for expansion


def test_direct_routes_take_no_pathway_slot():
    """Pilot4: H5 and its subtype H6 took two of six slots, so expansion never had room."""
    s = Settings(max_hypotheses=2)
    k = lambda a, b: pf.link_key(a, "increases", b)
    hyps = [Hypothesis(id=i, name=i, origin=o, links=[k(a, b)]) for i, o, a, b in (
        ("H1", "ledger_path", "A", "B"), ("H2", "ledger_path", "A", "C"),
        ("H3", "llm_seed", "A", "M1"), ("H4", "llm_seed", "A", "M2"), ("H5", "llm_seed", "A", "M3"))]
    claims = [make_claim("c1", "p1", "increases", subj="A", obj="B"), make_claim("c2", "p2", "increases", subj="A", obj="B"),
              make_claim("c3", "p3", "increases", grade="weak", subj="A", obj="C")]
    links = pf.build_links(claims, {}, {}, {}, s, extra={h.links[0] for h in hyps})
    kept = [h.id for h in pf.evaluate(hyps, links, {}, "none", s)]
    assert "H1" in kept and "H2" in kept                       # both direct routes survive
    assert sum(h in kept for h in ("H3", "H4", "H5")) == 2     # mechanism routes keep the cap
    assert pf.is_direct(hyps[0]) and not pf.is_direct(hyps[2])
    longer = Hypothesis(id="H6", name="x", origin="ledger_path", links=[k("A", "M1"), k("M1", "B")])
    assert not pf.is_direct(longer)                            # a found multi-step route is a mechanism


def test_seeded_pathways_end_at_the_first_readout():
    """Pilot4's H1 ended 'FOXP3 -> regulatory T cell': FOXP3 is a readout, so that link is a
    definition and no paper states it. The step stayed 'not found' and the pathway stuck at 0."""
    from bmira.schemas import ProposedLink, ProposedPathway
    r = EntityResolver(Settings(ontology_provider="off"))
    chain = [("butyrate", "decreases", "HDAC"), ("HDAC", "decreases", "Histone H3"),
             ("Histone H3", "increases", "FOXP3"), ("FOXP3", "increases", "regulatory T cell")]
    pw = ProposedPathway(name="p", links=[ProposedLink(source=s, relation=rel, target=t) for s, rel, t in chain])
    full, _ = pf.proposal_keys(pw, r)
    cut, _ = pf.proposal_keys(pw, r, {r.resolve("FOXP3").id, r.resolve("regulatory T cell").id})
    assert len(full) == 4 and cut == full[:3]
    assert pf.proposal_keys(pw, r, {r.resolve("butyrate").id})[0] == full           # a start is not an end


def test_binding_has_no_direction_and_signed_effects_show_modulation():
    """Pilot4: 'butyrate binds HCAR2' was split from 'HCAR2 binds butyrate' (the only moderate
    paper), and 'butyrate modulates DCs' was 'not found' beside 'butyrate increases DCs'."""
    binds = pf.link_key("B", "binds", "H")
    claims = [make_claim("A", "p1", "binds", subj="H", obj="B"), make_claim("C", "p2", "binds", grade="weak", subj="B", obj="H")]
    ln = pf.build_links(claims, {}, {}, {}, Settings(), extra={binds})[binds]
    assert ln.status == "supported" and ln.n_studies == 2 and ln.grade == "moderate"
    mod = pf.link_key("B", "modulates", "D")
    claims = [make_claim("E", "p3", "increases", subj="B", obj="D"), make_claim("F", "p4", "decreases", subj="B", obj="D"),
              make_claim("G", "p5", "no_effect", grade="strong", subj="B", obj="D")]
    ln = pf.build_links(claims[:2], {}, {}, {}, Settings(), extra={mod})[mod]
    assert ln.status == "supported" and ln.n_studies == 2
    assert pf.build_links(claims, {}, {}, {}, Settings(), extra={mod})[mod].n_contra_studies == 1   # a null still refutes
    inc = pf.link_key("B", "increases", "D")
    assert pf.build_links(claims[1:2], {}, {}, {}, Settings(), extra={inc})[inc].n_studies == 0     # no loosening here


def test_a_fan_shaped_expansion_is_reduced_to_its_connected_chain():
    """Pilot6's H7: butyrate->HIF, HIF->Th17, HIF->Treg was flagged 'steps do not connect' and failed on
    the Th17 side branch; the chain butyrate->HIF->Treg is Supported."""
    fan = ["B|decreases|HIF", "HIF|increases|TH17", "HIF|decreases|TREG"]
    assert pf.as_chain(fan) == ["B|decreases|HIF", "HIF|decreases|TREG"]
    chain = ["A|increases|M", "M|increases|Z"]
    assert pf.as_chain(chain) == chain
    assert pf.as_chain(["A|increases|M", "X|increases|Z"]) == ["A|increases|M", "X|increases|Z"]   # no route: unchanged
    assert pf.as_chain(["A|increases|Z"]) == ["A|increases|Z"]
    # a shortcut must not replace the mechanism: keep the route through the most proposed steps
    assert pf.as_chain(["A|increases|M", "A|increases|Z", "M|increases|Z"]) == ["A|increases|M", "M|increases|Z"]


def test_unfinished_routes_rank_by_progress_and_say_how_many_steps_hold():
    from bmira.schemas import LinkEvidence
    k1, k2, k3, k4 = (pf.link_key(*t) for t in (("E", "increases", "M"), ("M", "increases", "O"),
                                                 ("E", "increases", "N"), ("N", "increases", "O")))
    ev = lambda k, comp, st: LinkEvidence(key=k, subject=k.split("|")[0], relation="increases", object=k.split("|")[2],
                                          completeness=comp, status=st, reason="r")
    links = {k1: ev(k1, .7, "supported"), k2: ev(k2, 0, "insufficient"), k3: ev(k3, 0, "insufficient"),
             k4: ev(k4, 0, "insufficient")}
    h1 = Hypothesis(id="H1", name="n", origin="llm_seed", links=[k3, k4])
    h2 = Hypothesis(id="H2", name="n", origin="llm_seed", links=[k1, k2])
    out = pf.evaluate([h1, h2], links, {}, "up", Settings())
    assert [h.id for h in out] == ["H2", "H1"]                       # both score 0; H2 has one step done
    assert "1 of 2 steps supported" in out[0].reason and "steps supported" not in out[1].reason

