"""Entity resolution, ontology lookup, aliases and quote checks (bmira.normalize). Offline: no keys, no network."""
from bmira import portfolio as pf
from bmira.config import Settings
from bmira.normalize import EntityResolver, span_is_anchored
from bmira.offline import offline_runtime

from helpers import FixedLabelLLM, fake_ols, make_claim, stub_ols


# P2: alias merges survive re-normalization from raw surface text.
def test_alias_registry():
    r = EntityResolver(Settings(ontology_provider="off"))
    a, b = r.resolve("Lactic acid"), r.resolve("lactate")
    r.merge([a.id, b.id])
    assert r.resolve("lactic acid").id == r.resolve("Lactate").id


def test_claim_checks():
    from bmira.normalize import check_claim
    src = "Lactate increased GPR81 expression in tumor cells compared with controls."
    c = make_claim("c", "p", "")
    c.span, c.subject, c.object = src, "lactate", "GPR81 expression"
    c.relation = "did not change"
    assert check_claim(c, src)[0] == "null claim but the quote reports an effect"
    c.relation, c.object = "increased", "IFNG"
    assert check_claim(c, src)[0] == "object not named in quote"
    neg = "Lactate did not increase GPR81 expression in tumor cells compared with controls."
    c.span, c.object = neg, "GPR81"
    assert check_claim(c, neg)[0] == "quote negates the claimed effect"
    fabricated = "Lactate had no effect on GPR81 expression in tumor cells compared with controls."
    assert not span_is_anchored(fabricated, fabricated.replace("no effect", "an effect"))


def test_attribute_split_and_sections():
    import xml.etree.ElementTree as ET
    from bmira.normalize import split_attribute
    from bmira.sources import sectioned_text
    assert split_attribute("IFNG expression") == ("IFNG", "expression")
    assert split_attribute("NAD+ levels") == ("NAD+", "amount")
    assert split_attribute("T cell activation") == ("T cell activation", "none")
    body = ET.fromstring("<body><sec><title>Methods</title><p>M</p></sec><sec><title>Results</title>"
                         "<p>R</p><fig><caption>Fig 1 legend</caption></fig></sec></body>")
    text = sectioned_text("abs", body, 1000)
    assert text.index("RESULTS: R") < text.index("FIGURE LEGENDS") < text.index("METHODS: M")


def test_anchor_rejects_stitched_span():
    src = "Lactate increased histone lactylation in T cells. Many unrelated words follow here. IFNG fell."
    assert span_is_anchored("Lactate increased histone lactylation in T cells.", src)
    assert not span_is_anchored("Lactate increased histone lactylation and IFNG fell sharply.", src)


def test_entity_variants_collapse_to_one_node():
    from bmira.normalize import entity_of, entity_parts, lookup_key, singular
    variants = ["Induction of colonic regulatory T cells", "regulatory T-cell frequency",
                "Bone marrow, splenic and Peyer’s patch regulatory T cells",
                "Regulatory T cells in pancreatic lymph nodes", "butyrate-induced regulatory T cells"]
    assert {lookup_key(singular(entity_of(v)[0])) for v in variants} == {"regulatory t cell"}
    assert entity_of("Induction of colonic regulatory T cells")[1:] == ("differentiation", "colon")
    assert entity_of("T cell activation") == ("T cell activation", "none", "")      # phenotype kept
    assert entity_parts("NFAT1 and SMAD3") == (["NFAT1", "SMAD3"], "")
    assert entity_parts("signal transducer and activator of transcription 3")[0] == [
        "signal transducer and activator of transcription 3"]


def test_mention_check_accepts_abbreviations_and_previous_sentence():
    from bmira.normalize import abbreviations, check_claim
    src = ("Mice received sodium butyrate (NaB) in drinking water. NaB increased Foxp3 expression in "
           "colonic T cells compared with controls. This metabolite also increased IL-10 in the same cells.")
    c = make_claim("c", "p", "increases")
    c.subject, c.object, c.relation = "sodium butyrate", "Foxp3", "increased"
    c.span = "NaB increased Foxp3 expression in colonic T cells compared with controls."
    assert check_claim(c, src, abbreviations(src)) == ("", [])
    c.object, c.span = "IL-10", "This metabolite also increased IL-10 in the same cells."
    assert check_claim(c, src, abbreviations(src))[0] == ""             # named one sentence earlier
    c.subject = "not specified"
    assert check_claim(c, src)[0] == "subject not specified"


def test_ontology_prefers_species_agnostic_and_rejects_other_species(monkeypatch):
    def docs(*items):
        stub_ols(monkeypatch, *[(i, lab, ["IL-10"], "pr") for i, lab in items])
    r = EntityResolver(Settings(ontology_provider="ols"))
    docs(("PR:1", "interleukin-10 (chicken)"))
    assert r._ols("IL-10") is None
    docs(("PR:1", "interleukin-10 (chicken)"), ("PR:2", "interleukin-10 (mouse)"), ("PR:3", "interleukin-10"))
    assert r._ols("IL-10").id == "PR:3"
    docs(("PR:2", "interleukin-10 (mouse)"), ("PR:4", "interleukin-10 (human)"))
    assert r._ols("IL-10").id == "PR:4"


def test_batched_resolution_and_disk_cache(tmp_path):
    from bmira.normalize import EntityResolver
    from bmira.offline import SurrogateLLM, load_scenario
    s = Settings(ontology_provider="llm", cache_dir=str(tmp_path))
    llm = SurrogateLLM(load_scenario())
    r = EntityResolver(s, llm)
    r.resolve_many(["lactate", "NAD+", "glycolytic flux", "lactate"])
    assert llm.calls["entities"] == 1 and llm.items["entities"] == 3     # one call for three names
    r.save()
    llm2 = SurrogateLLM(load_scenario())
    r2 = EntityResolver(s, llm2)
    r2.resolve_many(["lactate", "NAD+"])
    assert llm2.calls["entities"] == 0 and r2.disk_hits == 2              # reused across runs


def test_llm_label_is_looked_up_before_going_local():
    import bmira.normalize as nz
    cl = nz.Concept("CL:1", "regulatory T cell", "cell_type", "ols", 0.9)
    r = nz.EntityResolver(Settings(ontology_provider="hybrid"), FixedLabelLLM())
    r._ols = lambda name: cl if nz.lookup_key(name) == "regulatory t cell" else None
    r.resolve_many(["Treg cell", "regulatory T cell"])                  # batched path
    assert r.resolve("Treg cell").id == "CL:1"
    assert r.resolve("Foxp3 Treg ratio").id == "CL:1"                   # single path


def test_ols_skips_allele_terms(monkeypatch):
    stub_ols(monkeypatch, ("NCIT:C102493", "HDAC9 wt Allele", ["HDAC"], "ncit"))
    assert EntityResolver(Settings(ontology_provider="ols"))._ols("HDAC") is None


def test_clean_drops_abbreviation_parentheses():
    from bmira.normalize import clean, lookup_key
    assert clean("histone deacetylase (HDAC)") == "histone deacetylase"
    assert clean("Interleukin-2 receptor subunit alpha (CD25") == "Interleukin-2 receptor subunit alpha"
    assert clean("(CD25)") == "CD25" and clean("CD25") == "CD25"
    assert clean("interleukin-10 (mouse)") == "interleukin-10 (mouse)"      # not an abbreviation
    assert lookup_key("histone deacetylase (HDAC)") == lookup_key("histone deacetylase")


def test_pathway_nodes_join_alias_consolidation():
    import bmira.normalize as nz
    r = nz.EntityResolver(Settings(ontology_provider="off"))
    a, b = r.resolve("histone H3 lysine 9 acetylation"), r.resolve("acetylated histone H3 lysine 9")
    claims = [make_claim("c1", "p", "increases", subj=a.id, obj="LOCAL:b")]       # b is only in a pathway
    assert nz.consolidate_aliases(claims, r, FixedLabelLLM(), {}, extra_ids=[b.id]) == 1
    assert r.canonical(a).id == r.canonical(b).id


def test_but_not_phrase_does_not_negate_the_claim():
    from bmira.normalize import check_claim
    src = ("We show that butyrate but not pentanoate exerts a concentration-dependent effect on "
           "Treg and Th17 differentiation.")
    c = make_claim("c", "p", "")
    c.subject, c.object, c.relation, c.span = "butyrate", "Treg", "exerts a concentration-dependent effect on", src
    assert check_claim(c, src)[0] == ""
    src2 = "Treg generation was potentiated by propionate, an HDAC inhibitor, but not acetate."
    c.subject, c.object, c.relation, c.span = "acetate", "HDAC", "lacks", src2
    assert check_claim(c, src2) == ("", [])                              # 'lacks' is a null relation


def test_greek_letters_and_charges_keep_entities_apart():
    from bmira.normalize import local_concept, lookup_key
    assert lookup_key("IFN-γ") != lookup_key("IFN-α")             # both used to be 'ifn'
    assert lookup_key("IL-1β") != lookup_key("IL-1α")
    assert lookup_key("TGF-β1") == lookup_key("TGF-beta1")
    assert lookup_key("NAD+") == lookup_key("NAD(+)") == lookup_key("NAD⁺")
    assert local_concept("IFN-γ").id != local_concept("IFN-α").id


def test_genotype_notation_is_not_split_into_two_entities():
    from bmira.normalize import entity_parts
    assert entity_parts("Tet2−/− bone marrow")[0] == ["Tet2−/− bone marrow"]
    assert entity_parts("Foxp3-/- mice")[0] == ["Foxp3-/- mice"]
    assert entity_parts("GPR81/HCAR1")[0] == ["GPR81", "HCAR1"]               # real alternatives still split


def test_loss_words_are_stripped_from_entities_but_not_from_phenotypes():
    from bmira.normalize import entity_change, entity_of
    assert entity_of("age-related NAD+ decline")[0] == "NAD+" and entity_change("age-related NAD+ decline") == "down"
    assert entity_of("Tet2 loss")[0] == "Tet2" and entity_change("Tet2 loss") == "down"
    assert entity_of("vitamin D deficiency")[0] == "vitamin D"
    assert entity_of("bone loss")[0] == "bone loss" and entity_change("bone loss") == ""        # a phenotype
    assert entity_change("Tet2-deficient macrophages") == "" and entity_change("Tet2") == ""     # a cell descriptor


def test_salt_acid_and_given_forms_name_the_parent_chemical():
    import bmira.normalize as nz
    assert nz.entity_of("Sodium butyrate treatment")[0] == "Sodium butyrate"
    assert nz.entity_of("Butyrate supplementation")[0] == "Butyrate"
    assert nz.entity_of("provision of butyrate")[0] == "butyrate"
    assert nz.parent_chemical("sodium butyrate") == "butyrate" and nz.parent_chemical("butyric acid") == "butyrate"
    assert nz.parent_chemical("nucleic acid") == "nucleic acid"
    assert nz.parent_chemical("magnesium sulfate") == "magnesium sulfate"     # the metal is the agent
    r = nz.EntityResolver(Settings(ontology_provider="ols"))
    r._ols = fake_ols({"nab": ("CHEBI:64103", "sodium butyrate"), "butyrate": ("CHEBI:17968", "butyrate")})
    assert {r.resolve(x).id for x in ("NaB", "sodium butyrate", "butyric acid", "butyrate")} == {"CHEBI:17968"}


def test_ols_rejects_measurements_strains_and_bad_ids_and_prefers_labels(monkeypatch):
    """Real OLS answers from pilot4's names."""
    def ols(*docs):
        stub_ols(monkeypatch, *docs)
    r = EntityResolver(Settings(ontology_provider="ols"))
    ols(("CHEBI:17154", "nicotinamide", ["niacin"], "chebi"), ("CHEBI:15940", "nicotinic acid", ["Niacin"], "chebi"),
        ("NCIT:C689", "Niacin", [], "ncit"))
    assert r._ols("niacin").id == "NCIT:C689"                     # was nicotinamide
    ols(("PR:O13754", "Hsp70/Hsp90 co-chaperone cns1 (Schizosaccharomyces pombe 972h-)", ["CNS1"], "pr"))
    assert r._ols("CNS1") is None                                  # the Foxp3 enhancer is not a yeast protein
    ols(("NCIT:C166072", "Forkhead Box Protein P3 Measurement", ["forkhead box p3"], "ncit"))
    assert r._ols("forkhead box p3") is None
    ols(("NCIT:C74814", "Interleukin 18 Measurement", ["interleukin 18"], "ncit"),
        ("NCIT:C20520", "Interleukin-18", [], "ncit"))
    assert r._ols("interleukin 18").id == "NCIT:C20520"
    assert r._ols("interleukin 18 measurement").id == "NCIT:C74814"   # asked for by name
    ols(("1318", "C3", [], "mondo"))
    assert r._ols("C3") is None


def test_modifications_and_inhibition_name_the_bare_entity_in_claims_and_pathways():
    from bmira.graph import _set_concepts
    from bmira.normalize import entity_change, entity_of, with_mark
    from bmira.schemas import ProposedLink, ProposedPathway
    assert entity_of("HDAC inhibition")[0] == "HDAC" and entity_change("HDAC inhibition") == "down"
    assert entity_of("inhibition of HDAC")[0] == "HDAC"
    assert entity_of("histone deacetylase")[0] == "histone deacetylase"
    span = "Butyrate enhanced histone H3 acetylation in the promoter of Foxp3."
    assert with_mark("Histone H3", "modification", span) == "Histone H3 acetylation"
    assert with_mark("Histone H3", "amount", span) == "Histone H3"
    rt, _ = offline_runtime()
    c = make_claim("c", "p", "increases", object_attribute="modification")
    c.subject, c.object, c.span = "butyrate", "Histone H3", span
    _set_concepts(c, rt)
    r = rt.resolver
    assert c.object_concept == r.resolve("Histone H3 acetylation").id != r.resolve("histone lactylation").id
    pw = ProposedPathway(name="p", links=[ProposedLink(source="HDAC inhibition", relation="increases",
                                                       target="Histone H3 acetylation")])
    keys, _ = pf.proposal_keys(pw, r)                                  # the pathway names the same node
    assert keys == [pf.link_key(r.resolve("HDAC").id, "decreases", c.object_concept)]


def test_local_cell_subtypes_inherit_the_cell_type_they_name():
    import bmira.normalize as nz
    treg = nz.Concept("CL:0000815", "regulatory T cell", "cell_type", "ols", 0.9, ("CL:0000084",))
    dc = nz.Concept("CL:0000451", "dendritic cell", "cell_type", "ols", 0.9)
    r = nz.EntityResolver(Settings(ontology_provider="hybrid"))
    r._ols = lambda n: {"regulatory t cell": treg, "dendritic cell": dc}.get(nz.lookup_key(n))
    sub = r._labelled("FOXP3-positive regulatory T cell", "cell_type", 0.9)
    assert sub.id.startswith("LOCAL:") and sub.parents == ("CL:0000815",) and "CL:0000084" in sub.ancestors
    assert r._labelled("Slc5a8-null dendritic cell", "cell_type", 0.9).parents == ("CL:0000451",)
    assert r._labelled("regulatory T cell balance", "phenotype", 0.9).ancestors == ()   # not a cell type
    link = pf.link_key("CHEBI:17968", "increases", "CL:0000815")
    claims = [make_claim("A", "p1", "increases", subj="CHEBI:17968", obj=sub.id),
              make_claim("B", "p2", "increases", subj="CHEBI:17968", obj="CL:0000815")]
    ln = pf.build_links(claims, {}, {}, {}, Settings(), extra={link}, ancestors={sub.id: sub.ancestors})[link]
    assert ln.n_studies == 2                                              # the subtype finding counts (R9)


def test_quote_check_reads_the_first_use_of_the_verb():
    """Pilot5 dropped these as 'quote negates the claimed effect'."""
    from bmira.normalize import check_claim

    def verdict(subj, rel, obj, span):
        c = make_claim("c", "p", "increases")
        c.subject, c.relation, c.object, c.span = subj, rel, obj, span
        return check_claim(c, span)[0]
    contrast = ("SB also clearly inhibited the phosphorylation of AKT and NF-kB p65 in LPS-induced WT mouse primary "
                "peritoneal macrophages, but failed to inhibit this phenomenon in LPS-induced GPR109a-/- macrophages.")
    assert verdict("SB", "inhibited", "AKT", contrast) == ""
    assert verdict("butyrate", "promote", "Tregs", "These genes suggest that butyrate could not only promote Tregs "
                   "but also suppress Tconvs and inflammatory cytokines.") == ""
    assert verdict("butyrate", "increased", "Foxp3", "Butyrate did not increase Foxp3, whereas propionate "
                   "increased Foxp3 in the same cultures.") == "quote negates the claimed effect"
    assert verdict("butyrate", "induced", "IL-10", "Butyrate failed to induce IL-10 in naive T cells.") \
        == "quote negates the claimed effect"


def test_paper_defined_abbreviations_name_the_long_form():
    """Pilot5: 'SB' became LOCAL:sb in the paper that defines 'sodium butyrate (SB)'; live OLS
    returns no exact match for SB or NaB."""
    from bmira.normalize import abbreviations, check_claim, expand_abbreviations
    text = ("Mice were treated with sodium butyrate (SB) or vehicle. Cells received butyrate (Bu) or acetate (Ac). "
            "Short-chain fatty acids (SCFAs) were measured. Sodium butyrate (NaB) was also tested.")
    full = expand_abbreviations(abbreviations(text))
    assert full["sb"] == "sodium butyrate" and full["bu"] == "butyrate" and full["ac"] == "acetate"
    assert full["scfas"] == "short-chain fatty acids"
    assert "nab" not in full                                       # 'Na' is a symbol, not letters of a word
    c = make_claim("c", "p", "decreases")                              # the quote check still sees the abbreviation
    c.subject, c.relation, c.object = "SB", "decreased", "intestinal permeability"
    c.span = "SB decreased the intestinal permeability in TNBS-induced WT mice."
    src = "Mice were treated with sodium butyrate (SB) or vehicle. " + c.span
    assert check_claim(c, src, abbreviations(src))[0] == ""


def test_identical_labels_merge_without_asking_the_model():
    """'t helper 17 cell' named two concepts in pilots 4 and 5: CL (cell_type) and NCIT (other)."""
    import bmira.normalize as nz
    cl = nz.Concept("CL:0000899", "T-helper 17 cell", "cell_type", "ols", 0.9)
    nc = nz.Concept("NCIT:C113815", "T Helper 17 Cell", "other", "ols", 0.9)
    gene = nz.Concept("PR:1", "T helper 17 cell", "gene_or_protein", "ols", 0.9)       # a clash is not merged
    r = nz.EntityResolver(Settings(ontology_provider="off"))
    for c in (cl, nc, gene):
        r.concepts[c.id] = c

    class NoLLM:
        def structured(self, *a, **k):
            raise AssertionError("the model should not be asked")
    claims = [make_claim(f"c{i}", "p", "increases", subj=c.id, obj=c.id) for i, c in enumerate((cl, nc))]
    assert nz.consolidate_aliases(claims, r, NoLLM(), {}) == 1
    assert r.canonical(nc).id == r.canonical(cl).id == "CL:0000899"


def test_activity_is_an_attribute_so_a_seeded_node_meets_the_claims():
    """Pilot6: seed node 'histone deacetylase activity' resolved to a GO process while claims said 'HDAC'
    (attribute activity) -> NCIT; butyrate->HDAC (2 papers) sat off the seeded route, which read 'not found yet'."""
    from bmira.normalize import entity_of
    assert entity_of("histone deacetylase activity") == ("histone deacetylase", "activity", "")
    assert entity_of("HDAC activity")[:2] == ("HDAC", "activity")
    assert entity_of("activity")[1] == "none" and entity_of("NF-kB signaling")[:2] == ("NF-kB", "activity")
    for x in ("AMPK activity", "mTORC1 activity", "telomerase activity", "HDAC enzymatic activity", "NF-kB activity"):
        assert entity_of(x)[1] == "activity", x
    # a descriptive word before 'activity' names a phenotype, not a molecule: these stay whole
    for x in ("physical activity", "disease activity", "NK cell cytotoxic activity", "phagocytic activity",
              "suppressive activity"):
        assert entity_of(x) == (x, "none", ""), x

