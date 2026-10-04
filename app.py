"""B-MiRA chat interface.  Run:  streamlit run app.py

First message  -> a full investigation (search, extract, grade, pathway portfolio, report).
Later messages -> follow-up questions answered only from that run's evidence.
'/new <question>' starts a fresh investigation.
"""
import json
import os

import pandas as pd
import streamlit as st

from bmira import Runtime, Settings, __version__
from bmira import portfolio as pf
from bmira.chat import answer, portfolio_rows
from bmira.offline import load_scenario, offline_runtime
from bmira.telemetry import execute, summarize

st.set_page_config(page_title="B-MiRA", page_icon="🧬", layout="wide")
ss = st.session_state
ss.setdefault("messages", [])     # {"role", "content", optional "rows", "report", "log"}
ss.setdefault("run", None)        # final state of the latest investigation
ss.setdefault("rt", None)         # runtime of that investigation (its LLM answers follow-ups)
ss.setdefault("metrics", None)    # telemetry summary of that investigation

# ── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"B-MiRA v{__version__}")
    st.caption("Biomedical Mechanism Inference Research Agent")
    mode = st.radio("Mode", ["Offline demo", "Live"], help=(
        "Offline demo runs a synthetic scenario with a scripted model: no keys, no network. "
        "Live searches PubMed and calls a real LLM."))
    live = mode == "Live"
    if live:
        provider = st.selectbox("LLM provider", ["openai", "anthropic"])
        api_key = st.text_input(f"{provider} API key", type="password",
                                value=os.environ.get(f"{provider.upper()}_API_KEY", ""))
        ncbi_email = st.text_input("NCBI email", value=os.environ.get("NCBI_EMAIL", ""))
        ncbi_key = st.text_input("NCBI API key (optional)", type="password",
                                 value=os.environ.get("NCBI_API_KEY", ""))
    max_rounds = st.slider("Max search rounds", 1, 8, 5)
    if st.button("New conversation", use_container_width=True):
        ss.messages, ss.run, ss.rt, ss.metrics = [], None, None, None
        st.rerun()
    st.divider()
    st.caption("Verdicts: **Supported** · **Contradicted** · **Insufficient evidence**. "
               "Scores rank pathways; they are not probabilities.")

DEMO_QUESTION = load_scenario()["question"]


def make_runtime() -> Runtime:
    if not live:
        return offline_runtime(max_rounds=max_rounds)[0]
    if not api_key:
        raise RuntimeError(f"Enter a {provider} API key in the sidebar.")
    settings = Settings(provider=provider, max_rounds=max_rounds, ncbi_email=ncbi_email,
                        ncbi_api_key=ncbi_key)
    return Runtime.live(settings, api_key=api_key)


def investigate(question: str):
    rt = make_runtime()
    with st.status("Investigating…", expanded=True) as status:
        def progress(nodes, lines):
            status.update(label=f"Investigating… ({', '.join(nodes)})")
            for line in lines:
                status.write(line)
        final, info = execute(question, rt, on_progress=progress, echo=False)
        if info["status"] != "completed":
            status.update(label="Investigation stopped", state="error", expanded=True)
            raise RuntimeError(f"{info['status']} at step {info['failed_node']}: {info['error']}")
        status.update(label="Investigation finished", state="complete", expanded=False)
    return final, rt, info


def summary(state) -> str:
    lead = state["hypotheses"][0] if state["hypotheses"] else None
    head = (f"**Leading pathway:** {lead.name}: **{pf.STATUS_LABEL[lead.status]}** "
            f"(score {lead.score}). {lead.reason}." if lead else "**No pathway could be formed.**")
    counts = pd.Series([pf.STATUS_LABEL[h.status] for h in state["hypotheses"]]).value_counts()
    tally = ", ".join(f"{n} {label}" for label, n in counts.items())
    warn = "".join(f"\n- ⚠️ {w}" for w in state.get("warnings", []))
    return (f"{head}\n\n{len(state['hypotheses'])} pathways considered: {tally}. "
            f"Search stopped: {pf.STOP_LABEL[state['gate']]} after {state['round_idx']} rounds."
            + warn + "\n\nAsk a follow-up question, or type `/new <question>` to start over.")


def render(m, i):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "rows" in m:
            st.dataframe(pd.DataFrame(m["rows"]), hide_index=True, use_container_width=True)
            with st.expander("Full report"):
                st.markdown(m["report"])
            with st.expander("Run log"):
                st.code("\n".join(m["log"]) or "(empty)")
            st.download_button("Download report (.md)", m["report"], f"bmira_report_{i}.md",
                               key=f"dl_{i}")
        for issue in m.get("issues", []):
            st.caption(f"⚠️ Verifier: {issue}")


# ── conversation ────────────────────────────────────────────────────────────
if not ss.messages:
    st.markdown("### Ask a mechanism question\nB-MiRA searches the literature, builds an evidence "
                "graph, and weighs several candidate pathways against each other.")
    if not live:
        st.info(f"Offline demo: whatever you type, the synthetic scenario runs: *{DEMO_QUESTION}*")

for i, m in enumerate(ss.messages):
    render(m, i)

prompt = st.chat_input("Ask a mechanism question, or a follow-up about the last run")
if prompt:
    new = prompt.strip().startswith("/new")
    question = prompt.strip()[4:].strip() if new else prompt.strip()
    ss.messages.append({"role": "user", "content": prompt})
    render(ss.messages[-1], len(ss.messages) - 1)
    try:
        if ss.run is None or new:
            if not question:
                raise ValueError("Type a question after /new.")
            final, rt, info = investigate(question if live else DEMO_QUESTION)
            if "links" not in final:             # out of scope: stopped after parsing, nothing to follow up on
                msg = {"role": "assistant", "content": final.get("report") or "No investigation was run."}
            else:
                ss.run, ss.rt, ss.metrics = final, rt, summarize(final, rt, info)
                msg = {"role": "assistant", "content": summary(final), "rows": portfolio_rows(final),
                       "report": final["report"], "log": info["log"]}
        else:
            with st.spinner("Reading the run's evidence…"):
                reply, issues = answer(ss.run, ss.rt, question, ss.messages[:-1])
            msg = {"role": "assistant", "content": reply, "issues": issues}
    except Exception as e:                       # show the reason; keep the conversation alive
        msg = {"role": "assistant", "content": f"**Could not complete this step:** {e}"}
    ss.messages.append(msg)
    render(msg, len(ss.messages) - 1)

if ss.run is not None:
    with st.sidebar:
        st.download_button("Download run summary (.json)", json.dumps(
            ss.metrics, indent=1, ensure_ascii=False,
            default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o)),
            "bmira_run_summary.json", use_container_width=True,
            help="Metrics and revision signals for this run; same format as the batch runner.")
        if ss.metrics["signals"]:
            st.caption("Revision signals: " + ", ".join(x["signal"] for x in ss.metrics["signals"]))
