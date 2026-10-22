import streamlit as st

from auth import login, signup
from pipeline import run_research_pipeline


st.set_page_config(
    page_title="Research desk",
    page_icon=":material/auto_stories:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero-kicker { color: #45d6b0; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
    .hero-title { font-size: clamp(2.25rem, 7vw, 6.5rem); font-weight: 850; letter-spacing: -0.04em; line-height: 0.98; margin: 0.3rem 0 0.8rem; white-space: nowrap; }
    .research-word { color: #58e0b5; }
    .desk-word { color: #62a9ff; }
    .hero-copy { color: #9aaba7; font-size: 1.05rem; max-width: 680px; }
    .metric-card { background: #142521; border: 1px solid #265348; border-radius: 14px; padding: 1rem 1.1rem; min-height: 92px; }
    .metric-label { color: #8aa59d; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-value { color: #e4fff5; font-size: 1.65rem; font-weight: 750; margin-top: 0.2rem; }
    .auth-copy { color: #9aaba7; max-width: 520px; font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_auth_page() -> None:
    st.markdown('<div class="hero-kicker">Private research workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title"><span class="research-word">Research</span> <span class="desk-word">Desk</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-copy">Sign in to your research workspace, or create an account to begin building sourced briefs.</div>', unsafe_allow_html=True)
    st.space("medium")

    login_tab, signup_tab = st.tabs([":material/login: Log in", ":material/person_add: Create account"])
    with login_tab:
        with st.container(border=True):
            with st.form("login_form"):
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log in", type="primary", width="stretch")
            if submitted:
                valid, message, name = login(email, password)
                if valid:
                    st.session_state["auth_user"] = {"name": name, "email": email.strip().lower()}
                    st.rerun()
                st.error(message, icon=":material/error:")

    with signup_tab:
        with st.container(border=True):
            with st.form("signup_form"):
                name = st.text_input("Your name", placeholder="Alex Morgan")
                email = st.text_input("Email address", placeholder="you@example.com", key="signup_email")
                password = st.text_input("Password", type="password", help="Use at least 8 characters.", key="signup_password")
                confirm_password = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create account", type="primary", width="stretch")
            if submitted:
                if password != confirm_password:
                    st.error("Passwords do not match.", icon=":material/error:")
                else:
                    created, message = signup(name, email, password)
                    if created:
                        st.success(message, icon=":material/check_circle:")
                    else:
                        st.error(message, icon=":material/error:")

    st.caption("Passwords are stored locally as salted hashes. This is a local app authentication flow, not production identity management.")


if "auth_user" not in st.session_state:
    render_auth_page()
    st.stop()


def render_sources(sources: list[str]) -> None:
    with st.expander(f"Sources ({len(sources)})", icon=":material/link:"):
        if sources:
            for index, source in enumerate(sources, start=1):
                st.markdown(f"{index}. [{source}]({source})")
        else:
            st.caption("No source URLs were returned.")


def render_report(result: dict) -> None:
    report = result["report"]
    mode = result["report_mode"]

    st.success("Research complete", icon=":material/check_circle:")
    st.markdown(f"### {result['topic']}")

    score = result.get("critique").score if result.get("critique") else "--"
    metric_one, metric_two, metric_three = st.columns(3)
    with metric_one:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Output</div><div class="metric-value">{"Summary" if mode == "summary" else "Detailed"}</div></div>', unsafe_allow_html=True)
    with metric_two:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Sources read</div><div class="metric-value">{len(report.sources)}</div></div>', unsafe_allow_html=True)
    with metric_three:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Quality score</div><div class="metric-value">{f"{score:g}/10" if score != "--" else "Pending"}</div></div>', unsafe_allow_html=True)

    if mode == "summary":
        with st.container(border=True):
            st.caption("One-line summary")
            st.markdown(f"#### {report.summary}")
        render_sources(report.sources)
        return

    executive_summary = getattr(report, "executive_summary", report.introduction)
    evidence_and_implications = getattr(report, "evidence_and_implications", [])
    limitations = getattr(report, "limitations", [])
    improvements_to_be_done = getattr(report, "improvements_to_be_done", [])

    with st.container(border=True):
        st.caption("Executive summary")
        st.write(executive_summary)

    with st.container(border=True):
        st.caption("Introduction")
        st.write(report.introduction)

    st.subheader("Key findings", icon=":material/lightbulb:")
    for finding in report.key_findings:
        st.markdown(f"- {finding}")

    if evidence_and_implications:
        st.subheader("Evidence and implications", icon=":material/insights:")
        for point in evidence_and_implications:
            st.markdown(f"- {point}")

    if limitations:
        st.subheader("Limitations", icon=":material/flag:")
        for limitation in limitations:
            st.markdown(f"- {limitation}")

    if improvements_to_be_done:
        st.subheader("Improvements to be done", icon=":material/auto_fix_high:")
        st.caption("Practical ways to make this report stronger, with examples.")
        for improvement in improvements_to_be_done:
            with st.container(border=True):
                st.markdown(f"- {improvement}")

    with st.container(border=True):
        st.caption("Conclusion")
        st.write(report.conclusion)

    render_sources(report.sources)

    if result.get("critique"):
        critique = result["critique"]
        with st.expander("Quality review", icon=":material/rate_review:"):
            st.metric("Score", f"{critique.score:g}/10")
            st.write(critique.feedback)
            st.caption(critique.verdict)


with st.sidebar:
    st.markdown("# :material/auto_stories: Research Desk")
    user = st.session_state["auth_user"]
    st.caption(f"Signed in as {user['name']}")
    if st.button("Log out", icon=":material/logout:", width="stretch"):
        del st.session_state["auth_user"]
        st.session_state.pop("research_result", None)
        st.rerun()
    st.caption("A lightweight, local-first research assistant")
    st.markdown("### Workflow")
    st.markdown(
        "1. Enter a topic\n"
        "2. Choose an output style\n"
        "3. Search and review the report"
    )
    st.markdown("### Model")
    st.badge("Local Ollama", icon=":material/lock:", color="green")
    st.caption("Runs with qwen2.5:3b on your machine. No paid LLM API is required.")
    st.markdown("### Output modes")
    st.caption("Detailed report for deeper reading, or one-line summary for a quick answer.")

st.markdown('<div class="hero-kicker">Local research studio</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title"><span class="research-word">Research</span> <span class="desk-word">Desk</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-copy">A focused research desk that searches, reads, writes, and reviews a sourced brief on demand.</div>', unsafe_allow_html=True)
st.space("small")

with st.container(border=True):
    st.subheader("Start a research brief", icon=":material/search:")
    with st.form("research_form", border=False):
        topic = st.text_input(
            "Research topic",
            placeholder="e.g. How is AI changing healthcare?",
        )
        report_mode = st.segmented_control(
            "Output style",
            options=["Detailed report", "One-line summary"],
            default="Detailed report",
            selection_mode="single",
        )
        submitted = st.form_submit_button(
            "Generate research",
            type="primary",
            icon=":material/arrow_forward:",
            width="stretch",
        )

if submitted:
    if not topic.strip():
        st.warning("Enter a topic before generating a report.", icon=":material/edit:")
    else:
        selected_mode = "summary" if report_mode == "One-line summary" else "detailed"
        with st.status("Researching your topic...", expanded=True) as status:
            progress_slot = st.empty()

            def show_progress(stage: str, message: str) -> None:
                labels = {
                    "search": ":material/search: Search agent",
                    "reader": ":material/menu_book: Reader agent",
                    "writer": ":material/edit_note: Writer agent",
                    "reviewer": ":material/rate_review: Reviewer agent",
                }
                progress_slot.markdown(f"**{labels.get(stage, stage)}**  \n{message}")

            try:
                result = run_research_pipeline(topic.strip(), selected_mode, on_progress=show_progress)
                status.update(label="Research complete", state="complete", expanded=False)
                st.session_state["research_result"] = result
            except Exception as error:
                status.update(label="Research failed", state="error", expanded=True)
                st.error(str(error))

if "research_result" in st.session_state:
    render_report(st.session_state["research_result"])
else:
    st.space("large")
    with st.container(horizontal_alignment="center"):
        st.markdown(":material/arrow_upward: Enter a topic above to begin")