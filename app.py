import streamlit as st
import tempfile
import os

from main import run_pipeline


# ================= PAGE =================

st.set_page_config(
    page_title="AI Job Matcher",
    page_icon="🎯",
    layout="wide"
)

st.title("AI Job Match & Resume Analyzer")


# ================= SIDEBAR =================

st.sidebar.header("Search Options")

company = st.sidebar.selectbox(
    "Company",
    ["Google", "Microsoft", "Amazon", "Walmart"]
)

location = st.sidebar.text_input(
    "Location",
    "India"
)

level = st.sidebar.selectbox(
    "Experience",
    ["ENTRY", "MID", "SENIOR"]
)

keyword = st.sidebar.text_input(
    "Keyword",
    "Software Engineer"
)

resume_file = st.sidebar.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

run_btn = st.sidebar.button("Run Analysis")


# ================= MAIN =================

if run_btn:

    if not resume_file:
        st.warning("Upload resume first.")
        st.stop()

    # Save temp resume
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(resume_file.read())
        temp_path = f.name

    with st.spinner("Analyzing jobs..."):

        results = run_pipeline(
            temp_path,
            company,
            keyword,
            location,
            level
        )

    os.remove(temp_path)

    if not results:
        st.error("No matching jobs found.")
        st.stop()

    st.success(f"Found {len(results)} roles")

    for i, r in enumerate(results):

        with st.container(border=True):

            st.subheader(f"{i+1}. {r['title']}")

            st.markdown(f"🔗 [Job Link]({r['link']})")

            with st.expander("View Report"):

                st.markdown(r["report"])

            st.download_button(
                "Download Report",
                r["report"],
                file_name=f"report_{i+1}.txt"
            )
