
import streamlit as st

st.set_page_config(
    page_title="Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #faf9f6;
    border-right: 1px solid #e8e6e1;
}
.block-container {
    padding-top: 2rem;
}
.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #888780;
    margin-top: 28px;
    margin-bottom: 14px;
}
.brand {
    font-size: 22px;
    font-weight: 700;
    color: #262626;
}
.subtitle {
    color: #777;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section-label">YOUR SPACE</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="brand">🟢 Study assistant</div>',
                unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📚 Explore a topic"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label">COURSE MATERIAL</div>',
                unsafe_allow_html=True)
    st.caption("200MB per file • PDF")
    uploaded_files = st.file_uploader(
        "Upload course materials",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label">STUDY AGENT</div>',
                unsafe_allow_html=True)
    st.caption("Ask questions about your materials.")

    st.markdown('<div class="section-label">RECENT SESSIONS</div>',
                unsafe_allow_html=True)
    st.caption("No recent sessions yet.")

st.title("◇ A calm place to learn.")
st.write("Explore a topic or ask questions about your course materials.")
