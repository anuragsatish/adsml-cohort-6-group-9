# app.py (Modular Main Entry Point)

import streamlit as st
from tabs import qa_manual, qa_validation, qa_rag

# --- Set Page Configuration (must be first Streamlit command) ---
st.set_page_config(
    page_title="QA Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar Configuration ---
st.sidebar.image("qa.png", width=320)
st.sidebar.title("QA Assistant SQuAD v2")
st.sidebar.markdown("AI-powered Question Answering System")

# --- Session State Defaults ---
def initialize_session_state():
    defaults = {
        "threshold": 0.4,
        "max_span_length": 30,
        "semantic_similarity_threshold": 0.8,
        "confidence_threshold": -1.0,
        "enable_semantic_override": True,
        "selected_model": "qa_model_checkpoint",
        "use_squad2_for_no_answer_only": False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# --- Main Function Routing ---
def main():
    initialize_session_state()

    tabs = {
        "Manual QA": qa_manual,
        "Batch Evaluation QA": qa_validation
        #"Document Retrieval QA": qa_rag
    }

    selected_tab = st.sidebar.radio("Navigate", list(tabs.keys()))
    st.markdown(f"# {selected_tab}")
    tabs[selected_tab].render()

if __name__ == "__main__":
    main()
