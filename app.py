import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import os
import docx2txt
from PyPDF2 import PdfReader


# ✅ This must be the first Streamlit command
st.set_page_config(
    page_title="QA Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
#from tabs.settings import get_current_settings
#get_current_settings()  # Force initialization at startup
# Import tab modules
from tabs import qa_manual, qa_validation, qa_rag 

# -------------------
# Sidebar Info
# -------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Iconic_image_of_question_answering.svg/1200px-Iconic_image_of_question_answering.svg.png", width=120)
st.sidebar.title("Hybrid QA Assistant_SQuAD_V2")
st.sidebar.markdown("AI-powered Question Answering System")



# -------------------
# Shared Session Config (safe defaults only if not already initialized elsewhere)
# -------------------
default_values = {
    "threshold": 0.4,
    "max_span_length": 30,
    "semantic_similarity_threshold": 0.8,
    "confidence_threshold": -1.0,
    "enable_semantic_override": True,
    "selected_model": "qa_model_checkpoint",
    "use_squad2_for_no_answer_only": False
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -------------------
# Tab Routing
# -------------------
tabs = {
     "🔍 Manual QA": qa_manual,
    "📊 Batch Evaluation": qa_validation,
    "🔗 RAG Retrieval QA": qa_rag
   # "⚙️ Settings": settings,
}

selection = st.sidebar.radio("Navigate", list(tabs.keys()))
st.markdown(f"# {selection}")
tabs[selection].render()

