import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import os
import docx2txt
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

# Updated: Accept model path dynamically
@st.cache_resource
def load_models(model_path):
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.eval(), tokenizer, semantic_model

def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".docx"):
        return docx2txt.process(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        pdf = PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        return ""

def chunk_text(text, max_words=120, overlap=40):
    """
    Split text into smaller overlapping chunks for better retrieval accuracy.
    New setting: 120 words per chunk, 40-word overlap.
    """
    import re
    words = re.split(r'\s+', text)
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        if chunk:
            chunks.append(chunk)
    return chunks

def predict_answer(model, tokenizer, question, context, threshold, max_span_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        inputs = tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=384,
            return_offsets_mapping=True,
            padding="max_length"
        )
        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        input_ids = inputs["input_ids"][0]

        best_score = float("-inf")
        best_start = best_end = 0
        for i in range(len(start_logits)):
            for j in range(i, min(i + max_span_length, len(end_logits))):
                score = start_logits[i] + end_logits[j]
                if score > best_score:
                    best_start = i
                    best_end = j
                    best_score = score

        start_char = offset_mapping[best_start][0].item()
        end_char = offset_mapping[best_end][1].item()
        predicted_answer = context[start_char:end_char].strip()

        null_score = (start_logits[0] + end_logits[0]).item()
        logit_diff = null_score - best_score
        na_prob = torch.sigmoid(torch.tensor(logit_diff)).item()

        return predicted_answer if na_prob < threshold and end_char > start_char else "", na_prob, best_score

def render():
    if "rag_qa_results" not in st.session_state:
        st.session_state.rag_qa_results = []
    with st.expander("⚙️ RAG QA Settings", expanded=False):
        selected_model = st.selectbox(
            "Select QA Model",
            ["qa_model_checkpoint", "models/bert_squad2_deepset", "qa_model_checkpoint_sk", "qa_model_checkpoint_170425","qa_model_checkpoint_512_290425"],
            index=0
        )
        st.session_state.threshold = st.selectbox("No-Answer Threshold", [0.2, 0.4,0.5, 0.6, 0.8], index=1)
        st.session_state.confidence_score_threshold = st.selectbox("Confidence Suppression Threshold", [-2.0, -1.0, 0.0,1.0,2.0,3.0,4.0,5.0], index=1)
        st.session_state.max_span_length = st.selectbox("Max Answer Span Length", [15, 30, 50, 80], index=0)
        st.session_state.semantic_similarity_threshold = st.selectbox("Semantic Similarity Threshold", [0.3, 0.4, 0.6, 0.8], index=1)
        st.session_state.enable_semantic_override = st.checkbox("Enable Semantic Override Logic", value=st.session_state.get("enable_semantic_override", True))

    top_k = st.slider("🔎 Retrieve Top-K Chunks", 1, 10, 3)

    uploaded_file = st.file_uploader("📁 Upload document (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])
    question = st.text_input("❓ Enter your question")

    model, tokenizer, semantic_model = load_models(selected_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if st.button("🔍 Predict Answer"):
            if not uploaded_file:
                st.warning("⚠️ Please upload a document file before proceeding.")
            elif not question.strip():
                st.warning("⚠️ Please enter a question before proceeding.")
            else:
                raw_text = extract_text(uploaded_file)
                if not raw_text:
                    st.error("❌ Could not extract text from file.")
                else:
                    chunks = chunk_text(raw_text)
                    chunk_embeddings = semantic_model.encode(chunks, convert_to_tensor=True)
                    question_embedding = semantic_model.encode(question, convert_to_tensor=True)
                    similarities = util.cos_sim(question_embedding, chunk_embeddings)[0]
                    top_k_actual = min(top_k, len(chunks))
                    top_indices = torch.topk(similarities, k=top_k_actual).indices.tolist()

                    # 🔵 Insert Boosting Code Here
                    question_keywords = set(question.lower().split())

                    boosted_scores = []
                    for idx in top_indices:
                        chunk_words = set(chunks[idx].lower().split())
                        keyword_overlap = len(question_keywords.intersection(chunk_words))
                        original_score = similarities[idx].item()
                        boosted_score = original_score + (0.01 * keyword_overlap)
                        boosted_scores.append((idx, boosted_score))

                    boosted_scores = sorted(boosted_scores, key=lambda x: x[1], reverse=True)
                    top_indices = [idx for idx, _ in boosted_scores]

                    st.markdown("---")
                    st.markdown(f"🔍 **Top {top_k_actual} Retrieved Chunks:**")
                    for i, idx in enumerate(top_indices):
                        st.markdown(f"**Chunk {i+1}**")
                        st.info(chunks[idx][:500] + ("..." if len(chunks[idx]) > 500 else ""))

                    best_answer = ""
                    best_conf = float("-inf")
                    best_chunk = ""
                    na_prob = 1.0
                    similarity_score = None
                    suppression_reason = "None"

                    for idx in top_indices:
                       pred, temp_na_prob, score = predict_answer(
                            model,
                            tokenizer,
                            question,
                            chunks[idx],
                            threshold=st.session_state.threshold,
                            max_span_length=st.session_state.max_span_length
                        )
                    if pred and score > best_conf:
                            best_answer = pred
                            best_conf = score
                            best_chunk = chunks[idx]
                            na_prob = temp_na_prob
                            similarity_score = util.cos_sim(
                                semantic_model.encode(question, convert_to_tensor=True),
                                semantic_model.encode(best_chunk, convert_to_tensor=True)
                            ).item()

                    if best_answer and best_conf < st.session_state.confidence_score_threshold:
                        st.warning(f"⚠️ Answer suppressed due to low confidence score: {best_conf:.2f}")
                        best_answer = ""
                        suppression_reason = "Low Confidence"
                    if best_answer and st.session_state.get("enable_semantic_override", False):
                        sim_thresh = st.session_state.get("semantic_similarity_threshold", 0.4)
                        na_thresh = st.session_state.get("threshold", 0.5)
                        if similarity_score is not None and similarity_score < sim_thresh and na_prob > na_thresh:
                            st.warning(f"⚠️ Answer suppressed by Semantic Override: Sim={similarity_score:.3f}, NA_Prob={na_prob:.2f}")
                            best_answer = ""
                            suppression_reason = "Semantic Override"

                    st.markdown("---")
                    st.subheader("🧠 Final Answer")
                    st.success(best_answer if best_answer else "[NO ANSWER]")
                    st.markdown("### 🔬 Debug Info")
                    st.json({
                        "Predicted Answer": best_answer if best_answer else "[NO ANSWER]",
                        "Raw Confidence Score": round(float(best_conf), 4),
                        "No-Answer Probability": round(float(na_prob), 4),
                        "Semantic Similarity Score": round(float(similarity_score), 4) if similarity_score else None,
                        "Suppression Reason": suppression_reason,
                        "Applied Settings": {
                            "Selected Model": selected_model,
                            "No-Answer Threshold": st.session_state.threshold,
                            "Confidence Threshold": st.session_state.confidence_score_threshold,
                            "Max Span Length": st.session_state.max_span_length,
                            "Semantic Similarity Threshold": st.session_state.semantic_similarity_threshold,
                            "Semantic Override Enabled": st.session_state.enable_semantic_override,
                            "Top-K Retrieved": top_k
                        }
                    })
                    if best_answer:
                        st.subheader("📜 Source Context Snippet")
                        highlighted = best_chunk.replace(best_answer, f"**{best_answer}**")
                        st.markdown(highlighted)
                    else:
                        st.info("No valid answer extracted from top chunks.")

                    st.session_state.rag_qa_results.append({
                        "Question": question,
                        "Answer": best_answer if best_answer else "[NO ANSWER]",
                        "No-Answer Probability": round(float(na_prob), 4),
                        "Confidence Score": round(float(best_conf), 4),
                        "Semantic Similarity": round(float(similarity_score), 4) if similarity_score else None,
                        "Top-K Similarities": [round(similarities[idx].item(), 4) for idx in top_indices],
                        "Context Snippet": best_chunk[:500] + ("..." if len(best_chunk) > 500 else ""),
                        "Suppression Reason": suppression_reason,
                    })

    # Show download/export section ONCE after results are appended
if "rag_qa_results" in st.session_state and st.session_state.rag_qa_results:
    st.markdown("---")
    st.subheader("📥 Export QA Results")

    df_export = pd.DataFrame(st.session_state.rag_qa_results)
    csv_data = df_export.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_data,
        file_name="RAG_QA_Results.csv",
        mime="text/csv",
        key="csv_download_once"  # important to prevent Streamlit re-adding button
    )


    # Plot Confidence Score
    st.markdown("### 📊 Confidence Scores vs. Threshold (Interactive)")
    df_plot = pd.DataFrame(st.session_state.rag_qa_results)
    threshold = st.session_state.get("confidence_score_threshold", -1.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df_plot["Confidence Score"],
        x=list(range(1, len(df_plot) + 1)),
        mode="lines+markers",
        name="Confidence Score",
        marker=dict(color="blue"),
        hovertext=df_plot["Question"]
    ))
    fig.add_trace(go.Scatter(
        x=[0, len(df_plot)],
        y=[threshold, threshold],
        mode="lines",
        name=f"Threshold ({threshold})",
        line=dict(color="red", dash="dash")
    ))
    fig.update_layout(
        title="Model Confidence per Prediction",
        xaxis_title="Prediction Index",
        yaxis_title="Confidence Score",
        legend=dict(x=0, y=1),
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Clear history
    if st.button("🗑 Clear History"):
        st.session_state.rag_qa_results.clear()
        st.success("✅ History cleared. You can start fresh.")
