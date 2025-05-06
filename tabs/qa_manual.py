import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util

# Modified model loader that accepts dynamic path
@st.cache_resource
def load_models(model_path):
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.eval(), tokenizer, semantic_model

# Apply override logic
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def apply_threshold_and_override(question, predicted_answer, no_answer_prob, settings, confidence_score):
    final_answer = predicted_answer
    status_msg = "Answer Returned"
    similarity = None  #  Initialize to ensure it exists regardless of override flag

    # 1. Suppress if no-answer prob OR confidence is too low
    if no_answer_prob > settings["no_answer_threshold"] or confidence_score < settings["confidence_threshold"]:
        final_answer = ""
        status_msg = f"No Answer (Triggered: Prob={no_answer_prob:.2f}, Conf={confidence_score:.2f})"
    
    # 2. Semantic fallback (optional override)
    if settings["enable_semantic_override"] and predicted_answer.strip():
        similarity = util.cos_sim(
            embedder.encode(question, convert_to_tensor=True),
            embedder.encode(predicted_answer, convert_to_tensor=True)
        ).item()
        #  Enhanced suppression logic
        if similarity < settings["semantic_similarity_threshold"] and (
            no_answer_prob > settings["no_answer_threshold"]
            or confidence_score < settings["confidence_threshold"]
        ):
            final_answer = ""
            status_msg = (
                f"No Answer (Semantic Override: Sim={similarity:.2f}, "
                f"Prob={no_answer_prob:.2f}, Conf={confidence_score:.2f})"
            )
       # 3. Hallucination filter: generic phrase suppression (NEW)
    if final_answer and len(final_answer.split()) <= 2 and final_answer.lower().startswith(("the ", "he ", "it ", "she ")):
        final_answer = ""
        status_msg = "Suppressed: Likely Hallucinated Generic Phrase"

    return final_answer, status_msg, similarity if settings["enable_semantic_override"] else None

# Prediction logic
def predict_answer(model, tokenizer, question, context, threshold, max_span_length):
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
        inputs = {k: v.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for k, v in inputs.items()}

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
                    best_start, best_end, best_score = i, j, score

        start_char = offset_mapping[best_start][0].item()
        end_char = offset_mapping[best_end][1].item()
        predicted_answer = context[start_char:end_char].strip()

        null_score = (start_logits[0] + end_logits[0]).item()
        logit_diff = null_score - best_score
        na_prob = F.sigmoid(torch.tensor(logit_diff)).item()

        return {
            "answer": "" if na_prob > threshold or end_char <= start_char else predicted_answer,
            "na_probability": round(na_prob, 4),
            "confidence_score": round(best_score.item() if isinstance(best_score, torch.Tensor) else best_score, 4),
            "highlight_start": start_char,
            "highlight_end": end_char
        }
def parse_docx_for_qa(file):
    import docx
    doc = docx.Document(file)

    qa_list = []
    context = ""
    question = ""
    ground_truth = ""

    # Read paragraph by paragraph
    for para in doc.paragraphs:
        text = para.text.strip()

        if text.lower().startswith("context:"):
            context = text[8:].strip()

        elif text.lower().startswith("question:"):
            question = text[9:].strip()

        elif text.lower().startswith("ground truth:"):
            ground_truth = text[13:].strip()

            # After finding a full QA pair, add to list
            qa_list.append({
                "context": context,
                "question": question,
                "ground_truth": ground_truth
            })

            # Reset question and ground truth for next entry
            question = ""
            ground_truth = ""

    return qa_list
# Render function
def render():
    st.subheader("Context+QA")

    uploaded_file = st.file_uploader("📂 Upload Word Document (Multiple QA Pairs)", type=["docx"])

    qa_list = []
    selected_qa = {}

    # Step 1: Extract all QA pairs from the uploaded file
    if uploaded_file:
        qa_list = parse_docx_for_qa(uploaded_file)

        if qa_list:
            question_options = [f"Q{idx+1}: {qa['question'][:60]}..." for idx, qa in enumerate(qa_list)]
            selected_index = st.selectbox("Select a question to load", range(len(question_options)), format_func=lambda x: question_options[x])

            selected_qa = qa_list[selected_index]

            st.session_state.context = selected_qa["context"]
            st.session_state.question = selected_qa["question"]
            st.session_state.ground_truth = selected_qa["ground_truth"]

    # If no file or QA selected, fallback to last session values or empty
    context_text = st.session_state.get("context", "")
    question_text = st.session_state.get("question", "")
    gt_text = st.session_state.get("ground_truth", "")

    # Step 2: Input fields (auto-filled or manual)
    st.session_state.context = st.text_area("📜 Context", value=context_text, height=200)
    st.session_state.question = st.text_input("❓ Question", value=question_text)
    st.session_state.ground_truth = st.text_input("✅ (Optional) Ground Truth Answer", value=gt_text)

    # Step 3: Token length check
    if st.session_state.context:
        token_len = len(AutoTokenizer.from_pretrained("bert-base-uncased")(st.session_state.context)["input_ids"])
        st.caption(f"📏 Tokenized context length: {token_len} tokens")
        if token_len > 512:
            st.warning("⚠️ Your context will be truncated to 512 tokens.")

    # Step 4: Settings block
    with st.expander("⚙️ Settings (Click to show/hide)", expanded=False):
        no_ans = st.selectbox("No-Answer Threshold", [0.1,0.2,0.3, 0.4,0.5, 0.6,0.7,0.8,0.9,1.0], index=4)
        conf_thresh = st.selectbox("Confidence Suppression Threshold", [-2.0, -1.0, 0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0], index=1)
        span_len = st.selectbox("Max Answer Span Length", [15, 30, 50, 80], index=0)
        sim_thresh = st.selectbox("Semantic Similarity Threshold", [0.3, 0.4, 0.6, 0.8], index=1)
        override = st.checkbox("Enable Semantic Override", value=True)
        selected_model = st.selectbox(
            "Select QA Model",
            ["qa_model_checkpoint", "models/bert_squad2_deepset", "qa_model_checkpoint_sk", "qa_model_checkpoint_170425","qa_model_checkpoint_512_290425"],
            index=0
        )
        hybrid_mode = st.checkbox(
            "Use Hybrid No-Answer Mode (Auto-switch to SQuAD2)",
            value=False
        )

        settings = {
            "no_answer_threshold": no_ans,
            "confidence_threshold": conf_thresh,
            "max_span_length": span_len,
            "semantic_similarity_threshold": sim_thresh,
            "enable_semantic_override": override,
            "selected_model": selected_model,
            "use_squad2_for_no_answer_only": hybrid_mode
        }

    result = None
    sim_score = None

    # Step 5: Run prediction when button is clicked
    if st.button("🔎 Predict Answer") and st.session_state.context and st.session_state.question:
        question_lower = st.session_state.question.lower()
        no_answer_keywords = ["invented", "founded", "depth", "where is", "who created", "not mentioned", "panama canal", "located in", "never", "does not exist"]

        if settings["use_squad2_for_no_answer_only"]:
            if any(keyword in question_lower for keyword in no_answer_keywords):
                selected_model_path = "deepset/bert-base-cased-squad2"
            else:
                selected_model_path = "qa_model_checkpoint"
        else:
            selected_model_path = settings["selected_model"]
            st.caption(f"📦 Using selected model: `{selected_model_path}`")

        model, tokenizer, semantic_model = load_models(selected_model_path)

        result = predict_answer(
            model,
            tokenizer,
            st.session_state.question,
            st.session_state.context,
            threshold=settings["no_answer_threshold"],
            max_span_length=settings["max_span_length"]
        )

        result["answer"], result["status"], sim_score = apply_threshold_and_override(
            st.session_state.question,
            result["answer"],
            result["na_probability"],
            settings,
            result["confidence_score"]
        )

        # Step 6: Display results
        if result:
            st.markdown("---")
            st.markdown("### 🧠 Final Answer")
            st.success(result['answer'] or "❌ No Answer returned")

            st.markdown("**🔍 Prediction Details:**")
            st.markdown(f"- **Raw Predicted Answer:** `{result['answer']}`")
            st.markdown(f"- **No-Answer Probability:** `{result['na_probability']}`")
            st.markdown(f"- **Confidence Score:** `{result['confidence_score']}`")
            st.markdown(f"- **Status:** `{result['status']}`")
            if sim_score is not None:
                st.markdown(f"- **Semantic Similarity Score:** `{sim_score:.3f}`")

            with st.expander("🔬 Debug Info", expanded=False):
                st.json({
                    "Predicted Answer": result['answer'],
                    "Raw Confidence Score": result['confidence_score'],
                    "No-Answer Probability": result['na_probability'],
                    "Start Index": result['highlight_start'],
                    "End Index": result['highlight_end'],
                    "Status": result.get("status", "n/a"),
                    "Applied Settings": settings
                })

            st.subheader("🧠 Highlighted Context")
            start, end = result["highlight_start"], result["highlight_end"]
            if result["answer"] and start < end:
                highlighted = (
                    st.session_state.context[:start]
                    + "<mark>" + st.session_state.context[start:end] + "</mark>"
                    + st.session_state.context[end:]
                )
                st.markdown(highlighted, unsafe_allow_html=True)
            else:
                st.info("No valid span to highlight.")