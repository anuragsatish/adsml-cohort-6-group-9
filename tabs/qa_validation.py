import streamlit as st
import json
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string, re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import io

# Load models dynamically
@st.cache_resource
def load_models(model_path):
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.eval(), tokenizer, semantic_model

# --- Evaluation Utilities ---
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_em(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

# Core QA prediction logic
@torch.no_grad()
def predict_answer(model, tokenizer, question, context, threshold, max_span_length):
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

    return predicted_answer, na_prob, best_score.item(), start_char, end_char

# Post-processing override logic
def apply_override(answer, question, na_prob, confidence, settings, semantic_model, is_impossible):
    if is_impossible:
        return "", "No Answer"

    status = "Answer Returned"
    if na_prob > settings["no_answer_threshold"] or confidence < settings["confidence_threshold"]:
        return "", f"No Answer (Triggered: Prob={na_prob:.2f}, Conf={confidence:.2f})"

    if settings["enable_semantic_override"] and answer:
        sim = util.cos_sim(
            semantic_model.encode(question, convert_to_tensor=True),
            semantic_model.encode(answer, convert_to_tensor=True)
        ).item()

        if sim < settings["semantic_similarity_threshold"] and na_prob > settings["no_answer_threshold"]:
            return "", f"No Answer (Semantic Override: Sim={sim:.2f})"

    return answer, status

# Tab render

def render():
    #st.subheader("📊 Batch Evaluation")
        # Local Settings Panel (Replaces dependency on get_current_settings)
    with st.expander("⚙️ Evaluation Settings", expanded=False):
        selected_model = st.selectbox(
            "Select QA Model",
            ["qa_model_checkpoint", "models/bert_squad2_deepset", "qa_model_checkpoint_sk", "qa_model_checkpoint_170425"],
            index=0
        )
        hybrid_mode = st.checkbox("Use Hybrid No-Answer Mode", value=False)
        no_ans_thresh = st.selectbox("No-Answer Threshold", [0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1.0], index=4)
        conf_thresh = st.selectbox("Confidence Suppression Threshold", [-2.0, -1.0, 0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8,0,9.0,10.0], index=1)
        span_len = st.selectbox("Max Answer Span Length", [15, 30, 50, 80], index=1)
        sim_thresh = st.selectbox("Semantic Similarity Threshold", [0.3, 0.4, 0.6, 0.8], index=1)
        override = st.checkbox("Enable Semantic Override Logic", value=True)

    settings = {
        "selected_model": selected_model,
        "use_squad2_for_no_answer_only": hybrid_mode,
        "no_answer_threshold": no_ans_thresh,
        "confidence_threshold": conf_thresh,
        "max_span_length": span_len,
        "semantic_similarity_threshold": sim_thresh,
        "enable_semantic_override": override
    }

    uploaded_file = st.file_uploader("📁 Upload SQuAD-style JSON file", type=["json"])

    if uploaded_file:
        data = json.load(uploaded_file)
        st.success("✅ File loaded. Ready to evaluate.")

       # batch_size = st.number_input("🔢 Evaluate this many questions2:", min_value=1, max_value=20000, value=1000, step=100)

        
        flat_qas = []
        for topic in data["data"]:
            for para in topic["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    flat_qas.append((qa["question"], context, qa["answers"], qa.get("is_impossible", False)))

        total_rows = len(flat_qas)

        if "batch_size" not in st.session_state:
            st.session_state.batch_size = min(1000, total_rows)

        st.session_state.batch_size = st.number_input("🔢 Evaluate this many questions1:", min_value=1, max_value=total_rows, value=st.session_state.batch_size, step=100)
        batch_size = st.session_state.batch_size

        if "current_page" not in st.session_state:
            st.session_state.current_page = 0

        if total_rows > batch_size:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("⬅️ Previous Page") and st.session_state.current_page > 0:
                    st.session_state.current_page -= 1
            with col2:
                if st.button("Next Page ➡️") and (st.session_state.current_page + 1) * batch_size < total_rows:
                    st.session_state.current_page += 1

        start_index = st.session_state.current_page * batch_size
        st.markdown(f"📄 Showing questions **{start_index + 1} to {min(start_index + batch_size, total_rows)}** of {total_rows}")

        run_eval = st.button("🧪 Run Evaluation")

        if run_eval or "batch_results" not in st.session_state:
            # ... (no change to evaluation logic)
            model_path = settings["selected_model"]
            model, tokenizer, semantic_model = load_models(model_path)

            batch = flat_qas[start_index:start_index + batch_size]
            rows = []
            confusion_pairs = []

            for question, context, answers, is_impossible in batch:
                ground_truth = answers[0]["text"] if answers else ""

                answer, na_prob, conf, s_idx, e_idx = predict_answer(
                    model, tokenizer, question, context,
                    settings["no_answer_threshold"],
                    settings["max_span_length"]
                )

                final_answer, status = apply_override(
                    answer, question, na_prob, conf, settings, semantic_model, is_impossible
                )

                em = compute_em(final_answer, ground_truth)
                f1 = compute_f1(final_answer, ground_truth)
                match_type = "Exact" if em else ("Partial" if f1 > 0 else "No Match")
                if is_impossible and final_answer == "":
                    predicted_label = "No Answer"
                else:
                    predicted_label = "Answer"
                actual_label = "No Answer" if is_impossible else "Answer"
                confusion_pairs.append((actual_label, predicted_label))

                rows.append({
                    "Question": question,
                    "Ground Truth": ground_truth,
                    "Prediction": final_answer,
                    "Confidence Score": round(float(conf), 4),
                    "No-Answer Probability": round(na_prob, 4),
                    "Status": status,
                    "Exact Match": em,
                    "F1 Score": round(f1, 2),
                    "Match Type": match_type,
                    "Is Unanswerable?": is_impossible,
                    "Semantic Score": round(util.cos_sim(
                        semantic_model.encode(question, convert_to_tensor=True),
                        semantic_model.encode(final_answer, convert_to_tensor=True)).item(), 4) if final_answer else 0.0
                })

            st.session_state.batch_results = rows
            st.session_state.confusion_pairs = confusion_pairs


        if "batch_results" in st.session_state:
            rows = st.session_state.batch_results
            confusion_pairs = st.session_state.confusion_pairs
            df_full = pd.DataFrame(rows)

            match_filter = st.selectbox("🔍 Filter by Match Type", ["All"] + sorted(df_full["Match Type"].unique()))
            df = df_full.copy()
            if match_filter != "All":
                df = df[df["Match Type"] == match_filter]

            # Recalculate metrics for filtered df
            em_total = df["Exact Match"].sum()
            f1_total = df["F1 Score"].sum()
            unanswerable_correct = df[(df["Is Unanswerable?"] == True) & (df["Prediction"] == "")].shape[0]
            match_type_counter = Counter(df["Match Type"])

            st.dataframe(df, use_container_width=True)

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV Results", data=csv_data, file_name="evaluation_results.csv", mime="text/csv")
            # 📋 Summary Metrics
            st.markdown("---")
            st.markdown("### 📋 Evaluation Summary")
            total = len(df)
            st.markdown(f"• Total Samples Evaluated: **{total}**")
            st.markdown(f"• Exact Match Accuracy: **{em_total / total:.2%}**")
            st.markdown(f"• Average F1 Score: **{f1_total / total:.2f}**")
            st.markdown(f"• Unanswerable Accuracy: **{unanswerable_correct / total:.2%}**")
            for label, count in match_type_counter.items():
                st.markdown(f"• {label} Matches: **{count}**")

            # Confusion Matrix
            st.markdown("### 🧮 Confusion Matrix (Actual vs Predicted)")
            cm_df = pd.DataFrame(confusion_pairs, columns=["Actual", "Predicted"])
            cm = pd.crosstab(cm_df["Actual"], cm_df["Predicted"])
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)

            # 📊 Visual Insights
            st.markdown("---")
            st.markdown("### 📊 Visual Insights")

            st.markdown("#### 🔹 Semantic Score vs Confidence Score (colored by Match Type)")
            fig4, ax4 = plt.subplots()
            colors = df["Match Type"].map({"Exact": "green", "Partial": "orange", "No Match": "red"})
            ax4.scatter(df["Semantic Score"], df["Confidence Score"], alpha=0.6, c=colors)
            ax4.set_xlabel("Semantic Score")
            ax4.set_ylabel("Confidence Score")
            ax4.set_title("Semantic vs Confidence")
            st.pyplot(fig4)

            

            st.markdown("#### 🔹 Confidence Score Histogram")
            fig6, ax6 = plt.subplots()
            df["Confidence Score"].hist(bins=30, ax=ax6)
            ax6.set_title("Confidence Score Histogram")
            ax6.set_xlabel("Confidence Score")
            st.pyplot(fig6)

            st.markdown("#### 🔹 No-Answer Probability Histogram")
            fig7, ax7 = plt.subplots()
            df["No-Answer Probability"].hist(bins=30, ax=ax7)
            ax7.set_title("No-Answer Probability Histogram")
            ax7.set_xlabel("No-Answer Probability")
            st.pyplot(fig7)

            st.markdown("#### 🔹 No-Answer Probability vs Confidence Threshold (Line Chart)")
            fig8, ax8 = plt.subplots()
            sorted_df = df.sort_values("No-Answer Probability")
            ax8.plot(sorted_df["No-Answer Probability"], sorted_df["Confidence Score"], linestyle='-', marker='o', alpha=0.6)
            ax8.set_xlabel("No-Answer Probability")
            ax8.set_ylabel("Confidence Score")
            ax8.set_title("No-Answer Probability vs Confidence Threshold")
            st.pyplot(fig8)
           

            # 🔍 Threshold Sweep: F1 & EM vs Threshold
            st.markdown("### 🔍 F1 & Exact Match vs. No-Answer Threshold")
            thresholds = [round(x * 0.02, 2) for x in range(51)]
            f1_scores = []
            em_scores = []

            for t in thresholds:
                temp_em = temp_f1 = 0
                for row in rows:
                    is_qa_unanswerable = row["Is Unanswerable?"]
                    predicted = row["Prediction"]
                    gt = row["Ground Truth"]
                    if row["No-Answer Probability"] > t:
                        predicted = ""
                    temp_em += compute_em(predicted, gt)
                    temp_f1 += compute_f1(predicted, gt)
                f1_scores.append(100 * temp_f1 / total)
                em_scores.append(100 * temp_em / total)

            best_f1_idx = f1_scores.index(max(f1_scores))
            best_em_idx = em_scores.index(max(em_scores))

            fig_sweep, ax_sweep = plt.subplots()
            ax_sweep.plot(thresholds, f1_scores, label="F1 Score")
            ax_sweep.plot(thresholds, em_scores, label="Exact Match")
            ax_sweep.axvline(x=thresholds[best_f1_idx], color='blue', linestyle='--', label=f"Best F1 @ {thresholds[best_f1_idx]:.2f}")
            ax_sweep.axvline(x=thresholds[best_em_idx], color='orange', linestyle='--', label=f"Best EM @ {thresholds[best_em_idx]:.2f}")
            ax_sweep.set_title("F1 & Exact Match vs. No-Answer Threshold")
            ax_sweep.set_xlabel("No-Answer Probability Threshold")
            ax_sweep.set_ylabel("Score (%)")
            ax_sweep.legend()
            st.pyplot(fig_sweep)

            st.markdown(f"**Best F1 Score:** {f1_scores[best_f1_idx]:.2f}% at threshold `{thresholds[best_f1_idx]}`")
            st.markdown(f"**Best EM Score:** {em_scores[best_em_idx]:.2f}% at threshold `{thresholds[best_em_idx]}`")
