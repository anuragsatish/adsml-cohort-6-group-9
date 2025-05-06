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
import numpy as np
import plotly.express as px


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
def extend_span_right_if_conjunction(answer, context, end_idx, tokenizer, max_extend=10):
    """
    Extend the answer span if it ends before a conjunction (like 'and') and is missing part of a compound entity.
    """
    words = context.split()
    if end_idx + 1 < len(words) and words[end_idx] == "and":
        extended = answer + " and"
        for i in range(1, max_extend):
            if end_idx + i + 1 < len(words):
                extended += " " + words[end_idx + i + 1]
            else:
                break
        return extended
    return answer


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

    print("predict_answer called")
   
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

    # Heuristic: Extend if span ends with "and ..."
    predicted_answer = extend_span_right_if_conjunction(
        predicted_answer, context, best_end, tokenizer
    )

    null_score = (start_logits[0] + end_logits[0]).item()
    logit_diff = null_score - best_score
    na_prob = F.sigmoid(torch.tensor(logit_diff)).item()

    debug = {
        "cls_score": round(null_score, 4),
        "start_logit": round(start_logits[best_start].item(), 4),
        "end_logit": round(end_logits[best_end].item(), 4)
    }
    print("DEBUG:", debug)
    return predicted_answer, na_prob, best_score.item(), start_char, end_char,debug


# Updated debug info renderer

def render_debug_metrics(row, settings):
    st.markdown("---")
    with st.expander("🔬 Debug Info", expanded=False):
        debug_info = {
            "Predicted Answer": row.get("Prediction", ""),
            "Raw Confidence Score": row.get("Confidence Score", "NA"),
            "No-Answer Probability": row.get("No-Answer Probability", "NA"),
            "Semantic Similarity Score": row.get("Semantic Score", "NA"),
            "Suppression Reason": row.get("Status", ""),
            "CLS Score": row.get("CLS Score", "NA"),
            "Start Logit": row.get("Start Logit", "NA"),
            "End Logit": row.get("End Logit", "NA"),
            "Applied Settings": {
                "Selected Model": settings.get("selected_model"),
                "No-Answer Threshold": settings.get("no_answer_threshold"),
                "Confidence Threshold": settings.get("confidence_threshold"),
                "Max Span Length": settings.get("max_span_length"),
                "Semantic Similarity Threshold": settings.get("semantic_similarity_threshold"),
                "Semantic Override Enabled": settings.get("enable_semantic_override"),
                "Top-K Retrieved": settings.get("top_k", "N/A")
            }
        }
        #st.json(debug_info)


# Post-processing override logic
def apply_override(answer, question, na_prob, confidence, settings, semantic_model, is_impossible):
    if settings.get("force_squad2_flag", False) and is_impossible:
        return "", "No Answer (Forced by SQuAD2 Flag)"

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
# Visualization: No-Answer Probability vs Span Confidence

def plot_no_answer_probability_curve_with_scatter(df, na_threshold=0.5, conf_threshold=1.0, cls_logit=5.0):
    span_logits = np.linspace(0, 10, 100)

    def compute_p_na(cls_logit, span_logit):
        max_logit = max(cls_logit, span_logit)
        exp_cls = np.exp(cls_logit - max_logit)
        exp_span = np.exp(span_logit - max_logit)
        return exp_cls / (exp_cls + exp_span)

    no_answer_probs = [compute_p_na(cls_logit, logit) for logit in span_logits]

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(span_logits, no_answer_probs, color='orange', linewidth=2, label="No-Answer Probability (Theoretical)")
    ax.axhline(y=na_threshold, color='red', linestyle='--', label=f"No-Answer Threshold ({na_threshold})")
    ax.axvline(x=conf_threshold, color='blue', linestyle='--', label=f"Confidence Threshold ({conf_threshold})")

    match_colors = {
        "Exact": "green",
        "Partial": "blue",
        "No Match": "red"
    }

    for match_type in df["Match Type"].unique():
        subset = df[df["Match Type"] == match_type]
        ax.scatter(
            subset["Confidence Score"],
            subset["No-Answer Probability"],
            label=f"{match_type} (Actual)",
            alpha=0.6,
            s=40,
            c=match_colors.get(match_type, "gray"),
            edgecolors='k'
        )

    ax.set_xlabel("Predicted Answer Span Confidence (Logit Score)")
    ax.set_ylabel("No-Answer Probability")
    ax.set_title("No-Answer Probability vs Confidence (with Actual Predictions)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
# 📊 Visualize Answer Length by Match Type
def plot_answer_length_vs_match_type(df):
    df = df.copy()
    df["Answer Length"] = df["Prediction"].apply(lambda x: len(x.split()) if x else 0)

    fig = px.box(
        df,
        x="Match Type",
        y="Answer Length",
        color="Match Type",
        title="📏 Answer Length vs Match Type",
        points="all",  # show individual points too
        hover_data=["Question", "Prediction"]
    )
    fig.update_layout(
        xaxis_title="Match Type",
        yaxis_title="Answer Length (words)",
        boxmode='group',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


# 📊 Interactive Semantic vs Confidence Plot

def plot_semantic_vs_confidence_interactive(df):
    fig = px.scatter(
        df,
        x="Semantic Score",
        y="Confidence Score",
        color="Match Type",
        symbol="Match Type",
        hover_data=["Question", "Prediction"],
        title="Semantic Score vs Confidence Score (Interactive)",
        color_discrete_map={
            "Exact": "green",
            "Partial": "blue",
            "No Match": "red"
        }
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# 📌 Outlier Finder (F1 < 0.3 and Semantic > 0.7)
def plot_outlier_cases(df):
    outliers = df[(df["F1 Score"] < 0.3) & (df["Semantic Score"] > 0.7)]
    if outliers.empty:
        st.info("No outlier cases found with F1 < 0.3 and Semantic Score > 0.7")
    else:
        fig = px.scatter(
            outliers,
            x="Semantic Score",
            y="F1 Score",
            color="Match Type",
            hover_data=["Question", "Prediction"],
            title="🔎 Outlier Finder: High Semantic, Low F1",
            color_discrete_map={
                "Exact": "green",
                "Partial": "blue",
                "No Match": "red"
            }
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(outliers[["Question", "Prediction", "Ground Truth", "F1 Score", "Semantic Score", "Match Type"]])




# Tab render

def render():
    #st.subheader("📊 Batch Evaluation")
        # Local Settings Panel (Replaces dependency on get_current_settings)
    with st.expander("⚙️ Evaluation Settings", expanded=False):
        selected_model = st.selectbox(
            "Select QA Model",
            ["qa_model_checkpoint", "models/bert_squad2_deepset", "qa_model_checkpoint_sk", "qa_model_checkpoint_170425","qa_model_checkpoint_512_290425"],
            index=0
        )
        hybrid_mode = st.checkbox("Use Hybrid No-Answer Mode", value=False)
        force_squad2_flag = st.checkbox("Force No Answer if Labeled as Unanswerable", value=False)
        no_ans_thresh = st.selectbox("No-Answer Threshold", [0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1.0], index=4)
        conf_thresh = st.selectbox("Confidence Suppression Threshold", [-2.0, -1.0, 0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8,0,9.0,10.0], index=1)
        span_len = st.selectbox("Max Answer Span Length", [15, 30, 50, 80], index=1)
        sim_thresh = st.selectbox("Semantic Similarity Threshold", [0.3, 0.4, 0.6, 0.8], index=1)
        override = st.checkbox("Enable Semantic Override Logic", value=True)

    settings = {
        "selected_model": selected_model,
        "use_squad2_for_no_answer_only": hybrid_mode,
        "force_squad2_flag": force_squad2_flag,
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

                answer, na_prob, conf, s_idx, e_idx, debug = predict_answer(
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
                        semantic_model.encode(final_answer, convert_to_tensor=True)).item(), 4) if final_answer else 0.0,
                    "CLS Score": debug["cls_score"],
                    "Start Logit": debug["start_logit"],
                    "End Logit": debug["end_logit"]
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
            with st.expander("📋 Evaluation Summary", expanded=False):
                total = len(df)
                st.markdown(f"• Total Samples Evaluated: **{total}**")
                st.markdown(f"• Exact Match Accuracy: **{em_total / total:.2%}**")
                st.markdown(f"• Average F1 Score: **{f1_total / total:.2f}**")
                st.markdown(f"• Unanswerable Accuracy: **{unanswerable_correct / total:.2%}**")
                for label, count in match_type_counter.items():
                    st.markdown(f"• {label} Matches: **{count}**")
            # Confusion Matrix
            with st.expander("🧮 Confusion Matrix (Actual vs Predicted)", expanded=False):
                cm_df = pd.DataFrame(confusion_pairs, columns=["Actual", "Predicted"])
                cm = pd.crosstab(cm_df["Actual"], cm_df["Predicted"])
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                ax_cm.set_title("Confusion Matrix")
                st.pyplot(fig_cm)


            # 📊 Visual Insights
            st.markdown("---")
            st.markdown("### 📊 Visual Insights")

            # 📊 Interactive Semantic vs Confidence Plot
            with st.expander("🧠 Semantic Score vs Confidence Score (Interactive)", expanded=False):
                plot_semantic_vs_confidence_interactive(df)

            with st.expander("🔹 Confidence Score Histogram", expanded=False):
                fig6, ax6 = plt.subplots()
                df["Confidence Score"].hist(bins=30, ax=ax6)
                ax6.set_title("Confidence Score Histogram")
                ax6.set_xlabel("Confidence Score")
                st.pyplot(fig6)

            with st.expander("📈 No-Answer Probability vs Confidence (Overlayed with Predictions)", expanded=False):
                plot_no_answer_probability_curve_with_scatter(
                    df,
                    na_threshold=settings["no_answer_threshold"],
                    conf_threshold=settings["confidence_threshold"]
                )
            with st.expander("📏 Answer Length vs Match Type", expanded=False):
                plot_answer_length_vs_match_type(df)


            with st.expander("🔍 Outlier Finder: High Semantic but Low F1", expanded=False):
                 plot_outlier_cases(df)

            if not df.empty:
                render_debug_metrics(df.iloc[0], settings)





            with st.expander("🔹 No-Answer Probability Histogram", expanded=False):
                fig7, ax7 = plt.subplots()
                df["No-Answer Probability"].hist(bins=30, ax=ax7)
                ax7.set_title("No-Answer Probability Histogram")
                ax7.set_xlabel("No-Answer Probability")
                st.pyplot(fig7)
           
        

            # 🔍 Threshold Sweep: F1 & EM vs Threshold
            with st.expander("🔍", expanded=False):
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


            st.markdown("### 🧪 Inspect Debug Info for a Specific Sample")

            # Create a unique label for each row
            df["Label"] = df.apply(lambda x: f"{x['Match Type']} | {x['Question'][:60]}...", axis=1)

            selected_label = st.selectbox("Select a Question:", df["Label"].tolist())

            # Match selected label to row
            selected_row = df[df["Label"] == selected_label].iloc[0]

            # Show debug info for the selected row
            render_debug_metrics(selected_row, settings)


     
    
