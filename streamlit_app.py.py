"""
Customer Complaint Severity Classifier — Streamlit App
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Complaint Severity Classifier",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid #2a2a4a;
    }
    [data-testid="stSidebar"] * { color: #e0e0ff !important; }

    /* Main background */
    .main { background: #f7f7fc; }

    /* Priority cards */
    .priority-card {
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.10);
    }
    .high-card   { background: linear-gradient(135deg,#ff4b4b,#ff8c42); color:#fff; }
    .medium-card { background: linear-gradient(135deg,#f7b731,#f0932b); color:#fff; }
    .low-card    { background: linear-gradient(135deg,#20bf6b,#0fb9b1); color:#fff; }

    /* Input area */
    .stTextArea textarea {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
        border-radius: 12px !important;
        border: 2px solid #d0d0ee !important;
        background: #fff !important;
        padding: 14px !important;
    }
    .stTextArea textarea:focus {
        border-color: #6c63ff !important;
        box-shadow: 0 0 0 3px rgba(108,99,255,0.15) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6c63ff, #48cfad);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 40px;
        font-size: 16px;
        font-weight: 700;
        font-family: 'DM Sans', sans-serif;
        letter-spacing: 0.5px;
        transition: transform 0.15s, box-shadow 0.15s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(108,99,255,0.35);
    }

    /* Section headings */
    h1, h2, h3 { font-family: 'DM Sans', sans-serif; font-weight: 800; }
    h1 { color: #1a1a2e; font-size: 2.2rem; }
    h2 { color: #2a2a4a; font-size: 1.5rem; }
    h3 { color: #444; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #fff;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }

    /* Divider */
    hr { border: none; border-top: 2px solid #e8e8f8; margin: 24px 0; }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        letter-spacing: 1px;
    }
    .badge-high   { background:#ffe0e0; color:#cc0000; }
    .badge-medium { background:#fff3cd; color:#8a5c00; }
    .badge-low    { background:#d4edda; color:#155724; }
</style>
""", unsafe_allow_html=True)


# ── Preprocessing (mirrors your pipeline) ────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet',   quiet=True)
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    USE_SPACY = True
except Exception:
    USE_SPACY = False

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in STOPWORDS])

def lemmatize(text):
    if USE_SPACY:
        doc = nlp(text)
        return " ".join([t.lemma_ for t in doc if not t.is_space])
    return text

def preprocess(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text


# ── Model loading / fallback demo model ──────────────────────────────────────
@st.cache_resource
def load_model():
    """Try to load saved model; fall back to a trained demo model."""
    # Try loading your saved pipeline
    for path in ["model_pipeline.pkl", "tfidf_pipeline.pkl", "saved_model.pkl"]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f), "loaded"

    # ── Fallback: train a quick demo model ───────────────────────────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    demo_texts = [
        # High
        "My account has been hacked and money was stolen unauthorized transaction",
        "Fraudulent charge on my card immediately need help this is an emergency",
        "I will sue you data breach personal information stolen fix this now",
        "Unauthorized payment fraud alert my card was compromised",
        "Emergency account locked cannot access funds fraudulent activity",
        "Someone stole my account fix immediately legal action",
        # Medium
        "My order has not arrived after two weeks please help",
        "Subscription renewal failed and I cannot access the service",
        "The app keeps crashing every time I try to log in",
        "Delivery was delayed by a week and no one contacted me",
        "My refund has not been processed after 10 days",
        "Cannot reset my password the link does not work",
        # Low
        "The app design could be improved it looks a bit outdated",
        "Would be nice to have a dark mode option",
        "Just a suggestion the checkout flow could be simplified",
        "The font size on mobile is slightly small",
        "Nice product but packaging could be more eco friendly",
        "Feedback the onboarding tutorial was a bit long",
    ]
    demo_labels = [2,2,2,2,2,2, 1,1,1,1,1,1, 0,0,0,0,0,0]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1,2), sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=500, class_weight="balanced")),
    ])
    pipe.fit([preprocess(t) for t in demo_texts], demo_labels)
    return pipe, "demo"

model, model_status = load_model()


# ── Prediction helper ─────────────────────────────────────────────────────────
LABEL_MAP   = {0: "Low", 1: "Medium", 2: "High"}
EMOJI_MAP   = {0: "🟢", 1: "🟡", 2: "🔴"}
COLOR_MAP   = {0: "#20bf6b", 1: "#f7b731", 2: "#ff4b4b"}
CARD_CLASS  = {0: "low-card", 1: "medium-card", 2: "high-card"}
BADGE_CLASS = {0: "badge-low", 1: "badge-medium", 2: "badge-high"}

def predict(text):
    cleaned = preprocess(text)
    proba   = model.predict_proba([cleaned])[0]
    label   = int(np.argmax(proba))
    return label, proba, cleaned


# ── Confidence bar chart ──────────────────────────────────────────────────────
def confidence_chart(proba):
    fig, ax = plt.subplots(figsize=(5, 2.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    classes = ["Low", "Medium", "High"]
    colors  = ["#20bf6b", "#f7b731", "#ff4b4b"]
    bars    = ax.barh(classes, proba * 100, color=colors,
                      height=0.5, edgecolor="none")

    for bar, val in zip(bars, proba * 100):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", ha="left",
                fontsize=11, fontweight="bold", color="#333")

    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", fontsize=10, color="#555")
    ax.tick_params(colors="#333", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ddd")
    ax.spines["bottom"].set_color("#ddd")
    ax.xaxis.label.set_color("#555")
    fig.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚨 Complaint Classifier")
    st.markdown("---")
    st.markdown("**Priority Levels**")
    st.markdown("🔴 **High** — Fraud, emergencies, data breach, unauthorized access")
    st.markdown("🟡 **Medium** — Delivery issues, app errors, refund delays")
    st.markdown("🟢 **Low** — Suggestions, minor feedback, cosmetic issues")
    st.markdown("---")
    st.markdown("**Model Info**")
    if model_status == "demo":
        st.warning("⚠️ Running demo model\nReplace `model_pipeline.pkl` with your trained model.")
    else:
        st.success("✅ Production model loaded")
    st.markdown("---")

    # Batch upload section
    st.markdown("**Batch Classification**")
    uploaded_file = st.file_uploader("Upload a CSV with a `message` column", type=["csv"])

    st.markdown("---")
    st.markdown("**Quick Test Samples**")
    sample_complaints = {
        "🔴 High — Fraud":     "There was an unauthorized transaction on my account I need help immediately this is fraud",
        "🟡 Medium — Delivery": "My order has not arrived after 10 days and no one is responding to my emails",
        "🟢 Low — Suggestion":  "The app interface looks a bit dated would be nice to have a dark mode",
    }
    for label, text in sample_complaints.items():
        if st.button(label, key=label):
            st.session_state["complaint_input"] = text


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("# Customer Complaint Severity Classifier")
st.markdown("Paste a complaint message below to instantly classify it as **High**, **Medium**, or **Low** priority.")
st.markdown("---")

col_input, col_result = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown("### ✍️ Enter Complaint")
    default_text = st.session_state.get("complaint_input", "")
    complaint = st.text_area(
        label="",
        value=default_text,
        height=200,
        placeholder="Type or paste a customer complaint here…\n\nExample: 'My card was charged twice and I see an unauthorized transaction. Please fix this immediately!'",
        key="complaint_input",
    )

    char_count = len(complaint.strip())
    word_count = len(complaint.strip().split()) if complaint.strip() else 0
    st.caption(f"📝 {word_count} words · {char_count} characters")

    classify_btn = st.button("🔍  Classify Complaint", use_container_width=True)

with col_result:
    st.markdown("### 📊 Classification Result")

    if classify_btn and complaint.strip():
        label_idx, proba, cleaned = predict(complaint)
        label_name = LABEL_MAP[label_idx]

        # Priority card
        st.markdown(
            f'<div class="priority-card {CARD_CLASS[label_idx]}">'
            f'{EMOJI_MAP[label_idx]}  {label_name.upper()} PRIORITY'
            f'</div>',
            unsafe_allow_html=True
        )

        # Confidence score
        conf = proba[label_idx] * 100
        st.metric("Confidence", f"{conf:.1f}%")

        # Confidence breakdown chart
        st.markdown("**Confidence Breakdown**")
        fig = confidence_chart(proba)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Preprocessed text expander
        with st.expander("🔍 See preprocessed text"):
            st.code(cleaned, language=None)

        # Action recommendation
        st.markdown("---")
        st.markdown("**Recommended Action**")
        if label_idx == 2:
            st.error("🚨 **Immediate escalation required.** Assign to senior support agent within 15 minutes.")
        elif label_idx == 1:
            st.warning("⚡ **Standard escalation.** Assign to support queue. Target response: 24 hours.")
        else:
            st.success("✅ **Routine ticket.** Add to general queue. Target response: 72 hours.")

    elif classify_btn and not complaint.strip():
        st.info("Please enter a complaint message first.")
    else:
        st.markdown(
            '<div style="background:#f0f0fa;border-radius:12px;padding:32px;text-align:center;color:#888;">'
            '<p style="font-size:48px;margin:0">🎯</p>'
            '<p style="font-size:15px;margin-top:8px">Results will appear here after classification</p>'
            '</div>',
            unsafe_allow_html=True
        )


# ── Batch mode ────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    st.markdown("---")
    st.markdown("## 📂 Batch Classification Results")

    try:
        batch_df = pd.read_csv(uploaded_file)

        if "message" not in batch_df.columns:
            st.error("CSV must have a column named `message`.")
        else:
            with st.spinner("Classifying all complaints…"):
                batch_df["clean_message"] = batch_df["message"].apply(preprocess)
                probas  = model.predict_proba(batch_df["clean_message"].tolist())
                labels  = np.argmax(probas, axis=1)
                batch_df["predicted_priority"] = [LABEL_MAP[l] for l in labels]
                batch_df["confidence"]          = [f"{probas[i,l]*100:.1f}%" for i,l in enumerate(labels)]

            # Summary stats
            c1, c2, c3, c4 = st.columns(4)
            counts = batch_df["predicted_priority"].value_counts()
            c1.metric("Total Complaints", len(batch_df))
            c2.metric("🔴 High Priority",   counts.get("High",   0))
            c3.metric("🟡 Medium Priority", counts.get("Medium", 0))
            c4.metric("🟢 Low Priority",    counts.get("Low",    0))

            # Distribution pie
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            wedge_colors = ["#ff4b4b","#f7b731","#20bf6b"]
            pie_labels   = ["High","Medium","Low"]
            pie_vals     = [counts.get(l,0) for l in pie_labels]
            ax2.pie(pie_vals, labels=pie_labels, colors=wedge_colors,
                    autopct="%1.1f%%", startangle=90,
                    textprops={"fontsize":12, "fontweight":"bold"})
            ax2.set_title("Priority Distribution", fontsize=14, fontweight="bold")
            st.pyplot(fig2)
            plt.close(fig2)

            # Table
            st.dataframe(
                batch_df[["message","predicted_priority","confidence"]],
                use_container_width=True,
                height=320,
            )

            # Download button
            csv_out = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️  Download Results CSV",
                data=csv_out,
                file_name="classified_complaints.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#aaa;font-size:13px;">'
    'Customer Complaint Severity Classifier · Built with Streamlit · '
    'Models: Logistic Regression · SVM · XGBoost · LSTM'
    '</p>',
    unsafe_allow_html=True
)



