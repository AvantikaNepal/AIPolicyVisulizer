import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import re

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="Policy Insight Summarizer", layout="wide")

st.title("📘 AI Policy Document Analyzer")
st.write("Upload your policy document to summarize it, extract insights, and visualize key terms.")


# -------------------------------
# TEXT CLEANING FUNCTION
# -------------------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9.,;:?!()\-\n ]', '', text)  # remove symbols
    return text.strip()


# -------------------------------
# LOAD SMALL SUMMARIZATION MODEL
# -------------------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")


# -------------------------------
# LOAD LOCAL POLICY INSIGHT MODEL
# -------------------------------
@st.cache_resource
def load_policy_insight_model():
    return pipeline(
        "text2text-generation",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn"
    )


# -------------------------------
# FUNCTION: EXTRACT POLICY INSIGHTS
# -------------------------------
def extract_policy_insights(text):
    model = load_policy_insight_model()
    prompt = (
        "Analyze the following policy text and extract structured information:\n\n"
        f"{text}\n\n"
        "Output in JSON format with these fields:\n"
        "{\n"
        "  'Policy Objectives': [...],\n"
        "  'Recommendations/Actions': [...],\n"
        "  'Stakeholders': [...]\n"
        "}"
    )

    response = model(prompt, max_length=512, do_sample=False)
    return response[0]['generated_text']


# -------------------------------
# UPLOAD DOCUMENT
# -------------------------------
uploaded_file = st.file_uploader("📄 Upload a text or policy document", type=["txt", "pdf"])

if uploaded_file:
    # Read text
    if uploaded_file.type == "application/pdf":
        import fitz  # PyMuPDF
        pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = " ".join(page.get_text() for page in pdf_doc)
    else:
        text = uploaded_file.read().decode("utf-8")

    cleaned_text = clean_text(text)

    # Display document preview
    st.subheader("📑 Document Preview")
    st.text_area("Preview:", cleaned_text[:1500] + "...", height=200)

    # -------------------------------
    # SUMMARY GENERATION
    # -------------------------------
    st.subheader("🧾 Document Summary")
    summarizer = load_summarizer()
    with st.spinner("Summarizing document..."):
        summary = summarizer(cleaned_text[:3000], max_length=200, min_length=60, do_sample=False)[0]['summary_text']
    st.write(summary)

    # -------------------------------
    # POLICY INSIGHTS EXTRACTION (LLM)
    # -------------------------------
    st.subheader("🤖 Policy Insights (Extracted by Local LLM)")
    with st.spinner("Extracting structured policy insights..."):
        insights = extract_policy_insights(cleaned_text[:2000])
    st.code(insights, language="json")

    # -------------------------------
    # TF-IDF KEYWORD ANALYSIS
    # -------------------------------
    st.subheader("📊 Top Keywords (TF-IDF)")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
    tfidf_matrix = vectorizer.fit_transform([cleaned_text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    tfidf_dict = dict(zip(feature_names, scores))
    top_keywords_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    df_tfidf = pd.DataFrame(top_keywords_tfidf, columns=["Keyword", "TF-IDF"]).set_index("Keyword")

    # Display clearer bar chart
    st.bar_chart(df_tfidf)

    # Optional: Show data table
    with st.expander("View TF-IDF Scores"):
        st.dataframe(df_tfidf.style.highlight_max(axis=0, color="lightblue"))

else:
    st.info("Please upload a policy document to begin.")
