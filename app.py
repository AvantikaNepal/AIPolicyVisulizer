import streamlit as st
import fitz
import re
import textstat
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import date

st.set_page_config(page_title="AI Policy Visualizer", layout="wide")

STOPWORDS = set("""a an the and or but if while is are was were be been being to of in on for
with as by at from this that these those it its it's not no nor so than then there here
which who whom whose what when where why how all any both each few more most other some such
only own same can will just don should now shall must may might""".split())

EU_AVG_FRE = 40  # rough real-world benchmark: typical EU/US legislative text scores ~30-45

# Term -> (category, weight). Weight reflects legal "bindingness":
# 3 = hard binding language, 2 = enforcement/consequence, 1 = soft/discretionary
OBLIGATION_TERMS = {
    "shall": ("obligation", 3),
    "shall not": ("prohibition", 3),
    "must": ("obligation", 3),
    "must not": ("prohibition", 3),
    "is required to": ("obligation", 3),
    "are required to": ("obligation", 3),
    "requires": ("obligation", 3),
    "required": ("obligation", 3),
    "requirement": ("obligation", 2),
    "obligation": ("obligation", 3),
    "obligations": ("obligation", 3),
    "obliged": ("obligation", 3),
    "duty to": ("obligation", 3),
    "responsible for": ("obligation", 2),
    "responsibility": ("obligation", 2),
    "ensure that": ("obligation", 2),
    "mandatory": ("obligation", 3),
    "in accordance with": ("obligation", 1),
    "compliance": ("obligation", 2),
    "non-compliance": ("enforcement", 2),
    "prohibited": ("prohibition", 3),
    "is prohibited": ("prohibition", 3),
    "forbidden": ("prohibition", 3),
    "not permitted": ("prohibition", 3),
    "penalty": ("enforcement", 2),
    "penalties": ("enforcement", 2),
    "fine": ("enforcement", 2),
    "fines": ("enforcement", 2),
    "sanction": ("enforcement", 2),
    "sanctions": ("enforcement", 2),
    "enforcement": ("enforcement", 2),
    "liable": ("enforcement", 2),
    "liability": ("enforcement", 2),
    "may": ("discretion", 1),
    "should": ("recommendation", 1),
    "recommended": ("recommendation", 1),
    "encouraged to": ("recommendation", 1),
    "responsible authority": ("obligation", 1),
}

# ---------------- Core functions ----------------

def extract_text(uploaded_file):
    """Read text from a PDF or TXT upload."""
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif uploaded_file.type == "application/pdf":
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        except Exception as e:
            st.error(f"Couldn't read '{uploaded_file.name}': {e}")
            return None
        text = "".join(page.get_text() for page in doc)
        if not text.strip():
            st.error(f"'{uploaded_file.name}' has no extractable text (likely a scanned image needing OCR).")
            return None
        return text
    st.error(f"Unsupported file type: {uploaded_file.name}")
    return None


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def word_frequencies(text, top_n=15):
    """Simple, explainable word-frequency keyword extraction for a single document."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    counts = Counter(words)
    return dict(counts.most_common(top_n))


def scan_obligations(text):
    """Count occurrences of obligation/prohibition/enforcement language, with a bindingness weight per term."""
    lower = text.lower()
    rows = []
    for term, (category, weight) in OBLIGATION_TERMS.items():
        n = len(re.findall(r"\b" + re.escape(term) + r"\b", lower))
        if n:
            rows.append({"Term": term, "Category": category, "Count": n, "Weight": weight, "Weighted Score": n * weight})
    cols = ["Term", "Category", "Count", "Weight", "Weighted Score"]
    df = pd.DataFrame(rows, columns=cols).sort_values("Weighted Score", ascending=False) if rows else pd.DataFrame(columns=cols)
    return df


def obligation_density(text):
    """Weighted obligation-language density per 1,000 words — a rough 'bindingness score'."""
    ob = scan_obligations(text)
    weighted_total = ob["Weighted Score"].sum() if not ob.empty else 0
    words = textstat.lexicon_count(text) or 1
    return round(weighted_total / words * 1000, 2)


def readability_summary(text):
    return {
        "Flesch Reading Ease": round(textstat.flesch_reading_ease(text), 1),
        "Grade Level (Flesch-Kincaid)": round(textstat.flesch_kincaid_grade(text), 1),
        "Words": textstat.lexicon_count(text),
        "Sentences": textstat.sentence_count(text),
    }


def plot_bar(freq_dict, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(freq_dict.keys()), list(freq_dict.values()), color="#3E7CB1")
    ax.set_title(title)
    ax.set_xticklabels(list(freq_dict.keys()), rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_wordcloud(freq_dict):
    wc = WordCloud(width=800, height=350, background_color="white").generate_from_frequencies(freq_dict)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def build_pdf_brief(doc_name, freq_dict, obligations_df, readability):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Policy Analysis Brief", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Document: {doc_name}", ln=True)
    pdf.cell(0, 8, f"Generated: {date.today().isoformat()}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Readability", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for k, v in readability.items():
        pdf.cell(0, 7, f"  {k}: {v}", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Top Keywords", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for w, c in list(freq_dict.items())[:10]:
        pdf.cell(0, 7, f"  {w}: {c}", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Obligation / Risk Language", ln=True)
    pdf.set_font("Helvetica", "", 10)
    if obligations_df.empty:
        pdf.cell(0, 7, "  None detected.", ln=True)
    else:
        for _, row in obligations_df.iterrows():
            pdf.cell(
                0, 7,
                f"  {row['Term']} ({row['Category']}, weight {row['Weight']}): {row['Count']} occurrences",
                ln=True,
            )

    return bytes(pdf.output(dest="S"))


# ---------------- UI ----------------

st.title("AI Policy Visualizer")
st.caption("Upload one or more policy documents to compare keyword salience, obligation language, and readability.")

uploaded_files = st.file_uploader(
    "Upload policy document(s) (.pdf or .txt)", type=["pdf", "txt"], accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload at least one document to begin.")
    st.stop()

docs = {}
for f in uploaded_files:
    raw = extract_text(f)
    if raw:
        docs[f.name] = clean_text(raw)

if not docs:
    st.stop()

tab_names = list(docs.keys()) + (["Compare"] if len(docs) > 1 else [])
tabs = st.tabs(tab_names)

for i, name in enumerate(docs.keys()):
    with tabs[i]:
        text = docs[name]
        freq = word_frequencies(text)
        obligations = scan_obligations(text)
        readability = readability_summary(text)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Keywords")
            plot_bar(freq, name)
        with col2:
            st.subheader("Word Cloud")
            plot_wordcloud(freq)

        st.subheader("Obligation & Risk Language")
        if obligations.empty:
            st.write("No obligation/prohibition/enforcement terms detected.")
        else:
            st.dataframe(obligations, use_container_width=True, hide_index=True)

        st.subheader("Readability")
        st.dataframe(pd.DataFrame([readability]), use_container_width=True, hide_index=True)
        fre = readability["Flesch Reading Ease"]
        vs_benchmark = "harder to read than" if fre < EU_AVG_FRE else "about as easy to read as, or easier than"
        st.caption(
            f"For reference, typical EU/US legislative text scores roughly {EU_AVG_FRE} on Flesch Reading Ease. "
            f"This document ({fre}) is {vs_benchmark} typical regulatory text."
        )

        pdf_bytes = build_pdf_brief(name, freq, obligations, readability)
        st.download_button(
            f"Download Policy Brief (PDF) — {name}",
            data=pdf_bytes,
            file_name=f"{name.rsplit('.', 1)[0]}_brief.pdf",
            mime="application/pdf",
        )

if len(docs) > 1:
    with tabs[-1]:
        st.subheader("Cross-Document Comparison")

        names = list(docs.keys())
        corpus = [docs[n] for n in names]

        # TF-IDF is meaningful here: a real multi-document corpus
        vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS), max_features=25)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        terms = vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=terms, index=names)

        st.write("Distinctive terms per document (TF-IDF — higher means more distinctive to that document):")
        top_per_doc = {n: tfidf_df.loc[n].sort_values(ascending=False).head(8) for n in names}
        st.dataframe(pd.DataFrame(top_per_doc), use_container_width=True)

        st.write("Obligation-language density (weighted score per 1,000 words — higher = more legally binding tone):")
        density_map = {n: obligation_density(docs[n]) for n in names}
        density_df = pd.DataFrame(
            [{"Document": n, "Bindingness score / 1,000 words": v} for n, v in density_map.items()]
        )
        st.dataframe(density_df, use_container_width=True, hide_index=True)

        st.write("Readability comparison:")
        readability_map = {n: readability_summary(docs[n]) for n in names}
        read_rows = [{"Document": n, **readability_map[n]} for n in names]
        st.dataframe(pd.DataFrame(read_rows), use_container_width=True, hide_index=True)

        # ---- Auto-generated interpretive summary (rule-based, not AI) ----
        st.subheader("Summary")
        most_binding = max(density_map, key=density_map.get)
        least_binding = min(density_map, key=density_map.get)
        hardest_read = min(readability_map, key=lambda n: readability_map[n]["Flesch Reading Ease"])
        easiest_read = max(readability_map, key=lambda n: readability_map[n]["Flesch Reading Ease"])

        insights = []
        if most_binding != least_binding and density_map[least_binding] > 0:
            ratio = round(density_map[most_binding] / max(density_map[least_binding], 0.01), 1)
            insights.append(
                f"**{most_binding}** shows the highest concentration of binding language "
                f"(shall/must/prohibited/penalty-type terms) — roughly **{ratio}x** the density of "
                f"**{least_binding}**, suggesting a more strictly regulatory posture."
            )
        elif density_map[most_binding] == 0:
            insights.append("None of the uploaded documents contain notable obligation/prohibition language.")

        if hardest_read != easiest_read:
            insights.append(
                f"**{hardest_read}** is the hardest to read (Flesch Reading Ease "
                f"{readability_map[hardest_read]['Flesch Reading Ease']}), while **{easiest_read}** is the "
                f"most accessible ({readability_map[easiest_read]['Flesch Reading Ease']})."
            )

        for line in insights:
            st.markdown(f"- {line}")
        st.caption(
            "These summaries are generated from rule-based term counts and standard readability formulas, "
            "not an AI model — figures are reproducible and auditable."
        )