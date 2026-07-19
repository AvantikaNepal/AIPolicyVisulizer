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

OBLIGATION_TERMS = {
    "shall": "obligation",
    "must": "obligation",
    "shall not": "prohibition",
    "must not": "prohibition",
    "prohibited": "prohibition",
    "is prohibited": "prohibition",
    "may": "discretion",
    "should": "recommendation",
    "penalty": "enforcement",
    "penalties": "enforcement",
    "fine": "enforcement",
    "non-compliance": "enforcement",
    "enforcement": "enforcement",
    "obligation": "obligation",
    "responsible for": "obligation",
    "liable": "enforcement",
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
    """Count occurrences of obligation/prohibition/enforcement language."""
    lower = text.lower()
    rows = []
    for term, category in OBLIGATION_TERMS.items():
        n = len(re.findall(r"\b" + re.escape(term) + r"\b", lower))
        if n:
            rows.append({"Term": term, "Category": category, "Count": n})
    df = pd.DataFrame(rows).sort_values("Count", ascending=False) if rows else pd.DataFrame(columns=["Term", "Category", "Count"])
    return df


def readability_summary(text):
    return {
        "Flesch Reading Ease": round(textstat.flesch_reading_ease(text), 1),
        "Grade Level (Flesch-Kincaid)": round(textstat.flesch_kincaid_grade(text), 1),
        "Words": textstat.lexical_count(text),
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
            pdf.cell(0, 7, f"  {row['Term']} ({row['Category']}): {row['Count']}", ln=True)

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

        st.write("Obligation-language density (occurrences per 1,000 words):")
        density_rows = []
        for n in names:
            ob = scan_obligations(docs[n])
            total = ob["Count"].sum() if not ob.empty else 0
            words = textstat.lexical_count(docs[n])
            density_rows.append({"Document": n, "Obligation terms / 1,000 words": round(total / words * 1000, 2)})
        st.dataframe(pd.DataFrame(density_rows), use_container_width=True, hide_index=True)

        st.write("Readability comparison:")
        read_rows = [{"Document": n, **readability_summary(docs[n])} for n in names]
        st.dataframe(pd.DataFrame(read_rows), use_container_width=True, hide_index=True)
# import streamlit as st
# import fitz
# import re
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import io

# # ---------------- NLTK Setup ----------------
# def ensure_nltk_data():
#     """Download stopwords/punkt only if not already present (avoids network hit on every rerun)."""
#     for pkg, path in [("stopwords", "corpora/stopwords"), ("punkt", "tokenizers/punkt")]:
#         try:
#             nltk.data.find(path)
#         except LookupError:
#             nltk.download(pkg, quiet=True)

# ensure_nltk_data()
# stop_words = set(stopwords.words('english'))

# # ---------------- Helper Functions ----------------
# def clean_text(text):
#     """Remove extra spaces/newlines."""
#     return re.sub(r'\s+', ' ', text).strip()

# def extract_text(uploaded_file):
#     """Read text from txt or PDF. Returns (full_text, list_of_page_or_paragraph_chunks)."""
#     if uploaded_file.type == "text/plain":
#         raw = uploaded_file.read().decode("utf-8", errors="ignore")
#         # Treat paragraphs as "documents" for a real TF-IDF corpus
#         chunks = [p for p in raw.split("\n\n") if p.strip()]
#         if not chunks:
#             chunks = [raw]
#         return raw, chunks

#     elif uploaded_file.type == "application/pdf":
#         try:
#             pdf_bytes = uploaded_file.read()
#             pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#         except Exception as e:
#             st.error(f"Couldn't read this PDF — it may be corrupted or encrypted. ({e})")
#             st.stop()

#         pages = [page.get_text() for page in pdf_doc]
#         pages = [p for p in pages if p.strip()]
#         full_text = "".join(pages)

#         if not full_text.strip():
#             st.error(
#                 "No extractable text found. This PDF may be a scanned image "
#                 "(needs OCR) rather than real text."
#             )
#             st.stop()

#         return full_text, pages

#     else:
#         st.error("Unsupported file type. Please upload .txt or .pdf")
#         st.stop()

# def get_top_keywords(chunks, top_n=10):
#     """
#     Return top TF-IDF keywords computed over a real corpus (pages/paragraphs),
#     not a single-document list, so IDF actually differentiates terms.
#     Falls back gracefully if there's only one chunk.
#     """
#     vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=50)
#     tfidf_matrix = vectorizer.fit_transform(chunks)
#     feature_names = vectorizer.get_feature_names_out()
#     # Average TF-IDF score per term across all chunks
#     scores = tfidf_matrix.mean(axis=0).A1
#     tfidf_dict = dict(zip(feature_names, scores))
#     top_keywords = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
#     return tfidf_dict, top_keywords

# def plot_bar_chart(top_keywords):
#     """Matplotlib bar chart for crisp bars."""
#     keywords, scores = zip(*top_keywords)
#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.bar(keywords, [s * 100 for s in scores], color='skyblue')
#     ax.set_ylabel("TF-IDF Score (%)")
#     ax.set_xticklabels(keywords, rotation=45, ha="right")
#     fig.tight_layout()
#     st.pyplot(fig)
#     plt.close(fig)

# def generate_wordcloud(tfidf_dict):
#     """Generate and display word cloud."""
#     wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(tfidf_dict)
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.axis("off")
#     st.pyplot(fig)
#     plt.close(fig)

# def chunk_by_sentences(text, max_chars=3000):
#     """
#     Sentence-aware chunking for summarization: packs whole sentences up to
#     max_chars instead of slicing mid-sentence/mid-word.
#     """
#     sentences = nltk.sent_tokenize(text)
#     chunks, current = [], ""
#     for sent in sentences:
#         if len(current) + len(sent) + 1 <= max_chars:
#             current = (current + " " + sent).strip()
#         else:
#             if current:
#                 chunks.append(current)
#             current = sent
#     if current:
#         chunks.append(current)
#     return chunks

# @st.cache_resource
# def load_summarizer():
#     from transformers import pipeline
#     return pipeline("summarization", model="facebook/bart-large-cnn")

# # ---------------- Sidebar ----------------
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Upload", "Visualization", "Summary", "About"])

# # ---------------- Header ----------------
# st.title("AI Policy Visualizer")
# st.write("Upload a policy document and visualize its main themes!")

# # ---------------- Upload Page ----------------
# if page == "Upload":
#     uploaded_file = st.file_uploader("Choose a policy document", type=["pdf", "txt"])
#     if uploaded_file:
#         full_text, chunks = extract_text(uploaded_file)
#         cleaned_text = clean_text(full_text)
#         st.subheader("Document Preview")
#         st.write(cleaned_text[:500] + "......." if len(cleaned_text) > 500 else cleaned_text)

#         # Save for other pages, and reset any stale cached results from a previous upload
#         st.session_state['text'] = cleaned_text
#         st.session_state['chunks'] = [clean_text(c) for c in chunks]
#         st.session_state.pop('top_keywords', None)
#         st.session_state.pop('tfidf_dict', None)
#         st.session_state.pop('summary', None)
#     else:
#         st.warning("Upload a document to continue.")

# # ---------------- Visualization Page ----------------
# elif page == "Visualization":
#     if 'text' not in st.session_state:
#         st.warning("Please upload a document first.")
#     else:
#         cleaned_text = st.session_state['text']
#         sentences = cleaned_text.split(". ")

#         # Compute once, cache in session state so we don't redo this on every page visit
#         if 'top_keywords' not in st.session_state:
#             tfidf_dict, top_keywords = get_top_keywords(st.session_state['chunks'])
#             st.session_state['tfidf_dict'] = tfidf_dict
#             st.session_state['top_keywords'] = top_keywords

#         st.subheader("Top Keywords (TF-IDF)")
#         plot_bar_chart(st.session_state['top_keywords'])

#         st.subheader("Word Cloud")
#         generate_wordcloud(st.session_state['tfidf_dict'])

#         # Download keyword table
#         import pandas as pd
#         kw_df = pd.DataFrame(st.session_state['top_keywords'], columns=["Keyword", "TF-IDF Score"])
#         st.download_button(
#             "Download keywords (CSV)",
#             kw_df.to_csv(index=False),
#             file_name="top_keywords.csv",
#             mime="text/csv",
#         )

#         if st.checkbox("Show first 5 sentences"):
#             st.write(sentences[:5])

# # ---------------- Summary Page ----------------
# elif page == "Summary":
#     if 'text' not in st.session_state:
#         st.warning("Please upload a document first.")
#     else:
#         cleaned_text = st.session_state['text']

#         st.subheader("Local AI Summary")

#         try:
#             summarizer = load_summarizer()
#         except Exception as e:
#             summarizer = None
#             st.error(
#                 "Couldn't load the summarization model. Check your internet connection "
#                 f"and available disk space. ({e})"
#             )

#         if summarizer and st.button("Generate Summary"):
#             chunks = chunk_by_sentences(cleaned_text, max_chars=3000)
#             summaries = []
#             progress = st.progress(0.0, text=f"Summarizing chunk 0/{len(chunks)}")

#             try:
#                 for i, chunk in enumerate(chunks, start=1):
#                     # Skip tiny leftover chunks that are too short to summarize meaningfully
#                     if len(chunk.split()) < 15:
#                         summaries.append(chunk)
#                     else:
#                         result = summarizer(chunk, max_length=180, min_length=60, do_sample=False)
#                         summaries.append(result[0]['summary_text'])
#                     progress.progress(i / len(chunks), text=f"Summarizing chunk {i}/{len(chunks)}")

#                 final_summary = " ".join(summaries)
#                 st.session_state['summary'] = final_summary
#                 progress.empty()
#                 st.success("Summary generated successfully!")
#             except Exception as e:
#                 progress.empty()
#                 st.error(f"Summarization failed partway through: {e}")

#         if 'summary' in st.session_state:
#             st.write(st.session_state['summary'])
#             st.download_button(
#                 "Download summary (.txt)",
#                 st.session_state['summary'],
#                 file_name="summary.txt",
#                 mime="text/plain",
#             )

#         # Show TF-IDF keywords as reference (reuse cached version if available)
#         st.subheader("Summary Keywords")
#         if 'top_keywords' not in st.session_state:
#             tfidf_dict, top_keywords = get_top_keywords(st.session_state['chunks'])
#             st.session_state['tfidf_dict'] = tfidf_dict
#             st.session_state['top_keywords'] = top_keywords
#         plot_bar_chart(st.session_state['top_keywords'])

# # ---------------- About Page ----------------
# elif page == "About":
#     st.info("""
#     **AI Policy Visualizer**
#     - Upload PDF/TXT policy documents
#     - View top TF-IDF keywords (computed across pages/paragraphs for a real corpus) and word cloud
#     - Generate local AI summary (offline Hugging Face model), with sentence-aware chunking
#     - Download keywords and summary
#     - Created by Avantika N @2025
#     """)
# import streamlit as st
# import time
# import fitz
# import re
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import pandas as pd
# from transformers import pipeline

# # ---------------- NLTK Setup ----------------
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# # ---------------- Helper Functions ----------------
# def clean_text(text):
#     """Remove extra spaces/newlines."""
#     return re.sub(r'\s+', ' ', text).strip()

# def extract_text(uploaded_file):
#     """Read text from txt or PDF."""
#     if uploaded_file.type == "text/plain":
#         return uploaded_file.read().decode("utf-8")
#     elif uploaded_file.type == "application/pdf":
#         pdf_bytes = uploaded_file.read()  #uploades the bianry chuck file of pdf from st memroy to pdf_bytes
#         pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")   #fitz convertes the binary file into human readable file
#         text = ""
#         for page in pdf_doc:
#             text += page.get_text()
#         return text
#     else:
#         st.error("Unsupported file type. Please upload .txt or .pdf")
#         st.stop()

# def get_top_keywords(text, top_n=10):
#     """Return top TF-IDF keywords."""
#     vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=50) #Sets up a tool to find important words, Tells the app how to process text
#     tfidf_matrix = vectorizer.fit_transform([text])  #Turns your document into numbers (scores for each word)Converts raw text → meaningful data
#     feature_names = vectorizer.get_feature_names_out()
#     scores = tfidf_matrix.toarray()[0]
#     tfidf_dict = dict(zip(feature_names, scores)) #pairs each word with its frequency in a dict to a tuple
#     top_keywords = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:top_n] #sort the the tuples form the dict , based on the second element of the tuple i.e the frequency, keep the highest ont he first, take the top 10 words only
#     return tfidf_dict, top_keywords

# def plot_bar_chart(top_keywords):
#     """Matplotlib bar chart for crisp bars."""
#     keywords, scores = zip(*top_keywords)
#     plt.figure(figsize=(8,5)) #plot will be 8 inches wide and 5 inches tall
#     plt.bar(keywords, [s*100 for s in scores], color='skyblue') #(x axis-> label, y axis-> numbers turned onto percentde, color-> color of the bar)
#     plt.ylabel("TF-IDF Score (%)")
#     plt.xticks(rotation=45) #rotates the x axis lables (keywords) to 45 degrees to make it clean and readle
#     plt.tight_layout()
#     st.pyplot(plt)
#     plt.clf()  #clears the current figure form the memory to avoid overwrite or mix

# def generate_wordcloud(tfidf_dict):
#     """Generate and display word cloud."""
#     wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(tfidf_dict)
#     fig, ax = plt.subplots(figsize=(10,5))
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.axis("off")
#     st.pyplot(fig)
#     plt.clf()

# # ---------------- Sidebar ----------------
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Upload", "Visualization", "Summary", "About"])

# # ---------------- Header ----------------
# st.title("AI Policy Visualizer")
# st.write("Upload a policy document and visualize its main themes!")

# # ---------------- Upload Page ----------------
# if page == "Upload":
#     uploaded_file = st.file_uploader("Choose a policy document", type=["pdf","txt"])
#     if uploaded_file:
#         text = extract_text(uploaded_file)
#         cleaned_text = clean_text(text)
#         st.subheader("Document Preview")
#         st.write(cleaned_text[:500] + "......." if len(cleaned_text) > 500 else cleaned_text)
#         st.session_state['text'] = cleaned_text  # Save for other pages
#     else:
#         st.warning("Upload a document to continue.")

# # ---------------- Visualization Page ----------------
# elif page == "Visualization":
#     if 'text' not in st.session_state:
#         st.warning("Please upload a document first.")
#     else:
#         cleaned_text = st.session_state['text']
#         sentences = cleaned_text.split(". ")
        
#         # TF-IDF Top Keywords
#         st.subheader("Top Keywords (TF-IDF)")
#         tfidf_dict, top_keywords = get_top_keywords(cleaned_text)
#         plot_bar_chart(top_keywords)

#         # Word Cloud
#         st.subheader("Word Cloud")
#         generate_wordcloud(tfidf_dict)

#         # Optional: show first 5 sentences
#         if st.checkbox("Show first 5 sentences"):
#             st.write(sentences[:5])

# # ---------------- Summary Page ----------------
# elif page == "Summary":
#     if 'text' not in st.session_state:
#         st.warning("Please upload a document first.")
#     else:
#         cleaned_text = st.session_state['text']

#         st.subheader("Local AI Summary")
#         @st.cache_resource
#         def load_summarizer():
#             return pipeline("summarization", model="facebook/bart-large-cnn")     #Loads a BART summarization model  Pretrained summarization model from Hugging Face
#             #pipeline() = Hugging Face’s built-in API for model tasks.
#             # pipeline('summarization')-> is like an endpoint of openAI API that tells to summarize any text that is passed inside this
#         summarizer = load_summarizer()  

#         if st.button("Generate Summary"):
#             with st.spinner("Generating summary..."):
#                 # Chunking for long text
#                 chunks = [cleaned_text[i:i+3000] for i in range(0, len(cleaned_text), 3000)]
#                 summaries = []
#                 for chunk in chunks:
#                     summary = summarizer(chunk, max_length=180, min_length=60, do_sample=False)
#                     summaries.append(summary[0]['summary_text'])
#                 final_summary = " ".join(summaries)
#                 st.success("Summary generated successfully!")
#                 st.write(final_summary)

#         # Show TF-IDF keywords as reference
#         st.subheader("Summary Keywords")
#         _, top_keywords = get_top_keywords(cleaned_text)
#         plot_bar_chart(top_keywords)

# # ---------------- About Page ----------------
# elif page == "About":
#     st.info("""
#     **AI Policy Visualizer**  
#     - Upload PDF/TXT policy documents  
#     - View top TF-IDF keywords and word cloud  
#     - Generate local AI summary (offline Hugging Face model) 
#     - Created by Avantika N @2025 
#     """)
