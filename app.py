import streamlit as st
import fitz
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# ---------------- NLTK Setup ----------------
def ensure_nltk_data():
    """Download stopwords/punkt only if not already present (avoids network hit on every rerun)."""
    for pkg, path in [("stopwords", "corpora/stopwords"), ("punkt", "tokenizers/punkt")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

ensure_nltk_data()
stop_words = set(stopwords.words('english'))

# ---------------- Helper Functions ----------------
def clean_text(text):
    """Remove extra spaces/newlines."""
    return re.sub(r'\s+', ' ', text).strip()

def extract_text(uploaded_file):
    """Read text from txt or PDF. Returns (full_text, list_of_page_or_paragraph_chunks)."""
    if uploaded_file.type == "text/plain":
        raw = uploaded_file.read().decode("utf-8", errors="ignore")
        # Treat paragraphs as "documents" for a real TF-IDF corpus
        chunks = [p for p in raw.split("\n\n") if p.strip()]
        if not chunks:
            chunks = [raw]
        return raw, chunks

    elif uploaded_file.type == "application/pdf":
        try:
            pdf_bytes = uploaded_file.read()
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            st.error(f"Couldn't read this PDF — it may be corrupted or encrypted. ({e})")
            st.stop()

        pages = [page.get_text() for page in pdf_doc]
        pages = [p for p in pages if p.strip()]
        full_text = "".join(pages)

        if not full_text.strip():
            st.error(
                "No extractable text found. This PDF may be a scanned image "
                "(needs OCR) rather than real text."
            )
            st.stop()

        return full_text, pages

    else:
        st.error("Unsupported file type. Please upload .txt or .pdf")
        st.stop()

def get_top_keywords(chunks, top_n=10):
    """
    Return top TF-IDF keywords computed over a real corpus (pages/paragraphs),
    not a single-document list, so IDF actually differentiates terms.
    Falls back gracefully if there's only one chunk.
    """
    vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=50)
    tfidf_matrix = vectorizer.fit_transform(chunks)
    feature_names = vectorizer.get_feature_names_out()
    # Average TF-IDF score per term across all chunks
    scores = tfidf_matrix.mean(axis=0).A1
    tfidf_dict = dict(zip(feature_names, scores))
    top_keywords = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return tfidf_dict, top_keywords

def plot_bar_chart(top_keywords):
    """Matplotlib bar chart for crisp bars."""
    keywords, scores = zip(*top_keywords)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(keywords, [s * 100 for s in scores], color='skyblue')
    ax.set_ylabel("TF-IDF Score (%)")
    ax.set_xticklabels(keywords, rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def generate_wordcloud(tfidf_dict):
    """Generate and display word cloud."""
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(tfidf_dict)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

def chunk_by_sentences(text, max_chars=3000):
    """
    Sentence-aware chunking for summarization: packs whole sentences up to
    max_chars instead of slicing mid-sentence/mid-word.
    """
    sentences = nltk.sent_tokenize(text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks

@st.cache_resource
def load_summarizer():
    from transformers import pipeline
    return pipeline("summarization", model="facebook/bart-large-cnn")

# ---------------- Sidebar ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload", "Visualization", "Summary", "About"])

# ---------------- Header ----------------
st.title("AI Policy Visualizer")
st.write("Upload a policy document and visualize its main themes!")

# ---------------- Upload Page ----------------
if page == "Upload":
    uploaded_file = st.file_uploader("Choose a policy document", type=["pdf", "txt"])
    if uploaded_file:
        full_text, chunks = extract_text(uploaded_file)
        cleaned_text = clean_text(full_text)
        st.subheader("Document Preview")
        st.write(cleaned_text[:500] + "......." if len(cleaned_text) > 500 else cleaned_text)

        # Save for other pages, and reset any stale cached results from a previous upload
        st.session_state['text'] = cleaned_text
        st.session_state['chunks'] = [clean_text(c) for c in chunks]
        st.session_state.pop('top_keywords', None)
        st.session_state.pop('tfidf_dict', None)
        st.session_state.pop('summary', None)
    else:
        st.warning("Upload a document to continue.")

# ---------------- Visualization Page ----------------
elif page == "Visualization":
    if 'text' not in st.session_state:
        st.warning("Please upload a document first.")
    else:
        cleaned_text = st.session_state['text']
        sentences = cleaned_text.split(". ")

        # Compute once, cache in session state so we don't redo this on every page visit
        if 'top_keywords' not in st.session_state:
            tfidf_dict, top_keywords = get_top_keywords(st.session_state['chunks'])
            st.session_state['tfidf_dict'] = tfidf_dict
            st.session_state['top_keywords'] = top_keywords

        st.subheader("Top Keywords (TF-IDF)")
        plot_bar_chart(st.session_state['top_keywords'])

        st.subheader("Word Cloud")
        generate_wordcloud(st.session_state['tfidf_dict'])

        # Download keyword table
        import pandas as pd
        kw_df = pd.DataFrame(st.session_state['top_keywords'], columns=["Keyword", "TF-IDF Score"])
        st.download_button(
            "Download keywords (CSV)",
            kw_df.to_csv(index=False),
            file_name="top_keywords.csv",
            mime="text/csv",
        )

        if st.checkbox("Show first 5 sentences"):
            st.write(sentences[:5])

# ---------------- Summary Page ----------------
elif page == "Summary":
    if 'text' not in st.session_state:
        st.warning("Please upload a document first.")
    else:
        cleaned_text = st.session_state['text']

        st.subheader("Local AI Summary")

        try:
            summarizer = load_summarizer()
        except Exception as e:
            summarizer = None
            st.error(
                "Couldn't load the summarization model. Check your internet connection "
                f"and available disk space. ({e})"
            )

        if summarizer and st.button("Generate Summary"):
            chunks = chunk_by_sentences(cleaned_text, max_chars=3000)
            summaries = []
            progress = st.progress(0.0, text=f"Summarizing chunk 0/{len(chunks)}")

            try:
                for i, chunk in enumerate(chunks, start=1):
                    # Skip tiny leftover chunks that are too short to summarize meaningfully
                    if len(chunk.split()) < 15:
                        summaries.append(chunk)
                    else:
                        result = summarizer(chunk, max_length=180, min_length=60, do_sample=False)
                        summaries.append(result[0]['summary_text'])
                    progress.progress(i / len(chunks), text=f"Summarizing chunk {i}/{len(chunks)}")

                final_summary = " ".join(summaries)
                st.session_state['summary'] = final_summary
                progress.empty()
                st.success("Summary generated successfully!")
            except Exception as e:
                progress.empty()
                st.error(f"Summarization failed partway through: {e}")

        if 'summary' in st.session_state:
            st.write(st.session_state['summary'])
            st.download_button(
                "Download summary (.txt)",
                st.session_state['summary'],
                file_name="summary.txt",
                mime="text/plain",
            )

        # Show TF-IDF keywords as reference (reuse cached version if available)
        st.subheader("Summary Keywords")
        if 'top_keywords' not in st.session_state:
            tfidf_dict, top_keywords = get_top_keywords(st.session_state['chunks'])
            st.session_state['tfidf_dict'] = tfidf_dict
            st.session_state['top_keywords'] = top_keywords
        plot_bar_chart(st.session_state['top_keywords'])

# ---------------- About Page ----------------
elif page == "About":
    st.info("""
    **AI Policy Visualizer**
    - Upload PDF/TXT policy documents
    - View top TF-IDF keywords (computed across pages/paragraphs for a real corpus) and word cloud
    - Generate local AI summary (offline Hugging Face model), with sentence-aware chunking
    - Download keywords and summary
    - Created by Avantika N @2025
    """)
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
