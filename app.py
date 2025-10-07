import streamlit as st
import time
import fitz
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from transformers import pipeline

# ---------------- NLTK Setup ----------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---------------- Helper Functions ----------------
def clean_text(text):
    """Remove extra spaces/newlines."""
    return re.sub(r'\s+', ' ', text).strip()

def extract_text(uploaded_file):
    """Read text from txt or PDF."""
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        pdf_bytes = uploaded_file.read()
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in pdf_doc:
            text += page.get_text()
        return text
    else:
        st.error("Unsupported file type. Please upload .txt or .pdf")
        st.stop()

def get_top_keywords(text, top_n=10):
    """Return top TF-IDF keywords."""
    vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=50)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    tfidf_dict = dict(zip(feature_names, scores))
    top_keywords = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return tfidf_dict, top_keywords

def plot_bar_chart(top_keywords):
    """Matplotlib bar chart for crisp bars."""
    keywords, scores = zip(*top_keywords)
    plt.figure(figsize=(8,5))
    plt.bar(keywords, [s*100 for s in scores], color='skyblue')
    plt.ylabel("TF-IDF Score (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def generate_wordcloud(tfidf_dict):
    """Generate and display word cloud."""
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(tfidf_dict)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
    plt.clf()

# ---------------- Sidebar ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload", "Visualization", "Summary", "About"])

# ---------------- Header ----------------
st.title("AI Policy Visualizer")
st.write("Upload a policy document and visualize its main themes!")

# ---------------- Upload Page ----------------
if page == "Upload":
    uploaded_file = st.file_uploader("Choose a policy document", type=["pdf","txt"])
    if uploaded_file:
        text = extract_text(uploaded_file)
        cleaned_text = clean_text(text)
        st.subheader("Document Preview")
        st.write(cleaned_text[:500] + "......." if len(cleaned_text) > 500 else cleaned_text)
        st.session_state['text'] = cleaned_text  # Save for other pages
    else:
        st.warning("Upload a document to continue.")

# ---------------- Visualization Page ----------------
elif page == "Visualization":
    if 'text' not in st.session_state:
        st.warning("Please upload a document first.")
    else:
        cleaned_text = st.session_state['text']
        sentences = cleaned_text.split(". ")
        
        # TF-IDF Top Keywords
        st.subheader("Top Keywords (TF-IDF)")
        tfidf_dict, top_keywords = get_top_keywords(cleaned_text)
        plot_bar_chart(top_keywords)

        # Word Cloud
        st.subheader("Word Cloud")
        generate_wordcloud(tfidf_dict)

        # Optional: show first 5 sentences
        if st.checkbox("Show first 5 sentences"):
            st.write(sentences[:5])

# ---------------- Summary Page ----------------
elif page == "Summary":
    if 'text' not in st.session_state:
        st.warning("Please upload a document first.")
    else:
        cleaned_text = st.session_state['text']

        st.subheader("Local AI Summary")
        @st.cache_resource
        def load_summarizer():
            return pipeline("summarization", model="facebook/bart-large-cnn")

        summarizer = load_summarizer()

        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                # Chunking for long text
                chunks = [cleaned_text[i:i+3000] for i in range(0, len(cleaned_text), 3000)]
                summaries = []
                for chunk in chunks:
                    summary = summarizer(chunk, max_length=180, min_length=60, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                final_summary = " ".join(summaries)
                st.success("Summary generated successfully!")
                st.write(final_summary)

        # Show TF-IDF keywords as reference
        st.subheader("Summary Keywords")
        _, top_keywords = get_top_keywords(cleaned_text)
        plot_bar_chart(top_keywords)

# ---------------- About Page ----------------
elif page == "About":
    st.info("""
    **AI Policy Visualizer**  
    - Upload PDF/TXT policy documents  
    - View top TF-IDF keywords and word cloud  
    - Generate local AI summary (offline Hugging Face model)  
    """)
