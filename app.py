import streamlit as st
import time
import fitz
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from openai import OpenAI





client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# --- NLTK setup ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Helper Functions ---
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
        st.error("Unsupported file type. Please upload a .txt or .pdf file.")
        st.stop()

def get_filtered_words(sentences):
    """Return list of words excluding stopwords and non-alphabetic tokens."""
    return [
        word.lower()
        for sentence in sentences
        for word in sentence.split()
        if word.isalpha() and word.lower() not in stop_words
    ]

# --- Sidebar ---
st.sidebar.title("Navigation")
st.sidebar.markdown("""
- Upload
- Visualization
- Analysis
- About
""")

# --- Header ---
st.title("AI Policy Visualizer")
st.write("Upload a policy document (PDF or text), and we'll visualize its main themes!")
st.markdown("---")

# --- File Upload Section ---
st.subheader("Upload a Document")
uploaded_file = st.file_uploader("Choose a policy document", type=["pdf", "txt"])

if uploaded_file:
    st.success("File uploaded successfully!")

    # --- Read & Clean Text ---
    text = extract_text(uploaded_file)
    cleaned_text = clean_text(text)

    # --- Document Preview ---
    st.subheader("Document Preview")
    with st.spinner("Analyzing document..."):
        time.sleep(1)
    st.write(cleaned_text[:500] + "......." if len(cleaned_text) > 500 else cleaned_text)

    # --- Split Into Sentences ---
    sentences = cleaned_text.split(". ")
    st.subheader("Split Sentences")
    st.write(sentences[:5])

    # --- TF-IDF Keywords & Word Cloud ---
    # --- TF-IDF Keywords & Rounded Bar Chart ---
    st.subheader("Top Keywords (TF-IDF)")

    # Vectorize document
    vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=50)
    tfidf_matrix = vectorizer.fit_transform([cleaned_text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    # Create dict word:score
    tfidf_dict = dict(zip(feature_names, scores))

    # Get top 10 keywords
    top_keywords_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    df_tfidf = pd.DataFrame(top_keywords_tfidf, columns=["Keyword", "TF-IDF"]).set_index("Keyword")

    # --- Display bar chart with rounded values ---
    df_tfidf_display = (df_tfidf * 100).round(0).astype(int)  # scale and round for display
    st.bar_chart(df_tfidf_display)

    # --- Word Cloud ---
    st.subheader("Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(tfidf_dict)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)


    st.markdown("---")
    st.subheader("Detected Topics (Auto)")

    # Take top 5 keywords from TF-IDF
    top_keywords_list = [word for word, _ in top_keywords_tfidf[:5]]

    topics = {}
    for kw in top_keywords_list:
        for sentence in sentences:
            if kw in sentence.lower():
                topics[kw] = sentence.strip()
                break  # first sentence containing the keyword

    for kw, sent in topics.items():
        st.markdown(f"- **{kw.capitalize()}** → {sent}")


    st.markdown("---")
    st.subheader("AI Summary")

    if st.button("Generate AI Summary"):
        with st.spinner("Analyzing document with AI..."):
            try:
                # --- MOCK API RESPONSE ---
                # Comment this block if you have working API credits
                summary = (
                    "⚡ This is a mock summary demonstrating AI integration.\n\n"
                    "1️⃣ First paragraph: The policy document discusses key areas such as governance and sustainability.\n\n"
                    "2️⃣ Second paragraph: It highlights challenges and proposes potential solutions in education and digital transformation.\n\n"
                    "3️⃣ Third paragraph: Overall, the document emphasizes strategic priorities for development and policy implementation."
                )

                # --- REAL API CALL ---
                # Uncomment this block if your API key is valid and has quota
                # response = client.chat.completions.create(
                #     model="gpt-4o-mini",
                #     messages=[
                #         {"role": "system", "content": "You are an AI assistant summarizing policy documents clearly and concisely."},
                #         {"role": "user", "content": f"Summarize this document in 3 short paragraphs: {cleaned_text[:4000]}"}
                #     ],
                #     temperature=0.5,
                # )
                # summary = response.choices[0].message.content

                st.success("Summary generated successfully!")
                st.write(summary)

            except Exception as e:
                st.error(f"Error: {e}")




else:
    st.warning("Please upload a document to get started.")
