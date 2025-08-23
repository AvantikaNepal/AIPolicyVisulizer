import streamlit as st
import time
import fitz
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import pandas as pd




nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Sidebar ---
st.sidebar.title("Navigation")
st.sidebar.markdown("""
- Upload
- Visualization (coming soon)
- Analysis (coming soon)
- About
""")

# --- Header ---
st.title("AI Policy Visualizer")
st.write("Upload a policy document (PDF or text), and we'll visualize its main themes!")

st.markdown("---")

# --- File Upload Section ---
st.subheader("Upload a Document")
uploaded_file = st.file_uploader("Choose a policy document", type=["pdf", "txt"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    # decodig the text file and showing the first 500 characters
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    
    elif uploaded_file.type == "application/pdf":
    # Read PDF from uploaded file buffer
        pdf_bytes = uploaded_file.read()
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
        text = ""
        for page in pdf_doc:
            text += page.get_text()
    
    else:
        st.write("Unsupported file type plesea upload txt or Pdf!")


    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)  # remove extra spaces/newlines
        return text.strip()
    cleaned_text = clean_text(text)
    st.subheader("Document Previewed")
    # Simulate AI processing
    with st.spinner("Analyzing document..."):
        time.sleep(2)  # fake processing time
    st.write(cleaned_text[:500] +"......." if len(text) > 500 else text)



    
    # --- Split Into Sentences ---
    sentences = cleaned_text.split(". ")
    st.subheader("Split Sentences")
    st.write(sentences[:5])  # show first 5 sentences



    

    words = [word.lower() for sentence in sentences for word in sentence.split() if word.lower() not in stop_words]
    word_counts = Counter(words)

    # --- Top Keywords Visualization ---
    st.subheader("Top Keywords (Bar Chart)")
    top_n = 10  # number of top keywords to show

    # Filter words: remove stopwords and non-alphabetic tokens
    words = [
        word.lower()
        for sentence in sentences
        for word in sentence.split()
        if word.isalpha() and word.lower() not in stop_words
    ]

    # Count frequency
    word_counts = Counter(words)

    # Get most common words
    most_common = word_counts.most_common(top_n)

    # Convert to DataFrame for Streamlit chart
    df = pd.DataFrame(most_common, columns=["Keyword", "Frequency"]).set_index("Keyword")

    # Display bar chart
    st.bar_chart(df)

    # Detected Topics
    st.markdown("---")
    st.subheader("Detected Topics")
    st.write("Here are some topics our AI *pretends* it found:")
    st.markdown("- Governance\n- Education Reform\n-  Sustainability\n- Digital Transformation")

else:
    st.warning("Please upload a document to get started.")