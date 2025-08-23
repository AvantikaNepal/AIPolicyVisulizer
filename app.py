import streamlit as st
import time
import fitz
import re

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
        def clean_text(text):
            text = re.sub(r'\s+', ' ', text)  # remove extra spaces/newlines
            # text = re.sub(r'[^a-zA-Z\s]', '', text)  # optional: remove numbers/punctuation
            return text.strip()
        cleaned_text = clean_text(text)
        # st.subheader("🧹 Cleaned Text")
        # st.write(cleaned_text[:500])
        st.subheader("Document Previewed")
        # Simulate AI processing
        with st.spinner("Analyzing document..."):
            time.sleep(2)  # fake processing time
        st.write(cleaned_text[:500] +"......." if len(text) > 500 else text)
    
    elif uploaded_file.type == "application/pdf":
    # Read PDF from uploaded file buffer
        pdf_bytes = uploaded_file.read()
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
        text = ""
        for page in pdf_doc:
            text += page.get_text()
    
        st.subheader("Document Preview (PDF)")
        with st.spinner("Analyzing document..."):
            time.sleep(2)  # fake processing time
        st.write(text[:500] + "..." if len(text) > 500 else text)
    else:
        st.write("Unsupported file type plesea upload txt or Pdf!")

    # Detected Topics
    st.markdown("---")
    st.subheader("Detected Topics")
    st.write("Here are some topics our AI *pretends* it found:")
    st.markdown("- Governance\n- Education Reform\n-  Sustainability\n- Digital Transformation")

    # Visual Summary (dummy chart)
    st.subheader("Visual Summary")
    st.bar_chart({"Frequency": [10, 7, 5, 3]}, height=300)

else:
    st.warning("Please upload a document to get started.")
