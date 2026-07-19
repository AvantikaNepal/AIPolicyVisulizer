**AI Policy Visualizer**

AI Policy Visualizer is a web-based application that allows users to upload policy documents (PDF or text) and explore their key insights through keyword extraction, TF-IDF analysis, word clouds, and local AI-generated summaries. This project demonstrates how natural language processing (NLP) and local large language models (LLMs) can help make sense of long policy documents efficiently.

**Features**
1.Document Upload
2.Upload policy documents in PDF or plain text format.
3.Preview the first 500 characters of the document.
4.Text Cleaning & Preprocessing
5.Removes extra spaces and newlines.
6.Splits the document into sentences for easier processing.
7.Keyword Extraction
8.Uses TF-IDF (Term Frequency–Inverse Document Frequency) to identify important words.
9.Displays the top keywords in a bar chart and a word cloud for visual analysis.
10.AI Summarization (Local Model)
11.Generates a summary of the document using a local BART-based model from Hugging Face Transformers.
12.Summary helps identify the main points quickly without requiring external APIs.
13.Detected Topics
14.Extracts top keywords from TF-IDF and finds sentences containing them.
15.Provides a simple view of the document’s key topics.

**Tech Stack**
1. Python 3.10+
2. Streamlit – Frontend and web app interface
3. PyMuPDF (fitz) – PDF reading
4. NLTK – Stopwords and basic NLP preprocessing
5. scikit-learn – TF-IDF vectorization
6. Matplotlib & WordCloud – Visualization of keywords
7. Hugging Face Transformers – Local summarization with BART model

**How to Run Locally**
-> Clone the repository:
-> git clone [https://github.com/yourusername/ai-policy-visualizer.git](https://github.com/AvantikaNepal/AIPolicyVisulizer.git)
-> cd ai-policy-visualizer
-> Install required packages:
-> pip install -r requirements.txt
-> Run the Streamlit app:
-> streamlit run app.py
-> Upload a policy document (PDF or text) and explore the extracted keywords, word cloud, and local AI-generated summary.

**Project Status**
1. Document upload and preview
2. Text preprocessing and sentence splitting
3. TF-IDF keyword extraction and visualization
4. Word cloud generation
5. Local AI summarization using Hugging Face Transformers
6.  LLM API integration (OpenAI GPT) is not active due to quota limitations

**Future Work**

Integrate structured AI summarization to extract policy objectives, recommendations, and stakeholders.
Add interactive graphs for visualizing relationships between keywords and topics.
Incorporate external LLM APIs for more advanced summarization once API access is available.

**License**
Author: Avantika Nepal
