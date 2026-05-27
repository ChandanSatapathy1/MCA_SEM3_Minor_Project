# Document-Based FAQ Chatbot Using RAG (Retrieval Augmented Generation)
# MCA Minor Project by Chandan Satapathy
# Idea: In BPO work, agents face high AHT (Average Handling Time) because they search long documents.
# This chatbot helps by quickly finding answers from FAQ files (PDF or CSV).

import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Set page title
st.set_page_config(page_title="Document‑Based FAQ Chatbot Using RAG", layout="centered")
st.title("Document‑Based FAQ Chatbot Using RAG")

# Upload file (PDF, CSV, or TXT)
uploaded_file = st.file_uploader("Upload FAQ file", type=["pdf", "csv", "txt"])

if uploaded_file is not None:
    # Step 1: Read the file
    if uploaded_file.type == "application/pdf":
        # For PDF files
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        faqs = text.split("\n")
        faqs = [f.strip() for f in faqs if f.strip() != ""]

    elif uploaded_file.type == "text/csv":
        # For CSV files (with Question and Answer columns)
        df = pd.read_csv(uploaded_file)
        faqs = (df["Question"] + " " + df["Answer"]).tolist()

    else:
        # For TXT files
        faqs = uploaded_file.read().decode("utf-8").split("\n")
        faqs = [f.strip() for f in faqs if f.strip() != ""]

    # Show success message
    st.success("File uploaded successfully! You can now ask questions below.")
    st.write("Total chunks created:", len(faqs))

    # Show first 3 FAQs for demo
    for i, chunk in enumerate(faqs[:3]):
        st.write(f"Chunk {i+1}:", chunk)

    # Step 2: TF-IDF setup (keyword search)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(faqs)

    # Step 3: Embedding setup (semantic search)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_model = model.encode(faqs)
    index = faiss.IndexFlatL2(embedding_model.shape[1])
    index.add(np.array(embedding_model, dtype='float32'))

    # Step 4: User query
    query = st.text_input("Ask your question:")
    if query:
        # First try TF-IDF
        tfidf_query = vectorizer.transform([query])
        scores = (X * tfidf_query.T).toarray()
        best_match = scores.argmax()

        if scores.max() > 0.2:
            answer = faqs[best_match]
        else:
            # If TF-IDF not strong, use embeddings
            q_embed = model.encode([query])
            dist, idx = index.search(np.array(q_embed, dtype='float32'), k=1)
            answer = faqs[idx[0][0]]

        # Show answer
        st.write("Answer:", answer)

# Clear Chat button
if st.button("Clear Chat"):
    st.experimental_rerun()

# Footer
st.caption("Developed by Chandan Satapathy | MCA Minor Project | Amity University Online")
