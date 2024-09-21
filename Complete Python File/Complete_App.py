import streamlit as st
import pinecone
import cohere
from transformers import AutoTokenizer, AutoModel
import torch
import PyPDF2
from pinecone import Pinecone

# ----------------------------------------------------------------------- BACkEND -------------------------------------------------------------------------------------------------

# Backend Setup: Initialize Cohere and Pinecone
cohere_api_key = "LgaWUuuPnuamPELt1VTEqP6WmwEYOjfLKFrsUg6P"
co = cohere.Client(cohere_api_key)

pc = Pinecone(api_key="b10a42ec-9e36-4523-a2e1-08ce6de826f6")
index = pc.Index("pdf384")

# Load pre-trained model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Function to embed text
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.detach().numpy()[0]

# Upload and process PDF
def process_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Save document embeddings
def save_document_embeddings(text):
    docs = text.split("\n")  # Split text into chunks (can be modified)
    for i, doc in enumerate(docs):
        vector = embed_text(doc)
        index.upsert(vectors=[(str(i), vector)])
    return docs

# Retrieve documents based on query
def retrieve_documents(query, top_k=3):
    query_embedding = embed_text(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k)
    relevant_docs = [documents[int(match['id'])] for match in results['matches']]
    return relevant_docs

# Generate answer using Cohere
def generate_answer(relevant_docs, query):
    context = " ".join(relevant_docs)
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=f"Answer the question based on the context: {context}\n\nQuestion: {query}",
        max_tokens=100,
        temperature=0.7
    )
    return response.generations[0].text
# ----------------------------------------------------------------------- FRONTEND -----------------------------------------------------------------------------------------------
# Frontend Setup: Streamlit UI
st.title("🧠 Interactive QA Bot with Document Upload 📄")

# Upload PDF
uploaded_file = st.file_uploader("📂 Upload a PDF document", type="pdf")

if uploaded_file is not None:
    # Process the PDF and extract text
    with st.spinner("Processing PDF... 🔄"):
        document_text = process_pdf(uploaded_file)
        documents = save_document_embeddings(document_text)
        st.success("✅ Document processed and embeddings saved!")

    # Ask a question
    query = st.text_input("💬 Ask a question about the document:")
    
    if query:
        # Retrieve relevant documents
        with st.spinner("🔍 Retrieving information..."):
            retrieved_docs = retrieve_documents(query)
            
            st.subheader("📄 Retrieved Documents:")
            for i, doc in enumerate(retrieved_docs, 1):
                st.write(f"**Document {i}:** {doc}")

        # Generate answer
        with st.spinner("🧠 Generating answer..."):
            answer = generate_answer(retrieved_docs, query)
            
            st.subheader("💡 Generated Answer:")
            st.write(f"**Answer:** {answer}")
