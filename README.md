# 🧠 Interactive QA Bot with Document Upload 📄

## Overview

The **Interactive QA Bot with Document Upload** is a web application that allows users to upload PDF documents, process their content, and ask questions based on the documents. The application uses **Pinecone** for vector retrieval and **Cohere** for generating answers, providing accurate and context-aware responses.

<p align = "center" width = 100% border = 'white'>
    <img src = "Images/Banner.png" alt = "BannerImg">
</p>

## Key Features

✨ **PDF Upload**: Easily upload and process PDF documents.  
🔍 **Document Retrieval**: Retrieve relevant sections of the document for any query.  
🤖 **AI-Powered Q&A**: Get answers generated from state-of-the-art language models.  
💻 **Interactive Web Interface**: User-friendly Streamlit interface for document interaction.  
🚀 **Fast and Scalable**: Utilizes Pinecone's vector database for efficient retrieval.

## Technology Stack

| **Tool**       | **Usage**                          |
| -------------- | ---------------------------------- |
| **Streamlit**  | Interactive web UI for uploading PDFs and asking questions |
| **Pinecone**   | Vector database for storing and retrieving document embeddings |
| **Cohere**     | Language model for generating natural language answers |
| **Transformers** (Hugging Face) | Embedding text for efficient search and retrieval |
| **PyPDF2**     | Extracting text from uploaded PDF files |

---

## 🚀 Quick Start Guide

### Prerequisites

Before you begin, ensure that you have the following installed:

- **Python 3.10+**
- API keys for **Cohere** and **Pinecone**.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/interactive-qa-bot.git
cd interactive-qa-bot
```

### 2. Setup Virtual Environment

Create a virtual environment and activate it.

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Dependencies

Install all necessary Python packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

You’ll need to add your **Cohere** and **Pinecone** API keys. You can either store them as environment variables or hard-code them (not recommended for production).

In **Back_End.py**, replace the placeholders with your actual keys:

```python
cohere_api_key = "your_cohere_api_key"
pinecone_api_key = "your_pinecone_api_key"
```

### 5. Run the Application

```bash
streamlit run Front_End.py
```

Open your browser and go to `http://localhost:8501` to use the application.

---

## 🛠 Project Structure

```
/project-directory
    ├── Front_End.py               # Streamlit-based frontend
    ├── Back_End.py                # Backend logic for PDF processing and QA
    ├── requirements.txt           # Python dependencies
    ├── Dockerfile                 # Docker configuration for deployment
    └── README.md                  # Project documentation
```

## ⚙️ Backend Functions (Back_End.py)

### 1. **Embedding Text**
We use Hugging Face's `sentence-transformers` model to create vector embeddings for document content, enabling efficient retrieval.

```python
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.detach().numpy()[0]
```

### 2. **Process PDF**
Extracts text from each page of the uploaded PDF.

```python
def process_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
```

### 3. **Save Document Embeddings**
Splits the extracted text into chunks and stores them as embeddings in Pinecone.

```python
def save_document_embeddings(text):
    docs = text.split("\n")
    for i, doc in enumerate(docs):
        vector = embed_text(doc)
        index.upsert(vectors=[(str(i), vector)])
    return docs
```

### 4. **Retrieve Documents**
Finds the most relevant document sections based on the user's query.

```python
def retrieve_documents(query, documents, top_k=3):
    query_embedding = embed_text(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k)
    relevant_docs = [documents[int(match['id'])] for match in results['matches']]
    return relevant_docs
```

### 5. **Generate Answer**
Cohere generates an answer based on the retrieved document context.

```python
def generate_answer(relevant_docs, query):
    context = " ".join(relevant_docs)
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=f"Answer the question based on the context: {context}\n\nQuestion: {query}",
        max_tokens=100,
        temperature=0.7
    )
    return response.generations[0].text
```

---

## 🌐 Frontend (Front_End.py)

The frontend is built using **Streamlit**, which allows users to upload PDFs, ask questions, and view generated answers in a simple UI.

### Upload PDF and Process

```python
uploaded_file = st.file_uploader("📂 Upload a PDF document", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF... 🔄"):
        document_text = process_pdf(uploaded_file)
        documents = save_document_embeddings(document_text)
        st.success("✅ Document processed and embeddings saved!")
```

### Ask Questions and Retrieve Documents

```python
query = st.text_input("💬 Ask a question about the document:")

if query:
    with st.spinner("🔍 Retrieving information..."):
        retrieved_docs = retrieve_documents(query, documents)
        st.subheader("📄 Retrieved Documents:")
        for i, doc in enumerate(retrieved_docs, 1):
            st.write(f"**Document {i}:** {doc}")
```

### Generate AI-Powered Answer

```python
with st.spinner("🧠 Generating answer..."):
    answer = generate_answer(retrieved_docs, query)
    st.subheader("💡 Generated Answer:")
    st.write(f"**Answer:** {answer}")
```

---

## 🐳 Docker Deployment

You can easily containerize the application using Docker.

### Dockerfile

```dockerfile
# Use the official Python image from Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "Front_End.py"]
```

### Build and Run the Docker Container

1. **Build the Docker Image**:
   ```bash
   docker build -t interactive-qa-bot .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 8501:8501 interactive-qa-bot
   ```

Access the app at `http://localhost:8501`.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📝 Contributions

We welcome contributions! If you'd like to contribute:

- Submit **issues** for bug reports and feature requests.
- Create **pull requests** for code contributions.
