import streamlit as st
from Back_End import embed_text, save_document_embeddings, process_pdf, retrieve_documents, generate_answer

# Frontend Setup: Streamlit UI
st.title("🧠 Interactive QA Bot with Document Upload 📄")

def main():
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
            with st.spinner("🔍 Retrieving information..."):
                retrieved_docs = retrieve_documents(query, documents)

                st.subheader("📄 Retrieved Documents:")
                for i, doc in enumerate(retrieved_docs, 1):
                    st.write(f"**Document {i}:** {doc}")

            # Generate answer
            with st.spinner("🧠 Generating answer..."):
                answer = generate_answer(retrieved_docs, query)

                st.subheader("💡 Generated Answer:")
                st.write(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()
