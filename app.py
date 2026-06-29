import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
import tempfile

# ------------------ CONFIG ------------------
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key
)
st.title("📚 Multi-PDF RAG Chatbot")

# ------------------ SESSION STATE ------------------
if "db" not in st.session_state:
    st.session_state.db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ PDF UPLOAD ------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# ------------------ PROCESS PDFs ------------------
if uploaded_files and st.button("Process PDFs"):

    all_docs = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)
    st.write(f"Total chunks: {len(chunks)}")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # FAISS DB
    st.session_state.db = FAISS.from_documents(chunks, embeddings)

    st.success("✅ PDFs processed and ready for chat!")

# ------------------ CHAT ------------------
query = st.text_input("Ask a question from your PDFs:")

if query and st.session_state.db:

    docs = st.session_state.db.similarity_search(query, k=4)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a helpful assistant. Use ONLY the context below.

Context:
{context}

Question: {query}

Answer clearly and concisely:
"""

    response = llm.invoke(prompt)

    st.write("### 🤖 Answer:")
    st.write(response.content)

    st.session_state.chat_history.append((query, response.content))

# ------------------ CHAT HISTORY ------------------
if st.session_state.chat_history:
    st.write("## 💬 Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {a}")
        st.write("---")