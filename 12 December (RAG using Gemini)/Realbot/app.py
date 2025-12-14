import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = "AIzaSyAB_tQ2biQ7RmNok47-LU9zh4bT_M_3naw"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Page configuration
st.set_page_config(page_title="Document Chat", page_icon="📄", layout="wide")

st.title("📄 Chat with Document")
st.markdown("Ask questions about your document using AI-powered RAG")

# Initialize vector store
@st.cache_resource
def initialize_vectorstore():
    """Initialize and cache the vector store"""
    with st.spinner("Loading and processing document..."):
        # Step 1: Load PDF
        pdf_path = "doc202517482201.pdf"
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        st.info(f"Loaded {len(pages)} pages from document")
        
        # Step 2: Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(pages)
        st.info(f"Created {len(chunks)} text chunks")
        
        # Step 3: Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004", 
            google_api_key=GOOGLE_API_KEY
        )
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.success("Document processed and ready for questions!")
        return vectorstore

# Initialize vector store
if st.session_state.vectorstore is None:
    st.session_state.vectorstore = initialize_vectorstore()
    st.session_state.initialized = True

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Step 1: Retrieve relevant context
            result = st.session_state.vectorstore.similarity_search(prompt, k=3)
            context = "\n\n".join([doc.page_content for doc in result])
            
            # Step 2: Create prompt with context
            rag_prompt = f"""Use the following document context to answer the question.
If the answer is not in the context, say that you don't have enough information to answer based on the document.

Context: {context}

Question: {prompt}

Answer:"""
            
            # Step 3: Generate response
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7
            )
            
            response = llm.invoke(input=rag_prompt)
            answer = response.content.strip()
            
            st.markdown(answer)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses RAG (Retrieval-Augmented Generation) to answer questions about your document.
    
    **How it works:**
    1. Document is loaded and split into chunks
    2. Chunks are converted to embeddings
    3. Your question is matched with relevant chunks
    4. AI generates an answer based on the context
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
