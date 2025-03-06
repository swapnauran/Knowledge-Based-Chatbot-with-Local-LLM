# MUST BE AT VERY TOP - Fixes NumPy initialization
import numpy as np
np._DummyFunc = lambda: None  # Critical FAISS workaround

import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DOCS_FOLDER = "company_docs"

def train_chatbot():
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Safer for compatibility
    )
    
    # Document processing
    docs = []
    for file in os.listdir(DOCS_FOLDER):
        file_path = os.path.join(DOCS_FOLDER, file)
        
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith((".docx", ".doc")):
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                continue
                
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            continue
    
    # Text splitting with error handling
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? "]
        )
        documents = text_splitter.split_documents(docs)
    except Exception as e:
        print(f"Text splitting failed: {e}")
        return

    # FAISS vector store with validation
    try:
        db = FAISS.from_documents(documents, embeddings)
        db.save_local("faiss_index")
        print("✅ Training successful! Knowledge base ready.")
    except Exception as e:
        print(f"Vector store creation failed: {e}")

if __name__ == "__main__":
    train_chatbot()