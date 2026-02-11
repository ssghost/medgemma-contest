import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "./pdf"       
DB_PATH = "./database"  

def build_db() -> None:
    print(f"Scanning for PDFs in '{DATA_PATH}'...")

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Folder '{DATA_PATH}' created.")
        print(f"Please put your medical PDF files into the '{DATA_PATH}' folder and run this script again!")
        exit()

    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    try:
        raw_documents = loader.load()
    except Exception as e:
        print(f"Error loading PDFs: {e}")
        exit()

    if not raw_documents:
        print(f"No PDFs found in '{DATA_PATH}'! Please add some files.")
        exit()

    print(f"Loaded {len(raw_documents)} pages from PDFs.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    documents = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(documents)} text chunks.")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("(Cleaned up old database files)")

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("Knowledge Base Built Successfully.")
    print(f"Indexed {len(documents)} chunks into '{DB_PATH}'.")

if __name__ == '__main__':
    build_db()