from pathlib import Path
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from pypdf import PdfReader
import docx2txt

Path("uploads").mkdir(exist_ok=True)
Path("chroma_db").mkdir(exist_ok=True)

embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001")

vector_store = Chroma(
    collection_name="my-gpt",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

def read_file_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"
        return text
    
    if suffix == ".docx":
        return docx2txt.process(file_path)
    
    if suffix in [".txt", ".md", ".py", ".csv"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    
    raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, OR CSV")


def add_document_to_rag(file_path:str, thread_id: str):
    text = read_file_text(file_path)

    if not text.strip():
        raise ValueError("No text could be extracted from this file")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 900,
        chunk_overlap = 150
    )

    chunks = splitter.split_text(text)
    docs:List[Document] = [
        Document(
            page_content = chunk,
            metadata = {
                "thread_id" : thread_id,
                "source": Path(file_path).name
            }
        )
        for chunk in chunks
    ]

    vector_store.add_documents(docs)
    return {
        "file_name": Path(file_path).name,
        "chunks" : len(docs)
    }

def retrive_from_rag(query:str, thread_id = str, k:int = 4) ->str:
    docs = vector_store.similarity_search(
    query= query,
    k=k,
    filter={"thread_id": thread_id}
)
    
    if not docs:
        return "No relevant uploaded document content found"
    
    results = []

    for i, doc in enumerate(docs, start = 1):
        source = doc.metadata.get("source", "uploade document")
        results.append(
            f"[Source {i} : {source}] \n {doc.page_content}"
        )
    return "\n\n".join(results)
























