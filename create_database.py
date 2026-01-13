from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
from typing import List
import spacy

CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Load spaCy model at module level for efficiency
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully")
except OSError:
    print("spaCy model not found. Downloading...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def main():
    print("Starting database creation...")
    documents = load_documents()
    if not documents:
        print("Error: No documents found")
        return
    chunks = split_documents(documents)
    save_to_chroma(chunks)
    print("Database creation complete")

def load_documents() -> List[Document]:
    print(f"Loading documents from '{DATA_PATH}'...")
    all_documents = []
    loaders_config = [
        ('**/*.txt', TextLoader, {'autodetect_encoding': True}, 'text files'),
        ('**/*.md', TextLoader, {'autodetect_encoding': True}, 'markdown files'),
        ('**/*.pdf', PyPDFLoader, {}, 'PDF files'),
        ('**/*.csv', CSVLoader, {}, 'CSV files'),
        ('**/*.docx', UnstructuredWordDocumentLoader, {}, 'Word documents'),
        ('**/*.html', UnstructuredHTMLLoader, {}, 'HTML files')
    ]
    for glob_pattern, loader_cls, loader_kwargs, description in loaders_config:
        try:
            loader = DirectoryLoader(
                DATA_PATH,
                glob=glob_pattern,
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs,
                show_progress=False,
                use_multithreading=True,
                silent_errors=True
            )
            docs = loader.load()
            if docs:
                print(f"  Loaded {len(docs)} {description}")
                all_documents.extend(docs)
        except Exception as e:
            print(f"  Warning: Could not load {description}: {str(e)}")
    
    if not all_documents:
        print("No documents found. Supported: .txt, .md, .pdf, .csv, .docx, .html")
        return []
    
    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents

def split_documents(documents: List[Document]) -> List[Document]:
    print("Splitting documents into chunks using spaCy for sentence detection...")
    
    # Process documents with spaCy for better sentence segmentation
    processed_documents = []
    for doc in documents:
        # Use spaCy to process the text
        spacy_doc = nlp(doc.page_content)
        
        # Extract sentences using spaCy's sentence boundary detection
        sentences = [sent.text.strip() for sent in spacy_doc.sents if sent.text.strip()]
        
        # Join sentences back (spaCy improves the quality of splitting later)
        processed_text = " ".join(sentences)
        
        # Create new document with processed text
        processed_doc = Document(
            page_content=processed_text,
            metadata=doc.metadata
        )
        processed_documents.append(processed_doc)
    
    print(f"Processed {len(processed_documents)} documents with spaCy")
    
    # Now split using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(processed_documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    if chunks:
        print(f"Sample chunk: {chunks[0].page_content[:150]}...")
    return chunks

def save_to_chroma(chunks: List[Document]):
    print(f"Saving to ChromaDB at '{CHROMA_PATH}'...")
    if os.path.exists(CHROMA_PATH):
        print("Removing existing database...")
        shutil.rmtree(CHROMA_PATH)
    
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    db = None
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"Processing batch {batch_num}/{total_batches}")
        
        if db is None:
            db = Chroma.from_documents(batch, embeddings, persist_directory=CHROMA_PATH)
        else:
            db.add_documents(batch)
    
    print(f"Saved {len(chunks)} chunks to ChromaDB")

if __name__ == "__main__":
    main()