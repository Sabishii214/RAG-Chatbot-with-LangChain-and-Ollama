# RAG Chatbot with LangChain and Ollama

A local, privacy-focused document question-answering chatbot built with LangChain, ChromaDB, and Ollama. Ask questions about your documents and get AI-powered answers with source citations.

## Features

- **Enhanced NLP Processing**: Uses spaCy for intelligent sentence detection and query optimization
- **Multi-format Document Support**: Load .txt, .md, .pdf, .csv, .docx, and .html files
- **Local LLM**: Uses Ollama for completely private, offline AI responses
- **RAG Architecture**: Retrieval-Augmented Generation ensures answers are grounded in your documents
- **Conversation Memory**: Maintains context across multiple questions
- **Source Citations**: Shows which documents were used to generate answers
- **Web Interface**: Clean Streamlit UI for easy interaction

## Architecture

This project implements a complete RAG (Retrieval-Augmented Generation) pipeline:

1. **Document Loading**: Multi-format document ingestion
2. **Text Processing**: spaCy-powered sentence boundary detection for cleaner chunks
3. **Text Splitting**: Intelligent chunking with overlap
4. **Embeddings**: Local sentence transformers for vector generation
5. **Vector Storage**: ChromaDB for efficient similarity search
6. **Query Enhancement**: spaCy lemmatization for improved search accuracy
7. **Retrieval**: Finds relevant document chunks for queries
8. **Generation**: Ollama LLM generates contextual answers

## Prerequisites

- Python 3.8+
- Ollama installed locally
- 4GB+ RAM recommended

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Sabishii214/RAG-Chatbot-with-LangChain-and-Ollama.git
cd RAG-Chatbot-with-LangChain-and-Ollama
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install "unstructured[md]"
```

### 4. Download spaCy language model
```bash
python -m spacy download en_core_web_sm
```

### 5. Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS/Windows:**
Download from [ollama.com](https://ollama.com)

### 6. Pull the LLM model
```bash
ollama pull llama3.2:3b
```

## Usage

### Step 1: Add Your Documents

Place your documents in the `data/` folder:
```bash
mkdir -p data
# Add your .txt, .md, .pdf, .csv, .docx, or .html files here
```

### Step 2: Create the Vector Database
```bash
python3 create_database.py
```

This will:
- Load all documents from the `data/` folder
- Process text with spaCy for better sentence detection
- Split them into optimized chunks
- Generate embeddings
- Store in ChromaDB

### Step 3: Launch the Web Interface
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

### Step 4: Ask Questions

Type your questions in the chat interface. The bot will:
- Enhance your query with spaCy lemmatization
- Search your documents for relevant information
- Generate answers based on the retrieved context
- Show source documents used

## Project Structure
```
RAG-Chatbot-with-LangChain-and-Ollama/
├── data/                  # Place your documents here
├── chroma/               # Vector database (auto-generated)
├── create_database.py    # Database creation script (with spaCy)
├── streamlit_app.py      # Web interface (with spaCy query enhancement)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Configuration

### Adjust Context Chunks

In the Streamlit sidebar, use the slider to control how many document chunks are retrieved (1-10).

### Change LLM Model

The default model is `llama3.2:3b`. To use a different model:

1. Pull the model: `ollama pull <model-name>`
2. Update `streamlit_app.py` line with the hardcoded model name

Available models:
- `llama3.2:3b` (default, ~2GB)
- `llama3.2:1b` (smaller, faster)
- `mistral` (~4GB)
- See more at [ollama.com/library](https://ollama.com/library)

## Features in Detail

### spaCy NLP Processing

**Document Processing:**
- Intelligent sentence boundary detection
- Cleaner text chunking for better embeddings
- Improved context preservation

**Query Enhancement:**
- Lemmatization for better matching (e.g., "running" → "run")
- Stop word removal for focused searches
- Enhanced semantic similarity

### Conversation Memory

The chatbot remembers the last 6 messages, allowing for natural follow-up questions:
```
You: Who is Alice?
Bot: Alice is the main character...

You: What does she do?
Bot: She falls down a rabbit hole...
```

### Source Citations

Every answer includes clickable sources showing which documents were used.

### Context Viewing

Toggle "Show context" in settings to see the exact text chunks used to generate answers.

## Technology Stack

- **LangChain**: Orchestration framework
- **Ollama**: Local LLM inference
- **ChromaDB**: Vector database
- **spaCy**: Advanced NLP processing for text analysis and query optimization
- **Sentence Transformers**: Local embeddings
- **Streamlit**: Web interface
- **Python**: Core language

## How spaCy Enhances the RAG Pipeline

### 1. Better Document Chunking
spaCy's sentence detection creates more coherent chunks that preserve meaning, leading to better embeddings and retrieval.

### 2. Improved Query Matching
Query lemmatization helps match different word forms:
- "What are the benefits?" → "what benefit"
- Matches documents containing "beneficial", "benefiting", etc.

### 3. Smarter Retrieval
By removing stop words and lemmatizing, searches focus on meaningful terms, improving relevance scores.
