# RAG Chatbot with LangChain and Ollama

A local, privacy-focused document question-answering chatbot built with LangChain, ChromaDB, and Ollama. Ask questions about your documents and get AI-powered answers with source citations.

## Features

- **Multi-format Document Support**: Load .txt, .md, .pdf, .csv, .docx, and .html files
- **Local LLM**: Uses Ollama for completely private, offline AI responses
- **RAG Architecture**: Retrieval-Augmented Generation ensures answers are grounded in your documents
- **Conversation Memory**: Maintains context across multiple questions
- **Source Citations**: Shows which documents were used to generate answers
- **Web Interface**: Clean Streamlit UI for easy interaction
- **No API Costs**: Everything runs locally on your machine

## Architecture

This project implements a complete RAG (Retrieval-Augmented Generation) pipeline:

1. **Document Loading**: Multi-format document ingestion
2. **Text Splitting**: Intelligent chunking with overlap
3. **Embeddings**: Local sentence transformers for vector generation
4. **Vector Storage**: ChromaDB for efficient similarity search
5. **Retrieval**: Finds relevant document chunks for queries
6. **Generation**: Ollama LLM generates contextual answers

## Prerequisites

- Python 3.8+
- Ollama installed locally
- 4GB+ RAM recommended

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
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

### 4. Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS/Windows:**
Download from [ollama.com](https://ollama.com)

### 5. Pull the LLM model

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
- Split them into chunks
- Generate embeddings
- Store in ChromaDB

### Step 3: Launch the Web Interface

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

### Step 4: Ask Questions

Type your questions in the chat interface. The bot will:
- Search your documents for relevant information
- Generate answers based on the retrieved context
- Show source documents used

## Project Structure

```
RAG Chatbot/
├── data/                  # Place your documents here
├── chroma/               # Vector database (auto-generated)
├── create_database.py    # Database creation script
├── streamlit_app.py      # Web interface
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
- **Sentence Transformers**: Local embeddings
- **Streamlit**: Web interface
- **Python**: Core language