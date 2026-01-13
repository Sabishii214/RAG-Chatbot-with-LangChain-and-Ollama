import streamlit as st
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import spacy

CHROMA_PATH = "chroma"

# Load spaCy model for query processing (optional enhancement)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model not found. Downloading...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("RAG Chatbot with LangChain and Ollama")
st.markdown("Ask questions about your documents")

if 'messages' not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    num_results = st.slider("Context chunks", 1, 10, 3)
    show_sources = st.checkbox("Show sources", value=True)
    show_context = st.checkbox("Show context", value=False)
    
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

@st.cache_resource
def load_database():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def process_query_with_spacy(query: str) -> str:
    """Optional: Process query with spaCy for better search results"""
    doc = nlp(query)
    # Remove stop words and lemmatize for better matching
    processed_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    # If processed query is too short, return original
    if len(processed_tokens) < 2:
        return query
    return " ".join(processed_tokens)

try:
    db = load_database()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and show_sources:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(f"- {source}")
            if "context" in message and show_context:
                with st.expander("Context"):
                    st.write(message["context"])
    
    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                # Optional: Use spaCy to enhance query (comment out if not needed)
                enhanced_query = process_query_with_spacy(prompt)
                
                # Use enhanced query for search, but original prompt for context
                results = db.similarity_search_with_relevance_scores(enhanced_query, k=num_results)
                
                if not results or results[0][1] < 0.3:
                    response = "No relevant information found in the documents."
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
                    sources = list(set([doc.metadata.get("source", "unknown") for doc, _ in results]))
                    
                    chat_history = "\n".join([
                        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
                        for msg in st.session_state.messages[-6:]
                    ])
                    
                    prompt_text = f"""Answer questions based on provided documents.

Previous conversation:
{chat_history}

Context:
{context_text}

Question: {prompt}

Provide a clear answer based on the context. Reference previous conversation if relevant."""
                    
                    with st.spinner("Generating answer..."):
                        try:
                            llm = OllamaLLM(model="llama3.2:3b")
                            response = llm.invoke(prompt_text)
                            
                            st.markdown(response)
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "sources": sources,
                                "context": context_text
                            })
                            
                            if show_sources:
                                with st.expander("Sources"):
                                    for source in sources:
                                        st.write(f"- {source}")
                            
                            if show_context:
                                with st.expander("Context"):
                                    st.write(context_text)
                                    
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})

except Exception as e:
    st.error(f"Error loading database: {str(e)}")
    st.info("Run 'python3 create_database.py' first")