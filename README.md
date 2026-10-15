# Mango Lassi Chat Assistant (Python & Streamlit)

A high-performance, privacy-conscious, agentic AI chat assistant powered by local Ollama models (`llama3.2`, `embeddinggemma`), ChromaDB vector store, Serper.dev live web search, and dual-layer PII guardrails built with Python and Streamlit.

---

## How to Run "Mango Lassi"

### Run Embedding & Llama Models

C:\Users\TEMP\AppData\Roaming\Microsoft\Windows\Start Menu\Programs>ollama run llama3.2

pulling manifest

pulling dde5aa3fc5ff: 100% ▕██████████████████████████████████████████████████████████▏ 2.0 GB

pulling 34bb5ab01051: 100% ▕██████████████████████████████████████████████████████████▏  561 B

verifying sha256 digest

writing manifest

success

> > > Hi
> > 
> > 
> 
> 

How can I assist you today?

### Run Streamlit Application

1. open cmd or terminal
2. cd E:\Products\mango-lassy-chat
3. streamlit run app.py

NOTE: Make sure your `.env` file contains your `SERPER_API_KEY`.

3. URL to open from browser:
http://localhost:8501/

### Sample Screens

## Prerequisites

1. **Python 3.10+** installed on your machine.
2. **Ollama** installed and running locally with `llama3.2` and `embeddinggemma` pulled.

---

## Install Python

* Install Python: Download and run the official installer from Python.org, ensuring pip is included.
* Add Python to your PATH:
1. Press the Windows key and type environment variables.
2. Click Edit the system environment variables, then click the Environment Variables button.
3. Under System variables or User variables, select Path and click Edit.
4. Add your Python installation directory and Scripts path.


* Restart your terminal: Close your current Command Prompt, PowerShell, or VS Code terminal window and open a fresh one.
* Verify installation: Type the following command to check if it works:

```bash
python --version
pip --version

```

## Install Git

* Step 1: Install Git
If you have not installed Git yet, download and run the installer from the Official Git Website. During setup, make sure to select the option that says "Git from the command line and also from 3rd-party software".
* Step 2: Add Git to your PATH (if already installed)
Add these two paths to your System Environment Variables under Path:
• C:\Program Files\Git\bin
• C:\Program Files\Git\cmd
* Step 3: Restart your terminal
Close your Command Prompt, PowerShell, or terminal window completely and open a new one.
* Step 4: Verify the fix
Type git --version to make sure it works, then run git init inside E:\Products\mango-lassy-chat

## Setup Git for project

1. git init
2. Add a .gitignore file so you don't upload large vector databases or keys

Create a file named .gitignore in the root folder (E:\Products\mango-lassy-chat.gitignore) and paste this inside:

chroma_db/
.env
**pycache**/
.DS_Store

3. git add .
4. git config --global user.email "tanu.kanala@gmail.com"
5. git commit -m "Initial commit: Mango Lassi Python Agentic RAG Chat"
6. git branch -M main
7. git remote add origin [https://github.com/tanukanala/mango-lassy-chat](https://github.com/tanukanala/mango-lassy-chat)
8. git push -u origin main

## Step 1: Project Directory Structure

Create your root folder and subdirectories:

```bash
mkdir mango-lassy-chat
cd mango-lassy-chat
mkdir documents

```

---

## Step 2: Install Python Dependencies

Navigate to your project root and install the required packages:

```bash
pip install streamlit langgraph langchain-ollama langchain-chroma langchain-core langchain-text-splitters langchain-community python-dotenv google-search-results presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_sm

```

---

## Step 3: Implement Guardrails (`guardrails.py`)

Create a file named **`guardrails.py`** in your project root (`E:\Products\mango-lassy-chat\guardrails.py`):

```python
import re

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    PRESIDIO_AVAILABLE = True
except Exception as e:
    PRESIDIO_AVAILABLE = False


def sanitize_text(text: str) -> str:
    """
    Sanitizes Personally Identifiable Information (PII) like emails,
    phone numbers, and student IDs.
    """
    if not text:
        return text

    sanitized = text

    if PRESIDIO_AVAILABLE:
        try:
            results = analyzer.analyze(
                text=sanitized,
                entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD"],
                language="en"
            )
            anonymized_result = anonymizer.anonymize(text=sanitized, analyzer_results=results)
            sanitized = anonymized_result.text
        except Exception as e:
            pass

    # Domain-specific Regex Safeguards for True Private PII
    sanitized = re.sub(r'\b\d{5,8}\b', '[REDACTED_ID]', sanitized)
    sanitized = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[REDACTED_EMAIL]', sanitized)
    sanitized = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[REDACTED_PHONE]', sanitized)

    return sanitized

```

---

## Step 4: Create Application Script (`app.py`)

Create your **`app.py`** file in the root directory:

```python
import os
import warnings
import streamlit as st
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv()
from guardrails import sanitize_text

st.set_page_config(
    page_title="MangoLassi RAG Assistant",
    page_icon="🥭",
    layout="centered"
)

st.header("✨ Mango🥭Lassi🥤RAG Agentic Chat ✨")
st.markdown("Powered by Python, Streamlit, LangGraph, ChromaDB, EmbeddingGemma, & Serper.dev")

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "embeddinggemma"  
LLM_MODEL = "llama3.2"
PERSIST_DIRECTORY = "./chroma_db"
DOCS_DIRECTORY = "./documents"

serper_search = GoogleSerperAPIWrapper()

@st.cache_resource
def load_embeddings_and_vector_store():
    emb_func = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vector_store = Chroma(
        collection_name="embeddinggemma_v3_guarded_collection",
        embedding_function=emb_func,
        persist_directory=PERSIST_DIRECTORY
    )
    
    if os.path.exists(DOCS_DIRECTORY):
        all_docs = []
        files_found = os.listdir(DOCS_DIRECTORY)
        for filename in files_found:
            if filename.endswith((".txt", ".md")):
                file_path = os.path.join(DOCS_DIRECTORY, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw_content = f.read()
                        sanitized_content = sanitize_text(raw_content)
                        doc_id = f"{filename}_{os.path.getmtime(file_path)}"
                        all_docs.append(Document(page_content=sanitized_content, metadata={"source": filename, "id": doc_id}))
                except Exception as e:
                    pass
        
        if all_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = text_splitter.split_documents(all_docs)
            texts = [d.page_content for d in split_docs]
            metadatas = [d.metadata for d in split_docs]
            ids = [f"{d.metadata['source']}_{i}" for i, d in enumerate(split_docs)]
            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    
    return vector_store

vector_store = load_embeddings_and_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 6})

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0.0,
    mirostat=0,
    base_url=OLLAMA_BASE_URL
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def route_intent(query: str) -> str:
    query_lower = query.lower()
    web_keywords = ["latest", "news", "current", "weather", "today", "search", "price", "stock", "who is", "what is happening", "google", "online", "recent", "days", "week", "month"]
    if any(keyword in query_lower for keyword in web_keywords):
        return "WEB"
        
    local_keywords = ["mango", "lassy", "lassi", "roadmap", "feature", "architecture", "assistant", "chat", "project", "doc", "document", "author", "phase", "stack", "implementation", "tanishka"]
    if any(keyword in query_lower for keyword in local_keywords):
        return "RAG"
        
    return "LLM"

def adaptive_router_node(state: State):
    latest_query = ""
    for msg in reversed(state["messages"]):
        msg_type = getattr(msg, "type", "")
        if msg_type in ("human", "user") or (isinstance(msg, dict) and msg.get("role") == "user"):
            if hasattr(msg, "content"):
                latest_query = msg.content.strip()
            elif isinstance(msg, dict):
                latest_query = msg.get("content", "").strip()
            break

    intent = route_intent(latest_query)
    
    if intent == "RAG":
        retrieved_docs = retriever.invoke(latest_query)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        system_prompt = (
            "You are an expert AI assistant. Answer the user's question accurately "
            "using ONLY the provided local context below. If the answer cannot be found "
            "in the context, state clearly that you don't know based on local documents.\n\n"
            f"--- LOCAL CONTEXT ---\n{context_text}\n----------------"
        )
        badge = "[RAG]"

    elif intent == "WEB":
        try:
            web_results = serper_search.run(latest_query)
        except Exception as e:
            web_results = f"Web search failed: {str(e)}"

        system_prompt = (
            "You are an expert web-connected AI assistant. Answer the user's question accurately "
            "using the live web search results provided below.\n\n"
            f"--- LIVE WEB RESULTS ---\n{web_results}\n----------------"
        )
        badge = "[WEB]"

    else:
        system_prompt = (
            "You are a helpful general-purpose AI assistant. Answer the user's question "
            "freely and accurately using your training knowledge base."
        )
        badge = "[LLM]"

    messages_with_context = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages_with_context)
    response.content = f"{badge} {response.content}"
    
    return {"messages": [response]}

def output_guardrail_node(state: State):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "content"):
        content_str = last_msg.content
        badge = ""
        for prefix in ("[RAG]", "[WEB]", "[LLM]"):
            if content_str.startswith(prefix):
                badge = prefix
                content_str = content_str[len(prefix):].strip()
                break
        
        sanitized_content = sanitize_text(content_str)
        last_msg.content = f"{badge} {sanitized_content}".strip()
        
    return {"messages": state["messages"]}

workflow = StateGraph(State)
workflow.add_node("router_agent", adaptive_router_node)
workflow.add_node("guardrail", output_guardrail_node)

workflow.add_edge(START, "router_agent")
workflow.add_edge("router_agent", "guardrail")
workflow.add_edge("guardrail", END)

app_graph = workflow.compile()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Mango Lassi assistant. Ask me about local docs, web search news, or anything else!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    graph_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            graph_messages.append(("user", msg["content"]))
        elif msg["role"] == "assistant":
            graph_messages.append(("assistant", msg["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = app_graph.invoke({"messages": graph_messages})
                ai_response = result["messages"][-1].content
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

```

---

## LangGraph Architecture Diagram

```mermaid
graph TD
    classDef startEnd fill:#f9f,stroke:#333,stroke-width:2px;
    classDef node fill:#bbf,stroke:#333,stroke-width:2px;
    classDef db fill:#fbb,stroke:#333,stroke-width:2px;

    START([START]) --> Router[<b>router_agent Node</b><br>• Ternary Intent Router: <code>[RAG]</code> | <code>[WEB]</code> | <code>[LLM]</code><br>• Injects Context / Search Results<br>• Invokes local LLM (llama3.2)]:::node
    
    Chroma[(ChromaDB<br>Vector Store)] -.->|Retrieved Chunks<br><code>[RAG]</code>| Router
    Serper[(Serper.dev<br>Web API)] -.->|Live Search Results<br><code>[WEB]</code>| Router

    Router --> Guardrail[<b>guardrail Node</b><br>• Extracts Response Badge<br>• Runs <code>sanitize_text()</code><br>• Strips PII / Preserves IDs]:::node

    Guardrail --> END([END])
    
    class START,END startEnd;

```

---

## Troubleshooting Model Errors

If you encounter connection or model errors, verify your Ollama status by running:

```bash
ollama list

```

And make sure both models are pulled:

```bash
ollama pull llama3.2
ollama pull embeddinggemma

```