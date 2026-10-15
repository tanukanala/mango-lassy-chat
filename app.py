import os
import streamlit as st
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="MangoLassi RAG Assistant",
    page_icon="🥭",
    layout="centered"
)

st.header("✨ Mango🥭Lassi RAG Agentic Chat ✨")
st.markdown("Powered by Python, Streamlit, LangGraph, ChromaDB, & EmbeddingGemma")

# --- 2. Initialize Models & Cached Vector Store ---
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "embeddinggemma"  
LLM_MODEL = "llama3.2"
PERSIST_DIRECTORY = "./chroma_db"
DOCS_DIRECTORY = "./documents"

@st.cache_resource
def load_embeddings_and_vector_store():
    """
    Cached resource function: Initializes models and runs ingestion ONLY ONCE
    on startup, avoiding repeated terminal logs and heavy re-computations on reruns.
    """
    print("[INIT] Initializing Ollama Embeddings model:", EMBEDDING_MODEL)
    emb_func = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    print("\n------------------------------------------------------------")
    print("[VECTOR STORE] Initializing Chroma vector store at:", PERSIST_DIRECTORY)
    vector_store = Chroma(
        collection_name="embeddinggemma_v2_collection",
        embedding_function=emb_func,
        persist_directory=PERSIST_DIRECTORY
    )
    
    print(f"[INGESTION] Checking documents directory: {DOCS_DIRECTORY}")
    if os.path.exists(DOCS_DIRECTORY):
        all_docs = []
        files_found = os.listdir(DOCS_DIRECTORY)
        print(f"[INGESTION] Found files in directory: {files_found}")
        
        for filename in files_found:
            if filename.endswith((".txt", ".md")):
                file_path = os.path.join(DOCS_DIRECTORY, filename)
                print(f"[INGESTION] Processing file: {filename}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        doc_id = f"{filename}_{os.path.getmtime(file_path)}"
                        all_docs.append(Document(page_content=content, metadata={"source": filename, "id": doc_id}))
                        print(f"[INGESTION] Successfully read '{filename}' ({len(content)} characters). Generated Doc ID: {doc_id}")
                except Exception as e:
                    print(f"[ERROR] Error reading file {filename}: {e}")
        
        if all_docs:
            print(f"[SPLITTER] Initializing RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50)")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = text_splitter.split_documents(all_docs)
            print(f"[SPLITTER] Split {len(all_docs)} source document(s) into {len(split_docs)} total chunks.")
            
            texts = [d.page_content for d in split_docs]
            metadatas = [d.metadata for d in split_docs]
            ids = [f"{d.metadata['source']}_{i}" for i, d in enumerate(split_docs)]
            
            print(f"[EMBEDDING] Generating embeddings via EmbeddingGemma and upserting {len(texts)} chunks into ChromaDB...")
            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            print(f"[EMBEDDING] Successfully upserted all chunks into vector store!")
        else:
            print(f"[INGESTION] No valid .txt or .md documents found to ingest.")
    else:
        print(f"[WARNING] Documents directory '{DOCS_DIRECTORY}' does not exist.")
    
    print("------------------------------------------------------------\n")
    return vector_store

# Load vector store and setup retriever once using caching
vector_store = load_embeddings_and_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

print("[INIT] Initializing local ChatOllama model:", LLM_MODEL)
llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0.0,
    mirostat=0,
    base_url=OLLAMA_BASE_URL
)

# --- 3. Define LangGraph State & RAG Node ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

def retrieve_and_generate(state: State):
    """
    Graph Node: Queries ChromaDB using the actual latest user message,
    injects context into system prompt, and calls the local LLM.
    """
    latest_query = ""
    # Robust loop to safely extract content from message instances or dicts
    for msg in reversed(state["messages"]):
        msg_type = getattr(msg, "type", "")
        if msg_type in ("human", "user") or (isinstance(msg, dict) and msg.get("role") == "user"):
            if hasattr(msg, "content"):
                latest_query = msg.content.strip()
            elif isinstance(msg, dict):
                latest_query = msg.get("content", "").strip()
            break

    print(f"\n[RAG SEARCH] User Query: '{latest_query}'")
    
    context_text = ""
    if latest_query:
        retrieved_docs = retriever.invoke(latest_query)
        print(f"[RAG SEARCH] Retrieved {len(retrieved_docs)} relevant document chunk(s) from ChromaDB.")
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    else:
        print("[RAG SEARCH] Query was empty, skipping document retrieval.")

    # Construct system prompt with retrieved context
    system_prompt = (
        "You are an expert AI assistant. Answer the user's question accurately "
        "using ONLY the provided context below. If the answer cannot be found "
        "in the context, state clearly that you don't know based on local documents.\n\n"
        f"--- CONTEXT ---\n{context_text}\n----------------"
    )

    messages_with_context = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    print(f"[LLM INFERENCE] Sending context and query to local LLM ({LLM_MODEL})...")
    response = llm.invoke(messages_with_context)
    print(f"[LLM INFERENCE] Response received successfully.")
    
    return {"messages": [response]}

# Build LangGraph Workflow
workflow = StateGraph(State)
workflow.add_node("rag_agent", retrieve_and_generate)
workflow.add_edge(START, "rag_agent")
workflow.add_edge("rag_agent", END)

app_graph = workflow.compile()

# --- 4. Streamlit Chat Interface Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your EmbeddingGemma RAG assistant. Ask me anything about your documents!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. Handle User Input & Execution ---
if prompt := st.chat_input("Ask something about your documents..."):
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
        with st.spinner("Searching documents with EmbeddingGemma & thinking..."):
            try:
                result = app_graph.invoke({"messages": graph_messages})
                ai_response = result["messages"][-1].content
                
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                print(f"[ERROR] {error_msg}")
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})