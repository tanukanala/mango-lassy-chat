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

# Suppress harmless LangChain community deprecation warnings in console output
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.utilities import GoogleSerperAPIWrapper

# --- 0. Load Environment Variables ---
load_dotenv()

# Import Guardrail Sanitizer
from guardrails import sanitize_text

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="MangoLassi RAG Assistant",
    page_icon="🥭",
    layout="centered"
)

st.header("✨ Mango🥭Lassi🥤RAG Agentic Chat ✨")
st.markdown("Powered by Python, Streamlit, LangGraph, ChromaDB, EmbeddingGemma, & Serper.dev")

# --- 2. Initialize Models, Search Wrapper, & Vector Store ---
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "embeddinggemma"  
LLM_MODEL = "llama3.2"
PERSIST_DIRECTORY = "./chroma_db"
DOCS_DIRECTORY = "./documents"

# Initialize GoogleSerperAPIWrapper (reads SERPER_API_KEY from .env)
serper_search = GoogleSerperAPIWrapper()

@st.cache_resource
def load_embeddings_and_vector_store():
    """
    Cached resource function: Initializes models and runs ingestion ONLY ONCE
    on startup, applying PII guardrails to raw documents before vector store insertion.
    """
    print("[INIT] Initializing Ollama Embeddings model:", EMBEDDING_MODEL)
    emb_func = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    print("\n------------------------------------------------------------")
    print("[VECTOR STORE] Initializing Chroma vector store at:", PERSIST_DIRECTORY)
    
    vector_store = Chroma(
        collection_name="embeddinggemma_v3_guarded_collection",
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
                        raw_content = f.read()
                        
                        # Apply Ingestion Guardrail
                        sanitized_content = sanitize_text(raw_content)
                        print(f"[GUARDRAIL] Applied PII sanitization to '{filename}'")
                        
                        doc_id = f"{filename}_{os.path.getmtime(file_path)}"
                        all_docs.append(Document(page_content=sanitized_content, metadata={"source": filename, "id": doc_id}))
                        print(f"[INGESTION] Successfully read '{filename}' ({len(sanitized_content)} chars). Generated Doc ID: {doc_id}")
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
            print(f"[EMBEDDING] Successfully upserted all guarded chunks into vector store!")
        else:
            print(f"[INGESTION] No valid .txt or .md documents found to ingest.")
    else:
        print(f"[WARNING] Documents directory '{DOCS_DIRECTORY}' does not exist.")
    
    print("------------------------------------------------------------\n")
    return vector_store

# Load vector store and setup retriever with k=6
vector_store = load_embeddings_and_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 6})

print("[INIT] Initializing local ChatOllama model:", LLM_MODEL)
llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0.0,
    mirostat=0,
    base_url=OLLAMA_BASE_URL
)

# --- 3. Define LangGraph State, Nodes, & Ternary Router Workflow ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

def route_intent(query: str) -> str:
    """
    Ternary Intent Router: Determines whether the query requires [RAG], [WEB], or [LLM].
    """
    query_lower = query.lower()
    
    # 1. Real-time / recent / web search keywords take priority -> WEB
    web_keywords = [
        "latest", "news", "current", "weather", "today", "search", 
        "price", "stock", "who is", "what is happening", "google", "online",
        "recent", "days", "week", "month"
    ]
    if any(keyword in query_lower for keyword in web_keywords):
        return "WEB"
        
    # 2. Local workspace / document keywords -> RAG
    local_keywords = [
        "mango", "lassy", "lassi", "roadmap", "feature", "architecture", 
        "assistant", "chat", "project", "doc", "document", "author", 
        "phase", "stack", "implementation", "tanishka"
    ]
    if any(keyword in query_lower for keyword in local_keywords):
        return "RAG"
        
    # 3. Default fallback -> LLM reasoning
    return "LLM"

def adaptive_router_node(state: State):
    """
    Graph Node 1: Routes query to [RAG], [WEB], or [LLM] based on intent.
    """
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
        print(f"\n[ROUTER] Intent matched: [RAG] | Query: '{latest_query}'")
        retrieved_docs = retriever.invoke(latest_query)
        print(f"[RAG SEARCH] Retrieved {len(retrieved_docs)} relevant document chunk(s).")
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        system_prompt = (
            "You are an expert AI assistant. Answer the user's question accurately "
            "using ONLY the provided local context below. If the answer cannot be found "
            "in the context, state clearly that you don't know based on local documents.\n\n"
            f"--- LOCAL CONTEXT ---\n{context_text}\n----------------"
        )
        badge = "[RAG]"

    elif intent == "WEB":
        print(f"\n[ROUTER] Intent matched: [WEB] (Serper.dev) | Query: '{latest_query}'")
        try:
            web_results = serper_search.run(latest_query)
            print(f"[WEB SEARCH] Serper search completed successfully.")
        except Exception as e:
            web_results = f"Web search failed: {str(e)}"
            print(f"[ERROR] Serper search failed: {e}")

        system_prompt = (
            "You are an expert web-connected AI assistant. Answer the user's question accurately "
            "using the live web search results provided below.\n\n"
            f"--- LIVE WEB RESULTS ---\n{web_results}\n----------------"
        )
        badge = "[WEB]"

    else:
        print(f"\n[ROUTER] Intent matched: [LLM] | Query: '{latest_query}'")
        system_prompt = (
            "You are a helpful general-purpose AI assistant. Answer the user's question "
            "freely and accurately using your training knowledge base."
        )
        badge = "[LLM]"

    messages_with_context = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    print(f"[LLM INFERENCE] Generating response with {badge} mode...")
    response = llm.invoke(messages_with_context)
    
    response.content = f"{badge} {response.content}"
    print(f"[LLM INFERENCE] Response received successfully.")
    
    return {"messages": [response]}

def output_guardrail_node(state: State):
    """
    Graph Node 2 (Output Guardrail): Sanitizes final response text.
    """
    print("[GUARDRAIL NODE] Checking output response for PII...")
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

# Build LangGraph Workflow
workflow = StateGraph(State)
workflow.add_node("router_agent", adaptive_router_node)
workflow.add_node("guardrail", output_guardrail_node)

workflow.add_edge(START, "router_agent")
workflow.add_edge("router_agent", "guardrail")
workflow.add_edge("guardrail", END)

app_graph = workflow.compile()

# --- 4. Streamlit Chat Interface Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Mango Lassi assistant. Ask me about local docs, web search news, or anything else!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. Handle User Input & Execution ---
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
                print(f"[ERROR] {error_msg}")
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})