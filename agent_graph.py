import os
import re
import json
import warnings
import streamlit as st
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import AIMessage

warnings.filterwarnings("ignore", category=DeprecationWarning)
from guardrails import sanitize_text
from logger import log_status, set_status_callback
from site_selector import site_selection_graph

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "embeddinggemma"  
LLM_MODEL = "llama3.2"
PERSIST_DIRECTORY = "./chroma_db"
DOCS_DIRECTORY = "./documents"

serper_search = GoogleSerperAPIWrapper()

@st.cache_resource
def load_embeddings_and_vector_store():
    """Create the embedding model and prepare the persistent document index.

    Supported text files are read from the documents folder, sanitized for PII,
    split into overlapping chunks, and added to Chroma for later retrieval.
    Streamlit caches the resulting vector store so it is not rebuilt per rerun.
    """
    log_status(f"Initializing Ollama Embeddings model: {EMBEDDING_MODEL}")
    emb_func = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    
    vector_store = Chroma(
        collection_name="embeddinggemma_v3_guarded_collection",
        embedding_function=emb_func,
        persist_directory=PERSIST_DIRECTORY
    )
    
    log_status(f"Checking documents directory: {DOCS_DIRECTORY}")
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
                    log_status(f"Error reading file {filename}: {e}")
        
        if all_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = text_splitter.split_documents(all_docs)
            texts = [d.page_content for d in split_docs]
            metadatas = [d.metadata for d in split_docs]
            ids = [f"{d.metadata['source']}_{i}" for i, d in enumerate(split_docs)]
            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            log_status("Successfully upserted guarded chunks into vector store.")
            
    return vector_store

vector_store = load_embeddings_and_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 6})

llm = ChatOllama(model=LLM_MODEL, temperature=0.0, mirostat=0, base_url=OLLAMA_BASE_URL)

class State(TypedDict):
    """Conversation state passed between nodes in the assistant graph."""

    messages: Annotated[list, add_messages]

def route_intent(query: str) -> str:
    """Choose a processing path by checking the query against keyword groups.

    Site-selection phrases take precedence, followed by live-web terms and
    project/document terms; anything unmatched uses general LLM reasoning.
    """
    query_lower = query.lower()
    if "site selection" in query_lower or "data center" in query_lower or "commercial real estate" in query_lower:
        return "SITE_SELECTION"
    web_keywords = ["latest", "news", "current", "weather", "today", "search", "price", "stock", "who is", "what is happening", "google", "online", "recent", "days", "week", "month"]
    if any(keyword in query_lower for keyword in web_keywords):
        return "WEB"
    local_keywords = ["mango", "lassy", "lassi", "roadmap", "feature", "architecture", "assistant", "chat", "project", "doc", "document", "author", "phase", "stack", "implementation", "tanishka"]
    if any(keyword in query_lower for keyword in local_keywords):
        return "RAG"
    return "LLM"

def extract_site_parameters(query: str) -> tuple:
    """Extract the requested country and locations for a site-selection run.

    The local LLM is asked for JSON first. If it fails or returns no usable
    object, country keywords select a small predefined set of candidate sites.
    """
    log_status("Analyzing user query for site selection parameters...")
    extraction_prompt = f"""
    You are a precise data extraction assistant for commercial real estate and data center site selection.
    Analyze the user query below and extract:
    1. "country": The target country (default to "United States" if unspecified).
    2. "locations": A list of specific locations (cities, states, or regions).
       - RULE A: If the user explicitly mentions specific locations (e.g. "Dallas", "London", "San Jose"), extract those exact locations.
       - RULE B: If the user asks a general question for a country without naming specific locations, output the top 3 premier commercial real estate / data center locations for that country (e.g. for United States: ["Northern Virginia", "Phoenix AZ", "Columbus OH"]).
    
    Return ONLY a valid JSON object with keys "country" (string) and "locations" (list of strings). No extra text or markdown formatting.
    
    User Query: "{query}"
    
    JSON Output:
    """
    log_status("Querying local LLM for parameter extraction...")
    try:
        response = llm.invoke(extraction_prompt)
        content = response.content.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            country = data.get("country", "United States")
            locations = data.get("locations", ["Northern Virginia", "Phoenix AZ", "Columbus OH"])
            if not locations:
                locations = ["Northern Virginia", "Phoenix AZ", "Columbus OH"]
            log_status(f"Target Country: {country} | Locations: {', '.join(locations)}")
            return country, locations
    except Exception as e:
        log_status(f"Extraction warning: {e}. Using fallback.")

    query_lower = query.lower()
    if "uk" in query_lower or "united kingdom" in query_lower or "london" in query_lower:
        fallback = ("United Kingdom", ["London", "Slough", "Dublin"])
    elif "germany" in query_lower or "frankfurt" in query_lower:
        fallback = ("Germany", ["Frankfurt", "Berlin", "Munich"])
    elif "japan" in query_lower or "tokyo" in query_lower:
        fallback = ("Japan", ["Tokyo", "Osaka", "Nagoya"])
    else:
        fallback = ("United States", ["Northern Virginia", "Phoenix AZ", "Columbus OH"])
    log_status(f"Target Country: {fallback[0]} | Locations: {', '.join(fallback[1])}")
    return fallback

def adaptive_router_node(state: State):
    """Route the latest user message and produce a mode-specific response.

    Site-selection requests run their dedicated research graph immediately.
    Other intents build a system prompt (with retrieved or searched context
    when applicable), call the chat model, and label the response with its mode.
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
    
    if intent == "SITE_SELECTION":
        log_status("Intent matched: [SITE_SELECTION]")
        country, locations = extract_site_parameters(latest_query)
        
        result = site_selection_graph.invoke({
            "country": country,
            "locations": locations,
            "raw_data": {},
            "evaluations": {},
            "comparison_report": "",
            "report_filepath": ""
        })
        
        badge = "[SITE SELECTION]"
        response_text = (
            f"### 🏢 Commercial Real Estate & Data Center Site Selection Report\n\n"
            f"**Locations Evaluated**: {', '.join(locations)} ({country})\n"
            f"**Multi-Agent Status**: Successfully synthesized via Hierarchical LangGraph Supervisor & Sub-Agents.\n\n"
            f"#### Summary & Comparison:\n{result['comparison_report']}\n\n"
            f"---\n*📁 **Report Saved Successfully** to: `{result['report_filepath']}`*"
        )
        return {"messages": [AIMessage(content=f"{badge} {response_text}")]}

    elif intent == "RAG":
        log_status("Intent matched: [RAG] (Retrieving local context)")
        retrieved_docs = retriever.invoke(latest_query)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Clean system prompt without dumping header tags into user output
        system_prompt = (
            "You are a helpful, professional AI assistant. "
            "Answer the user's question accurately, naturally, and concisely using ONLY "
            "the provided local context below. Do NOT output raw system headers, chunk tags, "
            "or '<eof>' tokens in your response.\n\n"
            f"--- LOCAL CONTEXT ---\n{context_text}\n----------------"
        )
        badge = "[RAG]"

    elif intent == "WEB":
        log_status("Intent matched: [WEB] (Running Serper.dev search)")
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
        log_status("Intent matched: [LLM] (General reasoning)")
        system_prompt = (
            "You are a helpful general-purpose AI assistant. Answer the user's question "
            "freely and accurately using your training knowledge base."
        )
        badge = "[LLM]"

    messages_with_context = [{"role": "system", "content": system_prompt}] + state["messages"]
    log_status(f"Generating response in {badge} mode...")
    response = llm.invoke(messages_with_context)
    response.content = f"{badge} {response.content}"
    log_status("Response generated successfully.")
    
    return {"messages": [response]}

def output_guardrail_node(state: State):
    """Sanitize the latest assistant response while preserving its mode badge."""
    log_status("Checking output response against PII guardrails...")
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "content"):
        content_str = last_msg.content
        badge = ""
        for prefix in ("[RAG]", "[WEB]", "[LLM]", "[SITE SELECTION]"):
            if content_str.startswith(prefix):
                badge = prefix
                content_str = content_str[len(prefix):].strip()
                break
        sanitized_content = sanitize_text(content_str)
        last_msg.content = f"{badge} {sanitized_content}".strip()
    return {"messages": state["messages"]}

# The main conversation graph routes each request first, then sanitizes the
# generated response before returning it to the Streamlit interface.
workflow = StateGraph(State)
workflow.add_node("router_agent", adaptive_router_node)
workflow.add_node("guardrail", output_guardrail_node)

workflow.add_edge(START, "router_agent")
workflow.add_edge("router_agent", "guardrail")
workflow.add_edge("guardrail", END)

app_graph = workflow.compile()