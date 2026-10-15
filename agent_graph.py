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

# Suppress harmless LangChain community deprecation warnings in console output
warnings.filterwarnings("ignore", category=DeprecationWarning)

from guardrails import sanitize_text
from site_selector import site_selection_graph

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "embeddinggemma"  
LLM_MODEL = "llama3.2"
PERSIST_DIRECTORY = "./chroma_db"
DOCS_DIRECTORY = "./documents"

serper_search = GoogleSerperAPIWrapper()

@st.cache_resource
def load_embeddings_and_vector_store():
    print("[INIT] Initializing Ollama Embeddings model:", EMBEDDING_MODEL)
    emb_func = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    
    vector_store = Chroma(
        collection_name="embeddinggemma_v3_guarded_collection",
        embedding_function=emb_func,
        persist_directory=PERSIST_DIRECTORY
    )
    
    print(f"[INGESTION] Checking documents directory: {DOCS_DIRECTORY}")
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
                    print(f"[ERROR] Error reading file {filename}: {e}")
        
        if all_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = text_splitter.split_documents(all_docs)
            texts = [d.page_content for d in split_docs]
            metadatas = [d.metadata for d in split_docs]
            ids = [f"{d.metadata['source']}_{i}" for i, d in enumerate(split_docs)]
            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            print("[EMBEDDING] Successfully upserted all guarded chunks into vector store!")
            
    return vector_store

vector_store = load_embeddings_and_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 6})

llm = ChatOllama(model=LLM_MODEL, temperature=0.0, mirostat=0, base_url=OLLAMA_BASE_URL)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def route_intent(query: str) -> str:
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
    """
    Robust parameter extractor with granular debug prints to trace execution speed and points of failure.
    """
    print(f"\n[DEBUG EXTRACT] Entering extract_site_parameters() for query: '{query}'")
    
    extraction_prompt = f"""
    You are a precise data extraction assistant for commercial real estate and data center site selection.
    Analyze the user query below and extract:
    1. "country": The target country (default to "United States" if unspecified).
    2. "locations": A list of specific locations (cities, states, or regions).
       - RULE A: If the user explicitly mentions specific locations (e.g. "Dallas", "London", "Ashburn"), extract those exact locations.
       - RULE B: If the user asks a general question for a country without naming specific locations (e.g. "which united state and location is best...", "best place for data center in US"), output the top 3 premier commercial real estate / data center locations for that country (e.g. for United States: ["Northern Virginia", "Phoenix AZ", "Columbus OH"]).
    
    Return ONLY a valid JSON object with keys "country" (string) and "locations" (list of strings). No extra text or markdown formatting.
    
    User Query: "{query}"
    
    JSON Output:
    """
    
    print(f"[DEBUG EXTRACT] Sending prompt to local LLM ({LLM_MODEL})... (waiting for response)")
    try:
        response = llm.invoke(extraction_prompt)
        content = response.content.strip()
        print(f"[DEBUG EXTRACT] Received response from LLM successfully. Raw content:\n{content}\n")
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            print(f"[DEBUG EXTRACT] Extracted JSON string via regex: {json_str}")
            data = json.loads(json_str)
            country = data.get("country", "United States")
            locations = data.get("locations", ["Northern Virginia", "Phoenix AZ", "Columbus OH"])
            if not locations:
                locations = ["Northern Virginia", "Phoenix AZ", "Columbus OH"]
            print(f"[DEBUG EXTRACT] Successfully parsed JSON -> Country: {country}, Locations: {locations}")
            return country, locations
        else:
            print("[DEBUG EXTRACT WARNING] No JSON bracket found in LLM response text.")
    except Exception as e:
        print(f"[DEBUG EXTRACT ERROR] Exception encountered during LLM invocation or JSON parsing: {e}")

    # Deterministic Fallback Rules
    print("[DEBUG EXTRACT] Falling back to deterministic rules...")
    query_lower = query.lower()
    if "uk" in query_lower or "united kingdom" in query_lower or "london" in query_lower:
        fallback = ("United Kingdom", ["London", "Slough", "Dublin"])
    elif "germany" in query_lower or "frankfurt" in query_lower:
        fallback = ("Germany", ["Frankfurt", "Berlin", "Munich"])
    elif "japan" in query_lower or "tokyo" in query_lower:
        fallback = ("Japan", ["Tokyo", "Osaka", "Nagoya"])
    else:
        fallback = ("United States", ["Northern Virginia", "Phoenix AZ", "Columbus OH"])
        
    print(f"[DEBUG EXTRACT] Fallback result -> Country: {fallback[0]}, Locations: {fallback[1]}")
    return fallback

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
    
    if intent == "SITE_SELECTION":
        print(f"\n[ROUTER] Intent matched: [SITE_SELECTION] | Query: '{latest_query}'")
        
        country, locations = extract_site_parameters(latest_query)
        print(f"[EXTRACTED PARAMS] Country: {country} | Locations: {locations}")
        
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
        print(f"\n[ROUTER] Intent matched: [RAG] | Query: '{latest_query}'")
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
        print(f"\n[ROUTER] Intent matched: [WEB] (Serper.dev) | Query: '{latest_query}'")
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
        print(f"\n[ROUTER] Intent matched: [LLM] | Query: '{latest_query}'")
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
        for prefix in ("[RAG]", "[WEB]", "[LLM]", "[SITE SELECTION]"):
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