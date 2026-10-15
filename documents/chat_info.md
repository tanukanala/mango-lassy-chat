================================================================================
## MANGO LASSI CHAT ASSISTANT: COMPREHENSIVE ARCHITECTURE, IMPLEMENTATION, AND ROADMAP
================================================================================

### 1. EXECUTIVE SUMMARY & SYSTEM OVERVIEW
--------------------------------------------------------------------------------
The Mango Lassi Chat Assistant is a modern, privacy-first, fully local generative AI chat application built on Python. Designed to mirror the responsiveness, visual styling, and interactivity of leading cloud-based AI interfaces like ChatGPT and Gemini, it operates entirely offline on local hardware without sending user data, prompt history, or proprietary documents to third-party cloud providers.

Core Technological Stack:
- Frontend & UI: Streamlit (provides native responsive chat bubbles, session state management, and real-time streaming feedback).
- Orchestration & Workflow: LangGraph (manages agentic state graphs, conversation loops, and modular node-based execution).
- Large Language Model: Ollama running local weights (e.g., Llama 3.2, configured with deterministic temperature settings and low latency).
- Retrieval-Augmented Generation (RAG): ChromaDB (an open-source persistent vector database storing chunked document embeddings locally).
- Embedding Engine: EmbeddingGemma (via Ollama embeddings, transforming local documentation into precise vector space vectors for semantic similarity search).

---

### 2. SYSTEM ARCHITECTURE & DATA FLOW
--------------------------------------------------------------------------------
The application utilizes a directed cyclic graph pattern powered by LangGraph, separating interface rendering, document retrieval, and text generation into decoupled nodes.

Data Flow Sequence:
1. User Input: The user types a prompt into the Streamlit chat interface input box.
2. State Update: The user prompt is appended to `st.session_state.messages` to preserve chat history across reruns.
3. Vector Similarity Search (RAG Node): 
   - The system isolates the latest user query.
   - The query is vectorized using EmbeddingGemma.
   - ChromaDB performs a cosine similarity lookup against pre-chunked local documents stored in `./chroma_db`.
   - The top matching chunks (k=3) are extracted.
4. Context Injection: A system prompt is dynamically constructed containing the retrieved document chunks as strict context.
5. LLM Inference: The system prompt, along with the prior conversation history, is routed to the local Ollama LLM node (`ChatOllama`).
6. Response Rendering: The generated response is streamed/rendered back to the Streamlit UI, markdown-formatted, and saved to the session state history.

---

### 3. DETAILED IMPLEMENTATION GUIDE
--------------------------------------------------------------------------------
The application is consolidated into a single clean Python script (`app.py`), interacting with a local Ollama server (`http://localhost:11434`) and a local vector store directory (`./chroma_db`).

#### 3.1 Page Configuration & Setup
- Configures Streamlit page metadata, custom icons (🥭), and centered layout configuration to deliver an optimized reading experience.

#### 3.2 Local Model Initialization
- Instantiates OllamaEmbeddings using the "embeddinggemma" model for semantic vector generation.
- Instantiates ChatOllama using "llama3.2" with temperature set to 0.0 to eliminate hallucination randomness and guarantee precise context adherence.

#### 3.3 Automated Document Ingestion Pipeline
- Scans the `./documents` folder for text (`.txt`) and markdown (`.md`) files.
- Splits files dynamically using `RecursiveCharacterTextSplitter` (chunk size: 500, overlap: 50) to optimize context window sizing.
- Generates deterministic unique identifiers based on filenames and modification timestamps.
- Upserts chunks into ChromaDB, allowing instantaneous sync when files are modified or added without duplicating vector entries.

#### 3.4 LangGraph State Graph
- Defines a TypedDict schema holding message history (`add_messages` reducer).
- Implements the `retrieve_and_generate` node responsible for combining RAG context retrieval and LLM completion.

---

### 4. DETAILED FEATURE BREAKDOWN
--------------------------------------------------------------------------------
- Full Markdown & Code Syntax Highlighting: Automatically formats tables, bullet points, headers, bold text, and color-codes code snippets for Python, JavaScript, and shell commands.
- 100% Offline & Private: No internet connectivity required after initial model pulls; total data sovereignty.
- Persistent Vector Storage: Documents are stored on disk in `./chroma_db`, ensuring fast startup times and persistence across app restarts.
- Dynamic Document Sync: Adding new `.txt` or `.md` files to the `documents/` folder automatically updates the vector index on reload.
- Deterministic Guardrails: Configured system instructions restrict the model to answering queries strictly using provided document context, minimizing hallucinations.

---

### 5. COMPREHENSIVE ROADMAP (PHASES 1 TO 4)
--------------------------------------------------------------------------------
Phase 1: Foundation & Local UI (Completed)
- Node.js/Express backend with React frontend migrated to a lightweight, highly efficient Python Streamlit + LangGraph implementation.

Phase 2: Local RAG & Vector Database Integration (Completed)
- ChromaDB vector store integration with EmbeddingGemma and Recursive Character text chunking for local document ingestion.

Phase 3: Advanced Tool Calling & Web Search (Upcoming)
- Integration of Tavily Search API (`TavilySearchResults`) as a secondary tool node inside LangGraph.
- Implementation of agentic decision routing (allowing the model to choose between local RAG retrieval, web search, or pure LLM reasoning).

Phase 4: Document Upload UI & Multi-Format Support (Future)
- Adding a Streamlit file uploader widget in the sidebar so users can upload PDF, DOCX, and TXT files directly through the browser UI.
- Automatic background parsing, chunking, and embedding of uploaded files into the active Chroma collection.
- Persistent session history management using SQLite checkpoints for LangGraph.

<eof>