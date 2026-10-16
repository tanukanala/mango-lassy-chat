# 🥭 Mango Lassi: Agentic RAG Chat Assistant

**Developed by:** Tanishka Sree Kanala (Tanu)  
**Grade:** 10th Grade, Coppell High School, Dallas, Texas  
**Project Category:** Artificial Intelligence & Software Engineering  

---

## 📌 Project Overview
Mango Lassi is a high-performance, privacy-conscious AI assistant built using modern agentic software architecture. The assistant provides intelligent retrieval-augmented generation (RAG) over local documents, live web search capabilities, and a multi-agent system designed for commercial real estate and infrastructure site selection.

---

## 🚀 Technical Highlights
* **Agentic Graph Workflow (`LangGraph`)**: Implements an adaptive intent router that dynamically directs user queries across four modes: Local Document Retrieval (`[RAG]`), Live Web Search (`[WEB]`), Multi-Agent Site Analysis (`[SITE_SELECTION]`), and General Reasoning (`[LLM]`).
* **Multi-Agent Infrastructure Engine**: Features a supervisor node that delegates research to specialized sub-agents to evaluate zoning, power grid reliability, and tax incentives across global candidate sites.
* **Dual-Layer Guardrails & Privacy**: Applies automated PII (Personally Identifiable Information) sanitization during document ingestion and response generation to protect sensitive user data.
* **In-Chat Telemetry & Streaming UI**: Built with Streamlit, providing real-time execution tracing and workflow status updates (`st.status`) directly within the chat window.

---

## 🛠️ Technology Stack
* **Language & Frameworks**: Python, Streamlit, LangGraph, LangChain
* **AI Models**: Local Ollama (`llama3.2`, `embeddinggemma`)
* **Vector Store**: ChromaDB
* **APIs**: Serper.dev Google Search API Wrapper

## 1. Executive Summary & Project Overview

**Mango Lassi** is an enterprise-grade, privacy-conscious multi-agent AI assistant designed for **Commercial Real Estate (CRE) & Data Center Site Selection**, document-based Retrieval-Augmented Generation (RAG), real-time web search integration, and general AI reasoning.

Built on top of **LangGraph**, **Streamlit**, and local Large Language Models (LLMs) via **Ollama**, Mango Lassi features dual-layer Personally Identifiable Information (PII) guardrails, hierarchical multi-agent graph workflows, and real-time live execution telemetry streamed directly to the user interface.

---

## 2. Key Features

### 🏢 Hierarchical Multi-Agent Site Selection Engine

* **Supervisor-Worker Architecture**: Uses a Lead Investment Director supervisor node that orchestrates parallel research tasks across dedicated sub-agents.
* **Triple-Domain Analysis**: Automatically researches and evaluates candidate sites across three critical vectors:
1. **Zoning & Environmental Regulations**
2. **Power Grid Capacity & Renewable Infrastructure**
3. **Tax Incentives & Financial Abatements**


* **Automated Comparative Reporting**: Synthesizes quantitative scoring (out of 10) and qualitative pros/cons into structured Markdown Investment Memos automatically exported to disk.

### 🎯 Adaptive Ternary Intent Router

* Intelligently inspects incoming user queries and routes them to the optimal execution path:
* **`[SITE_SELECTION]`**: Triggers the multi-agent graph for site analysis and report generation.
* **`[RAG]`**: Queries local vector database for workspace and technical project documentation.
* **`[WEB]`**: Executes live internet searches via Serper.dev for current news, market prices, and dynamic data.
* **`[LLM]`**: Falls back to local general-purpose LLM reasoning.



### 🛡️ Dual-Layer PII Protection (Guardrails)

* **Ingestion Guardrail**: Automatically sanitizes sensitive data (emails, phone numbers, SSNs, names) from local `.txt` and `.md` files prior to chunking and embedding in the vector store.
* **Output Guardrail**: Validates and redacts final AI responses before rendering them in the user interface.

### 📡 Live In-Chat Telemetry & Execution Tracing

* Streams real-time sub-agent execution logs (parameter parsing, worker delegation, scoring, and report saving) directly inside an expandable Streamlit `st.status()` container within the chat bubble.

---

## 3. System Architecture & Tech Stack

### Technology Stack

| Layer | Technology / Library | Purpose |
| --- | --- | --- |
| **User Interface** | Streamlit | Lightweight web interface with `st.status` live telemetry container |
| **Agentic Framework** | LangGraph & LangChain | Graph-based state orchestration and multi-agent coordination |
| **Local LLM & Embeddings** | Ollama (`llama3.2`, `embeddinggemma`) | On-device, privacy-preserving LLM inference and vector embedding generation |
| **Vector Database** | ChromaDB | Local persistent vector database for document chunk retrieval |
| **Search Engine** | Google Serper.dev API Wrapper | Live web search capabilities for real-time site data and news |
| **Security & Privacy** | Custom Presidio / Regex Sanitizer | PII redaction during ingestion and response generation |

---

## 4. Architecture Diagram

```text
                               ┌───────────────────────────┐
                               │   Streamlit UI (`app.py`) │
                               │  - Chat Input / History   │
                               │  - Live st.status() Feed  │
                               └─────────────┬─────────────┘
                                             │ Invokes app_graph.invoke()
                                             ▼
                               ┌───────────────────────────┐
                               │ Main LangGraph StateGraph │
                               │   (State: messages)       │
                               └─────────────┬─────────────┘
                                             │ START
                                             ▼
                               ┌───────────────────────────┐
                               │    `router_agent` Node    │
                               │  (Adaptive Intent Router) │
                               └─────────────┬─────────────┘
                                             │
      ┌──────────────────┬───────────────────┼───────────────────┐
      │                  │                   │                   │
      ▼ [SITE_SELECTION] ▼ [RAG]             ▼ [WEB]             ▼ [LLM]
┌──────────────────┐┌────────────────┐┌────────────────┐┌──────────────────┐
│ site_selector.py ││ ChromaDB +     ││ Google Serper  ││ Direct Ollama    │
│ StateGraph:      ││ EmbeddingGemma ││ API Wrapper    ││ LLM Inference    │
│                  ││ Vector Store   ││ (Live Search)  │└──────────────────┘
│ 1. supervisor    ││ (Guarded Chunks│└────────────────┘          
│    (Serper API)  ││  via Guardrails│                            
│ 2. evaluator     ││  Module)       │                            
│    (LLM Scoring) │└────────────────┘                            
│ 3. synthesizer   │                                              
│    (Markdown Memo│                                              
│     Export)      │                                              
└─────────┬────────┘                                              
          │                                                       
          └──────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
                       ┌───────────────────────────┐
                       │     `guardrail` Node      │
                       │   (Output PII Sanitizer   │
                       │    via `guardrails.py`)   │
                       └─────────────┬─────────────┘
                                     │ END
                                     ▼
                       ┌───────────────────────────┐
                       │ Return AIMessage to UI &  │
                       │ Real-time Telemetry Feed  │
                       │   via `logger.py`         │
                       └───────────────────────────┘

```

---

## 5. Implementation Details & Module Breakdown

The application is architected into decoupled modules for high maintainability, non-circular execution, and clear separation of concerns:

### 1. `app.py` (Presentation & UI Layer)

* Manages Streamlit page configuration, chat history session state (`st.session_state`), and input handling.
* Instantiates an `st.status()` context container that listens to status updates during graph execution and renders progress live to the user.

### 2. `agent_graph.py` (Core Router & Main Graph Orchestration)

* Defines the primary `StateGraph` containing `router_agent` and `guardrail` nodes.
* Initializes and caches the local **ChromaDB** vector store with **EmbeddingGemma**.
* Implements dynamic parameter extraction (`extract_site_parameters`) using `llama3.2` and regex parsing to extract countries and specific target locations from natural language prompts.

### 3. `site_selector.py` (Sub-Graph Multi-Agent Workflow)

* Implements a 3-node sub-graph (`supervisor` $\rightarrow$ `evaluator` $\rightarrow$ `synthesizer`):
1. **Supervisor**: Dispatches parallel search tasks to Serper.dev for each candidate market.
2. **Evaluator**: Prompt-engineers LLM domain experts to score Power, Zoning, and Financials.
3. **Synthesizer**: Constructs the final investment memo, sanitizes filename paths, and exports Markdown reports to `reports/`.

### 4. `guardrails.py` (Privacy & Security Layer)

* Contains the `sanitize_text()` function utilizing Microsoft Presidio / Regex patterns to strip PII before embedding storage and before final UI rendering.

### 5. `logger.py` (Execution Telemetry)

* Acts as a neutral callback bridge between the backend LangGraph workflows and Streamlit UI components, completely avoiding Python circular import issues.

---

How does this project documentation draft look for Tanishka's presentation? Let me know if you would like to expand on any specific section or add performance benchmarking details!

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
> 

How can I assist you today?

### Run Streamlit Application

1. open cmd or terminal
2. cd E:\Products\mango-lassy-chat
3. streamlit run app.py

NOTE: Make sure your `.env` file contains your `SERPER_API_KEY`.

3. URL to open from browser:
http://localhost:8501/

## Sample Screens

![alt text](image-1.png)

![alt text](image.png)

![alt text](image-2.png)

![alt text](image-3.png)

![alt text](image-4.png)

![alt text](image-5.png)

![alt text](image-6.png)

## Prerequisites

1. **Python 3.10+** installed on your machine (python-3.12.10).
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
venv/
.env
__pycache__/

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
mkdir reports

```

---

## Step 2: Install Python Dependencies

Setup environment for python. Navigate to your project root and install the required packages:

1. python -m venv venv
2. Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
3. .\venv\Scripts\activate


```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

```

---

## LangGraph Architecture Diagram

Here is the accurate, up-to-date architecture diagram reflecting the exact module breakdown, data flows, and routing logic implemented in your code (app.py, agent_graph.py, site_selector.py, guardrails.py, and logger.py):


                               ┌───────────────────────────┐
                               │   Streamlit UI (`app.py`) │
                               │  - Chat Input / History   │
                               │  - Live st.status() Feed  │
                               └─────────────┬─────────────┘
                                             │ Invokes app_graph.invoke()
                                             ▼
                               ┌───────────────────────────┐
                               │ Main LangGraph StateGraph │
                               │   (State: messages)       │
                               └─────────────┬─────────────┘
                                             │ START
                                             ▼
                               ┌───────────────────────────┐
                               │    `router_agent` Node    │
                               │  (Adaptive Intent Router) │
                               └─────────────┬─────────────┘
                                             │
      ┌──────────────────┬───────────────────┼───────────────────┐
      │                  │                   │                   │
      ▼ [SITE_SELECTION] ▼ [RAG]             ▼ [WEB]             ▼ [LLM]
┌──────────────────┐┌────────────────┐┌────────────────┐┌──────────────────┐
│ site_selector.py ││ ChromaDB +     ││ Google Serper  ││ Direct Ollama    │
│ StateGraph:      ││ EmbeddingGemma ││ API Wrapper    ││ LLM Inference    │
│                  ││ Vector Store   ││ (Live Search)  │└──────────────────┘
│ 1. supervisor    ││ (Guarded Chunks│└────────────────┘          
│    (Serper API)  ││  via Guardrails│                            
│ 2. evaluator     ││  Module)       │                            
│    (LLM Scoring) │└────────────────┘                            
│ 3. synthesizer   │                                              
│    (Markdown Memo│                                              
│     Export)      │                                              
└─────────┬────────┘                                              
          │                                                       
          └──────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
                       ┌───────────────────────────┐
                       │     `guardrail` Node      │
                       │   (Output PII Sanitizer   │
                       │    via `guardrails.py`)   │
                       └─────────────┬─────────────┘
                                     │ END
                                     ▼
                       ┌───────────────────────────┐
                       │ Return AIMessage to UI &  │
                       │ Real-time Telemetry Feed  │
                       │   via `logger.py`         │
                       └───────────────────────────┘
Key Components mapped to your code:
app.py: Handles the Streamlit session state, user chat inputs, and the live st.status() container updated via callback hooks.

agent_graph.py: Houses the core state graph, intent routing (route_intent), dynamic parameter extraction, ChromaDB document ingestion (with guardrail checks), and the final guardrail output sanitization node.

site_selector.py: A specialized sub-graph handling hierarchical multi-agent orchestration (supervisor, evaluator, and synthesizer nodes) for Commercial Real Estate and Data Center site analysis and .md file generation.

guardrails.py: Provides text sanitization to protect against PII during file ingestion and final response generation.

logger.py: Acts as a neutral utility to route live execution telemetry directly into the UI without circular imports.

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

### What is LangSmith?

**LangSmith** is an enterprise-grade observability, tracing, evaluation, and monitoring platform built by LangChain. It provides complete visibility into LLM-powered applications and multi-agent workflows.

Instead of treating an LLM execution as a black box where you only see the final response, LangSmith captures the **entire execution tree**—from the exact user prompt, routing decisions, and vector database retrievals to sub-agent reasoning loops, API payload calls, token usage, latency, and costs.

---

### Key Benefits of LangSmith for the Mango Lassi Chat Assistant

Integrating LangSmith into **Mango Lassi** solves several key operational challenges across your multi-agent architecture (`agent_graph.py`, `site_selector.py`, `guardrails.py`, and `app.py`):

#### 1. Deep Observability into LangGraph Multi-Agent Flows

* **The Problem:** In your site selection engine (`site_selector.py`), a single query triggers a supervisor node, Serper web search workers, sub-agent evaluations, and report generation. If a sub-agent generates an incomplete report, pinpointing which node failed can be difficult.
* **LangSmith Benefit:** Captures state transitions, inputs, and outputs node-by-node. You can inspect the exact payload sent to Serper.dev and see how each sub-agent scored zoning, power, or financial metrics.

#### 2. Fine-Grained Prompt & Parameter Extraction Debugging

* **The Problem:** The `extract_site_parameters()` function uses local Ollama calls (`llama3.2`) to extract target countries and locations into JSON formats. When parsing fails or takes too long, local console logs alone may not capture the raw text generated by Ollama.
* **LangSmith Benefit:** Logs raw model outputs, token generation speed, and temperature settings per call. You can immediately see if Ollama timed out or returned malformed JSON.

#### 3. RAG Retrieval & Vector Store Inspection

* **The Problem:** When queries trigger `[RAG]`, ChromaDB returns context chunks derived from local documents.
* **LangSmith Benefit:** Shows the exact top-$k$ document chunks retrieved from ChromaDB alongside similarity scores. This helps confirm whether context chunking in `agent_graph.py` provided sufficient data to answer questions correctly.

#### 4. Guardrails & PII Sanitization Verification

* **The Problem:** The app routes data through `guardrails.py` to sanitize PII prior to vector store upsert and UI rendering.
* **LangSmith Benefit:** You can compare raw text outputs against sanitized outputs directly within trace spans to ensure sensitive information (like emails or IDs) is properly redacted before reaching the UI.

#### 5. Cost, Latency & Token Usage Analytics

* **The Problem:** Running local LLMs (`llama3.2`) combined with external Serper API searches requires tracking overall pipeline latency.
* **LangSmith Benefit:** Provides dashboards tracking P50/P99 latency, token consumption, and error rates per execution path (`[SITE_SELECTION]`, `[RAG]`, `[WEB]`, `[LLM]`).

---

### Step-by-Step Instructions: Setting Up LangSmith in Mango Lassi

LangSmith integrates natively with LangChain and LangGraph. Enabling full workspace tracing requires no structural code rewrites—only zero-code environment variable setup.

#### Step 1: Create a Free LangSmith Account & Get an API Key

1. Go to [smith.langchain.com](https://smith.langchain.com) and sign up/log in.
2. Click on **Settings** (gear icon) in the bottom-left corner $\rightarrow$ **API Keys**.
3. Click **Create API Key**, copy the key (starts with `lsv2_pt_...`), and save it securely.

#### Step 2: Configure Environment Variables in `.env`

Open your project's `.env` file (`E:\Products\mango-lassy-chat\.env`) and add the following lines:

```env
# Enable LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_actual_langsmith_api_key_here"
LANGCHAIN_PROJECT="mango-lassi-assistant"

```

#### Step 3: Run the Assistant and View Live Traces

1. Start the Streamlit application:
```bash
streamlit run app.py

```
2. Send a query in the chat interface (for example: *"Is Malaysia a good investment location for buying a shopping complex?"*).
3. Open your browser to [smith.langchain.com](https://smith.langchain.com) and select the **`mango-lassi-assistant`** project.

#### Step 4: Inspect execution runs in the LangSmith UI

Inside the LangSmith dashboard, click on any execution trace to view:

* **The Root Node (`app_graph`)**: Displays overall latency and completion status.
* **`router_agent` Node**: Shows how `route_intent` evaluated the query and dispatched it.
* **`site_selection_graph` Sub-Graph**: Shows child spans for `supervisor`, `evaluator`, and `synthesizer` nodes, including raw Serper search results and model responses.
* **`guardrail` Node**: Displays input vs. sanitized output comparisons.

#### Step 5: Step-by-Step Guide to Enable Graph View

Step 1: Install the LangGraph CLI
Open your terminal inside your project directory and install the development CLI package:

Bash
pip install -U "langgraph-cli[inmem]"

Step 2: Add a langgraph.json Configuration File
Create a new file named langgraph.json in your project root (E:\Products\mango-lassy-chat\langgraph.json) to point LangSmith Studio to your compiled graph:

{
  "dependencies": ["."],
  "graphs": {
    "mango_lassi_agent": "./agent_graph.py:app_graph"
  },
  "env": ".env"
}

Step 3: Launch the Local Agent Server
Run the local development server in your terminal:

Bash Command:
langgraph dev

### Intelligent LLM Intent Classifier

def route_intent(query: str) -> str:
    """
    Uses LLM-based intent classification for flexible natural language understanding,
    with fast keyword heuristics as a fallback.
    """
    classification_prompt = f"""
    Analyze the user query below and classify its primary intent into EXACTLY ONE of the following categories:
    
    1. SITE_SELECTION: Queries asking about commercial real estate, buying/leasing property, shopping centers/malls, data centers, site feasibility, or evaluating cities/countries for real estate development/investment.
    2. WEB: Queries asking for real-time information, recent news, current prices, weather, stocks, or search online.
    3. RAG: Queries asking about local project documentation, system architecture, "Mango Lassi", "Tanishka", or roadmap details.
    4. LLM: General conversational queries, coding questions, math, or basic Q&A.

    User Query: "{query}"

    Return ONLY the category name (SITE_SELECTION, WEB, RAG, or LLM). No explanations or extra punctuation.
    """
    try:
        response = llm.invoke(classification_prompt)
        intent = response.content.strip().upper()
        for valid_intent in ["SITE_SELECTION", "WEB", "RAG", "LLM"]:
            if valid_intent in intent:
                return valid_intent
    except Exception as e:
        log_status(f"[ROUTER WARNING] LLM intent classification failed ({e}). Using keyword fallback.")

    # Fallback heuristic rules
    query_lower = query.lower()
    site_keywords = [
        "site selection", "data center", "commercial real estate", "investment location",
        "shopping complex", "shopping center", "shopping mall", "property investment",
        "real estate", "retail space", "warehouse", "buying property", "commercial property"
    ]
    if any(k in query_lower for k in site_keywords):
        return "SITE_SELECTION"
        
    web_keywords = ["latest", "news", "current", "weather", "today", "search", "price", "stock", "who is", "google", "online"]
    if any(k in query_lower for k in web_keywords):
        return "WEB"
        
    local_keywords = ["mango", "lassy", "lassi", "roadmap", "architecture", "tanishka"]
    if any(k in query_lower for k in local_keywords):
        return "RAG"
        
    return "LLM"