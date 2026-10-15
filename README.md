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

## Sample Screens

![alt text](image-1.png)

![alt text](image.png)

![alt text](image-2.png)

![alt text](image-3.png)

![alt text](image-4.png)

![alt text](image-5.png)