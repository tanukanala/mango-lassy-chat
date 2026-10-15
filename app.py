import streamlit as st
from dotenv import load_dotenv

# --- 0. Load Environment Variables First ---
load_dotenv()

from agent_graph import app_graph

st.set_page_config(
    page_title="MangoLassi RAG Assistant",
    page_icon="🥭",
    layout="centered"
)

st.header("✨ Mango🥭Lassi🥤RAG Agentic Chat ✨")
st.markdown("Powered by Python, Streamlit, LangGraph, ChromaDB, EmbeddingGemma, Serper.dev, & Multi-Agent Site Selection")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Mango Lassi assistant. Ask me about local docs, web search news, or request commercial real estate site selection reports!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question or request a site selection report..."):
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
        with st.spinner("Processing request through Multi-Agent Workflow..."):
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