import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent_graph import app_graph, set_status_callback

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
        with st.status("Initializing agentic workflow...", expanded=True) as status:
            def update_status(msg: str):
                status.update(label=msg)
                st.write(f"• {msg}")

            set_status_callback(update_status)

            try:
                result = app_graph.invoke({"messages": graph_messages})
                ai_response = result["messages"][-1].content
                
                status.update(label="Workflow completed successfully!", state="complete", expanded=False)
            except Exception as e:
                ai_response = f"Error processing request: {str(e)}"
                status.update(label="Workflow failed!", state="error", expanded=True)

        st.markdown(ai_response)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})