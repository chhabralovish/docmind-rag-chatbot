import streamlit as st
from rag_pipeline import RAGPipeline
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — Chat with your Documents",
    page_icon="🧠",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 DocMind")
st.caption("Upload any PDF and chat with it using AI. Powered by RAG + LLMs.")
st.divider()

# ── Sidebar — LLM & API Config ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    llm_provider = st.selectbox(
        "Choose LLM Provider",
        ["OpenAI (GPT-4)", "Google Gemini", "Groq (LLaMA3)"]
    )

    api_key = st.text_input(
        "Enter API Key",
        type="password",
        placeholder="Paste your API key here"
    )

    st.divider()
    st.markdown("**Supported Providers:**")
    st.markdown("- 🟢 [OpenAI](https://platform.openai.com/)")
    st.markdown("- 🔵 [Google Gemini](https://aistudio.google.com/)")
    st.markdown("- 🟡 [Groq](https://console.groq.com/) ← Free & Fast")
    st.divider()
    st.markdown("Built by [Lovish Chhabra](https://www.linkedin.com/in/lovish-chhabra/)")

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📄 Upload your PDF document",
    type=["pdf"],
    help="Upload any PDF — research paper, report, manual, contract, etc."
)

if uploaded_file and api_key:
    with st.spinner("🔍 Reading and indexing your document..."):
        try:
            rag = RAGPipeline(
                pdf_file=uploaded_file,
                llm_provider=llm_provider,
                api_key=api_key
            )
            st.success(f"✅ Document indexed! **{uploaded_file.name}** is ready to chat.")
            st.session_state["rag"] = rag
            st.session_state["messages"] = []
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")

elif uploaded_file and not api_key:
    st.warning("⚠️ Please enter your API key in the sidebar to continue.")

# ── Chat Interface ─────────────────────────────────────────────────────────────
if "rag" in st.session_state:
    st.subheader("💬 Chat with your Document")

    # Display chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Ask anything about your document..."):
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state["rag"].query(prompt)
                    st.markdown(response)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": response
                    })
                except Exception as e:
                    err = f"❌ Error generating response: {str(e)}"
                    st.error(err)

    # Clear chat button
    if st.session_state["messages"]:
        if st.button("🗑️ Clear Chat"):
            st.session_state["messages"] = []
            st.rerun()

else:
    # Placeholder when no doc uploaded
    st.info("👆 Upload a PDF and enter your API key to get started.")
    with st.expander("💡 Example questions you can ask"):
        st.markdown("""
        - *"Summarise this document in 5 bullet points"*
        - *"What are the key findings?"*
        - *"Explain the methodology used"*
        - *"What does the document say about [topic]?"*
        - *"List all action items mentioned"*
        """)
