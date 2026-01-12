"""
AI Document Assistant - Streamlit UI
Modern RAG application with beautiful interface.
"""

import os
import sys
import time
import streamlit as st
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_engine import RAGEngine


# Page configuration
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    /* Main theme */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Chat container */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Upload button */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed rgba(255, 255, 255, 0.3);
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    /* Title */
    h1 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-weight: 800;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# Initialize RAG engine
@st.cache_resource
def get_rag_engine():
    """Initialize and cache RAG engine."""
    return RAGEngine()


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()


def main():
    """Main application."""
    
    # Get RAG engine
    rag = get_rag_engine()
    
    # Title
    st.title("📚 AI Document Assistant")
    st.markdown("*Upload your documents, ask questions - Powered by Llama 3.2*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # System stats
        stats = rag.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Documents", stats.get("total_documents", 0))
        with col2:
            health = "🟢" if stats.get("ollama_health", False) else "🔴"
            st.metric("Ollama", health)
        
        st.divider()
        
        # File upload
        st.header("📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Select PDF or TXT file",
            type=["pdf", "txt", "md"],
            help="Upload your document here"
        )
        
        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state.uploaded_files:
                with st.spinner("📖 Processing document..."):
                    # Save temporarily
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Ingest
                    result = rag.ingest_document(temp_path)
                    
                    # Clean up
                    os.remove(temp_path)
                    
                    # Update state
                    st.session_state.uploaded_files.add(uploaded_file.name)
                    
                    st.success(f"✅ {result['file_name']} uploaded successfully!")
                    st.info(f"📊 Split into {result['chunks_created']} chunks")
                    
                    # Rerun to update stats
                    st.rerun()
        
        st.divider()
        
        # Uploaded documents
        st.header("📁 Uploaded Documents")
        documents = rag.list_documents()
        
        if documents:
            for doc in documents:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {doc}")
                with col2:
                    if st.button("🗑️", key=f"delete_{doc}"):
                        rag.delete_document(doc)
                        st.session_state.uploaded_files.discard(doc)
                        st.rerun()
        else:
            st.info("No documents uploaded yet")
        
        st.divider()
        
        # Clear all
        if st.button("🗑️ Clear All", type="secondary"):
            rag.clear_all()
            st.session_state.uploaded_files.clear()
            st.session_state.messages.clear()
            st.rerun()
    
    # Main chat area
    st.divider()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"][:3]):
                        st.caption(f"**Source {i+1}:** {source['metadata'].get('source', 'Unknown')}")
                        st.text(source['content'][:200] + "...")
    
    # Chat input
    if prompt := st.chat_input("Ask questions about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            try:
                with st.spinner("🤔 Thinking..."):
                    response = rag.query(prompt, stream=False)
                    
                    st.markdown(response["answer"])
                    
                    # Show sources
                    if response["sources"]:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(response["sources"][:3]):
                                st.caption(f"**Source {i+1}:** {source['metadata'].get('source', 'Unknown')}")
                                st.text(source['content'][:200] + "...")
                
                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"]
                })
                
            except ConnectionError as e:
                error_msg = f"❌ **Ollama Connection Error:** {str(e)}\n\nCheck if Ollama is running: `ollama serve`"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })
            except Exception as e:
                error_msg = f"❌ **Error occurred:** {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })


if __name__ == "__main__":
    main()
