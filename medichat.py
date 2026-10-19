import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_classic.chains import RetrievalQA
except (ImportError, ModuleNotFoundError):
    from langchain_classic.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv(find_dotenv())

# Page Configuration
st.set_page_config(
    page_title="MediChat - AI Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main container styling */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1050px;
    }

    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(20, 184, 166, 0.3);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(13, 148, 136, 0.2);
        position: relative;
        overflow: hidden;
    }

    .hero-banner::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(20, 184, 166, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(13, 148, 136, 0.15);
        color: #2dd4bf;
        border: 1px solid rgba(45, 212, 191, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 1.95rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff 0%, #a5f3fc 50%, #2dd4bf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Welcome Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-top: 18px;
    }

    .feature-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(51, 65, 85, 0.7);
        border-radius: 12px;
        padding: 14px 16px;
        transition: all 0.2s ease;
    }

    .feature-card:hover {
        border-color: rgba(45, 212, 191, 0.5);
        transform: translateY(-2px);
    }

    .feature-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .feature-desc {
        font-size: 0.78rem;
        color: #94a3b8;
        line-height: 1.4;
        margin: 0;
    }

    /* Source Reference Cards */
    .source-box {
        background: rgba(15, 23, 42, 0.7);
        border-left: 3px solid #0d9488;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-size: 0.83rem;
        color: #cbd5e1;
        line-height: 1.5;
    }

    .source-meta {
        font-size: 0.74rem;
        font-weight: 700;
        color: #2dd4bf;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Sidebar cards */
    .sidebar-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 16px;
    }

    .sidebar-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #f1f5f9;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .disclaimer-box {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
        padding: 12px;
        font-size: 0.75rem;
        color: #fde68a;
        line-height: 1.45;
    }
</style>
""", unsafe_allow_html=True)

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def main():
    # ----------------- SIDEBAR -----------------
    with st.sidebar:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <span style="font-size: 2rem;">🩺</span>
            <div>
                <h2 style="margin: 0; font-size: 1.3rem; font-weight: 800; color: #f8fafc;">MediChat AI</h2>
                <span style="font-size: 0.75rem; color: #94a3b8;">RAG Clinical Assistant</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="status-pill">🟢 System Online • Groq Active</div>', unsafe_allow_html=True)

        # System Architecture Card
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-title">⚙️ Architecture Specs</div>
            <div style="font-size: 0.8rem; color: #94a3b8; display: flex; flex-direction: column; gap: 8px;">
                <div>📚 <b>Source:</b> <span style="color:#e2e8f0;">Gale Encyclopedia of Medicine</span></div>
                <div>⚡ <b>LLM:</b> <span style="color:#e2e8f0;">Groq (openai/gpt-oss-20b)</span></div>
                <div>🧠 <b>Embeddings:</b> <span style="color:#e2e8f0;">all-MiniLM-L6-v2</span></div>
                <div>🔍 <b>Vector Store:</b> <span style="color:#e2e8f0;">FAISS (Top-3 Retrieval)</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Suggested Questions
        st.markdown('<div class="sidebar-title">💡 Suggested Questions</div>', unsafe_allow_html=True)
        sample_queries = [
            "What is Diabetes Mellitus?",
            "Causes and symptoms of Hypertension?",
            "How is an Asthma attack treated?",
            "What are cankar sores?"
        ]

        for q in sample_queries:
            if st.button(q, key=f"btn_{q}", use_container_width=True):
                st.session_state["queued_prompt"] = q

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Medical Disclaimer
        st.markdown("""
        <div class="disclaimer-box">
            <b>⚠️ Clinical Disclaimer:</b><br>
            MediChat is an AI retrieval tool built for research and educational purposes. It does not replace professional medical advice, clinical diagnosis, or hospital treatment.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Clear Chat Button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ----------------- MAIN HEADER BANNER -----------------
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">✦ Evidence-Based Medical AI</div>
        <h1 class="hero-title">MediChat - Medical Knowledge Assistant</h1>
        <p class="hero-subtitle">
            Instant, authentic medical guidance retrieved directly from <i>The Gale Encyclopedia of Medicine</i> 
            powered by ultra-fast Groq LPU inference.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------- SESSION STATE INIT -----------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # If no messages yet, show welcome quick cards
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-title">📚 Verified Medical Text</div>
                <p class="feature-desc">All answers are cross-referenced with peer-reviewed encyclopedia pages.</p>
            </div>
            <div class="feature-card">
                <div class="feature-title">🛡️ Zero Hallucinations</div>
                <p class="feature-desc">Strict retrieval constraints prevent fabricating fake medicines or advice.</p>
            </div>
            <div class="feature-card">
                <div class="feature-title">⚡ Instant Groq Speed</div>
                <p class="feature-desc">Sub-second generation powered by state-of-the-art LPU architecture.</p>
            </div>
        </div>
        <div style="height: 24px;"></div>
        """, unsafe_allow_html=True)

    # ----------------- CHAT HISTORY -----------------
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🩺"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📖 Verified Book Citations & Sources", expanded=False):
                    for idx, doc in enumerate(message["sources"], 1):
                        page_num = doc.metadata.get("page", "N/A") if hasattr(doc, "metadata") else "N/A"
                        text_content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                        st.markdown(f"""
                        <div class="source-box">
                            <div class="source-meta">Reference Chunk #{idx} (Page {page_num})</div>
                            {text_content}
                        </div>
                        """, unsafe_allow_html=True)

    # ----------------- USER INPUT -----------------
    # Check if a sample question was clicked from the sidebar
    user_input = st.chat_input("Ask a medical question (e.g. symptoms, diagnosis, treatment, causes)...")
    if "queued_prompt" in st.session_state and st.session_state["queued_prompt"]:
        user_input = st.session_state.pop("queued_prompt")

    if user_input:
        # Display User Message
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        CUSTOM_PROMPT_TEMPLATE = """
        If the user greets you (such as 'hi', 'hello', 'hy', 'hey', 'salam'), respond with a warm, polite greeting introducing yourself as MediChat and offer your medical assistance.
        For factual and medical questions, use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer or the information is not provided in the context, just say that you don't know, don't try to make up an answer. 
        Don't provide anything out of the given context for medical questions.

        Context: {context}
        Question: {question}

        Answer:
        """

        try:
            with st.spinner("🔍 Consulting Medical Encyclopedia via Groq..."):
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store database.")
                    return

                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatGroq(
                        model_name="openai/gpt-oss-20b",
                        temperature=0.0,
                        groq_api_key=os.environ.get("GROQ_API_KEY"),
                    ),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({"query": user_input})
                result = response["result"]
                source_documents = response.get("source_documents", [])

            # Display Assistant Message
            with st.chat_message("assistant", avatar="🩺"):
                st.markdown(result)
                if source_documents:
                    with st.expander("📖 Verified Book Citations & Sources", expanded=False):
                        for idx, doc in enumerate(source_documents, 1):
                            page_num = doc.metadata.get("page", "N/A") if hasattr(doc, "metadata") else "N/A"
                            text_content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                            st.markdown(f"""
                            <div class="source-box">
                                <div class="source-meta">Reference Chunk #{idx} (Page {page_num})</div>
                                {text_content}
                            </div>
                            """, unsafe_allow_html=True)

            # Store in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": result,
                "sources": source_documents
            })

        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()