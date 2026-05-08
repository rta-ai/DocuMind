import os
import fitz
import streamlit as st
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Page config ──
st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .answer-box {
        background: #f0f4ff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #667eea;
        margin-top: 1rem;
    }
    .source-badge {
        background: #667eea;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin-right: 5px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="main-header">
    <h1>🧠 DocuMind</h1>
    <p>Your intelligent PDF assistant — powered by Groq + LLaMA 3.3</p>
</div>
""", unsafe_allow_html=True)

# ── Session state init ──
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# ── Load models (cached) ──
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_chroma():
    return chromadb.Client()

@st.cache_resource
def load_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

embedder      = load_embedder()
chroma_client = load_chroma()
groq_client   = load_groq_client()

# ── Sidebar ──
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("DocuMind")
    st.caption("RAG-powered PDF Intelligence")
    st.divider()

    # API key fallback input
    # (only shown if GROQ_API_KEY not set as HF secret)
    if not os.environ.get("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY not found in environment.")
        api_key_input = st.text_input(
            "Enter Groq API Key manually",
            type="password",
            placeholder="gsk_..."
        )
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
            groq_client = Groq(api_key=api_key_input)
            st.success("API key set!")
    else:
        st.success("Groq API key loaded ✓")

    st.divider()

    # PDF Upload
    st.subheader("📂 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Max 200MB"
    )

    st.divider()

    # Document stats
    if st.session_state.filename:
        st.subheader("📊 Document Info")
        st.markdown(f"""
        <div class="metric-card">
            📄 <b>File:</b> {st.session_state.filename}<br>
            📑 <b>Pages:</b> {st.session_state.total_pages}<br>
            🧩 <b>Chunks:</b> {st.session_state.total_chunks}<br>
            🤖 <b>Model:</b> llama-3.3-70b
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear & Upload New"):
            st.session_state.chat_history = []
            st.session_state.collection   = None
            st.session_state.filename     = None
            st.session_state.total_pages  = 0
            st.session_state.total_chunks = 0
            st.rerun()

    st.divider()
    st.caption("Built with Streamlit · Groq · ChromaDB · LangChain")

# ── PDF Processing ──
def process_pdf(uploaded_file):
    with st.spinner("📖 Reading PDF..."):
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        doc        = fitz.open(temp_path)
        pages_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                pages_text.append({
                    "page": page_num + 1,
                    "text": text
                })
        doc.close()

        if not pages_text:
            st.error("No text found. This may be a scanned PDF.")
            return False

    with st.spinner("✂️ Chunking text..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )
        all_chunks = []
        for page in pages_text:
            chunks = splitter.split_text(page["text"])
            for chunk in chunks:
                all_chunks.append({
                    "text" : chunk,
                    "page" : page["page"],
                    "source": uploaded_file.name
                })

    with st.spinner(f"🧠 Embedding {len(all_chunks)} chunks..."):
        collection_name = uploaded_file.name.replace(".pdf","").replace(" ","_")[:50]
        collection      = chroma_client.get_or_create_collection(name=collection_name)

        if collection.count() == 0:
            texts      = [c["text"] for c in all_chunks]
            metadatas  = [{"page": c["page"], "source": c["source"]} for c in all_chunks]
            ids        = [f"chunk_{i}" for i in range(len(all_chunks))]
            embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

        st.session_state.collection   = collection
        st.session_state.filename     = uploaded_file.name
        st.session_state.total_pages  = len(pages_text)
        st.session_state.total_chunks = collection.count()

    return True

# ── Trigger processing ──
if uploaded_file and uploaded_file.name != st.session_state.filename:
    success = process_pdf(uploaded_file)
    if success:
        st.success(f"✅ DocuMind has read **{uploaded_file.name}** — ready!")
        st.balloons()

# ── Main area ──
if not st.session_state.collection:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\nUpload any PDF using the sidebar")
    with col2:
        st.info("**Step 2**\n\nChoose a mode: Q&A, Summarize or Extract")
    with col3:
        st.info("**Step 3**\n\nGet instant AI-powered answers")

else:
    mode = st.radio(
        "Choose mode",
        ["💬 Q&A", "📋 Summarize", "🔍 Key Extract"],
        horizontal=True
    )
    st.divider()

    # ── Q&A MODE ──
    if mode == "💬 Q&A":
        st.subheader("Ask anything about your document")

        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])
                pages = chat["pages"]
                st.markdown(
                    " ".join([f'<span class="source-badge">Page {p}</span>' for p in pages]),
                    unsafe_allow_html=True
                )

        question = st.chat_input("Ask a question about your PDF...")

        if question:
            if not groq_client:
                st.error("Please set your Groq API key in the sidebar.")
            else:
                with st.chat_message("user"):
                    st.write(question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        q_embedding = embedder.encode(question).tolist()
                        results     = st.session_state.collection.query(
                            query_embeddings=[q_embedding],
                            n_results=5
                        )
                        chunks    = results["documents"][0]
                        metadatas = results["metadatas"][0]

                        context = ""
                        for chunk, meta in zip(chunks, metadatas):
                            context += f"[Page {meta['page']}]\n{chunk}\n\n"

                        prompt = f"""You are DocuMind, an intelligent PDF assistant.
Answer the question based ONLY on the context provided.
If the answer is not in the context say: "I couldn't find this in the document."
Be clear, concise and helpful.

Context:
{context}

Question: {question}

Answer:"""

                        stream = groq_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            max_tokens=1024,
                            stream=True
                        )

                        answer = st.write_stream(
                            (chunk.choices[0].delta.content or "" for chunk in stream)
                        )

                        pages_used = sorted(set([m["page"] for m in metadatas]))
                        st.markdown(
                            "**Sources:** " + " ".join([f'<span class="source-badge">Page {p}</span>' for p in pages_used]),
                            unsafe_allow_html=True
                        )

                st.session_state.chat_history.append({
                    "question": question,
                    "answer"  : answer,
                    "pages"   : pages_used
                })

    # ── SUMMARIZE MODE ──
    elif mode == "📋 Summarize":
        st.subheader("Document summarization")

        sum_mode = st.radio(
            "Summary type",
            ["Short (5 bullet points)", "Detailed"],
            horizontal=True
        )

        if st.button("Generate Summary"):
            if not groq_client:
                st.error("Please set your Groq API key in the sidebar.")
            else:
                with st.spinner("Summarizing..."):
                    all_stored   = st.session_state.collection.get()
                    all_texts    = all_stored["documents"]
                    all_meta     = all_stored["metadatas"]
                    paired       = sorted(zip(all_meta, all_texts), key=lambda x: x[0]["page"])
                    sorted_texts = [text for _, text in paired]
                    full_text    = "\n\n".join(sorted_texts)[:12000]

                    if sum_mode == "Short (5 bullet points)":
                        prompt = f"""You are DocuMind. Summarize this document in exactly 5 clear bullet points.
Each bullet should capture one major idea.

Document:
{full_text}

5-Point Summary:"""
                    else:
                        prompt = f"""You are DocuMind. Write a detailed structured summary including:
main topic, key points, important findings, and conclusions.

Document:
{full_text}

Detailed Summary:"""

                    stream = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=1024,
                        stream=True
                    )

                    st.markdown("### Summary")
                    st.write_stream(
                        (chunk.choices[0].delta.content or "" for chunk in stream)
                    )

    # ── KEY EXTRACT MODE ──
    elif mode == "🔍 Key Extract":
        st.subheader("Extract key information")

        extract_type = st.selectbox(
            "What to extract",
            ["All", "Dates & Timeline", "People & Organizations",
             "Numbers & Figures", "Technical Terms"]
        )

        if st.button("Extract Now"):
            if not groq_client:
                st.error("Please set your Groq API key in the sidebar.")
            else:
                with st.spinner("Extracting..."):
                    all_stored   = st.session_state.collection.get()
                    all_texts    = all_stored["documents"]
                    all_meta     = all_stored["metadatas"]
                    paired       = sorted(zip(all_meta, all_texts), key=lambda x: x[0]["page"])
                    sorted_texts = [text for _, text in paired]
                    full_text    = "\n\n".join(sorted_texts)[:12000]

                    prompts = {
                        "All"                     : f"Extract and organize: 1) Important Dates 2) People & Organizations 3) Key Numbers 4) Technical Terms 5) Action Items\n\nDocument:\n{full_text}\n\nStructured Extraction:",
                        "Dates & Timeline"        : f"Extract ALL dates and time references as a numbered list with context.\n\nDocument:\n{full_text}\n\nDates:",
                        "People & Organizations"  : f"Extract ALL people and organization names as two separate lists.\n\nDocument:\n{full_text}\n\nNames:",
                        "Numbers & Figures"       : f"Extract ALL numbers, stats and figures as a numbered list with context.\n\nDocument:\n{full_text}\n\nFigures:",
                        "Technical Terms"         : f"Extract ALL technical terms with brief definitions from the document.\n\nDocument:\n{full_text}\n\nTerms:"
                    }

                    stream = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompts[extract_type]}],
                        temperature=0.1,
                        max_tokens=1024,
                        stream=True
                    )

                    st.markdown(f"### {extract_type}")
                    st.write_stream(
                        (chunk.choices[0].delta.content or "" for chunk in stream)
                    )