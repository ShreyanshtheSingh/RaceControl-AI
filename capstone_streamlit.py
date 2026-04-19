"""
RaceControl AI — Streamlit Web Interface
=========================================
Run with: streamlit run capstone_streamlit.py
"""

import streamlit as st
import uuid

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RaceControl AI — F1 Regulations Assistant",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── F1 Theme CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Rajdhani:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d0d0d;
        color: #f0f0f0;
    }

    /* ── Main container ── */
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 900px;
    }

    /* ── App header ── */
    .rc-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 18px 24px;
        background: linear-gradient(135deg, #1a0000 0%, #2a0000 50%, #1a0000 100%);
        border-bottom: 3px solid #e8002d;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(232,0,45,0.25);
    }
    .rc-header h1 {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: 0.04em;
    }
    .rc-header .subtitle {
        font-size: 0.85rem;
        color: #aaaaaa;
        margin-top: 2px;
        font-weight: 300;
    }
    .rc-accent { color: #e8002d; }

    /* ── Chat messages ── */
    .chat-user {
        background: #1c1c1c;
        border-left: 4px solid #e8002d;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.97rem;
    }
    .chat-assistant {
        background: #141414;
        border-left: 4px solid #444;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.97rem;
    }

    /* ── Metadata badge ── */
    .meta-badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 99px;
        margin-right: 6px;
        margin-top: 8px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-retrieve  { background: #1a3a1a; color: #4ade80; border: 1px solid #4ade80; }
    .badge-tool      { background: #1a1a3a; color: #60a5fa; border: 1px solid #60a5fa; }
    .badge-memory    { background: #3a3a1a; color: #fbbf24; border: 1px solid #fbbf24; }
    .badge-faith     { background: #2a0a0a; color: #f87171; border: 1px solid #f87171; }
    .badge-faith-ok  { background: #0a2a0a; color: #4ade80; border: 1px solid #4ade80; }

    /* ── Input box ── */
    .stTextInput > div > div > input {
        background-color: #1a1a1a !important;
        border: 1.5px solid #333 !important;
        border-radius: 8px !important;
        color: #f0f0f0 !important;
        font-size: 0.97rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #e8002d !important;
        box-shadow: 0 0 0 2px rgba(232,0,45,0.2) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #e8002d !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.4rem 1.2rem !important;
        transition: background 0.2s !important;
    }
    .stButton > button:hover {
        background: #b0001f !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #222 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-family: 'Rajdhani', sans-serif;
        color: #e8002d;
        font-size: 1.1rem;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff;
        font-size: 0.9rem;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #aaaaaa;
        font-size: 0.82rem;
    }

    /* ── Dividers ── */
    hr { border-color: #222 !important; }

    /* ── Spinner / info ── */
    .stSpinner > div { color: #e8002d !important; }

    /* ── Welcome card ── */
    .welcome-card {
        background: #141414;
        border: 1px solid #222;
        border-radius: 12px;
        padding: 24px;
        margin: 1rem 0 1.5rem 0;
        text-align: center;
    }
    .welcome-card h2 {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.4rem;
        color: #ffffff;
        margin-bottom: 8px;
    }
    .welcome-card p { color: #888; font-size: 0.9rem; margin-bottom: 0; }

    /* ── Example pills ── */
    .example-grid {
        display: flex; flex-wrap: wrap; gap: 8px; justify-content: center;
        margin-top: 16px;
    }
    .example-pill {
        background: #1c1c1c;
        border: 1px solid #333;
        border-radius: 99px;
        padding: 5px 14px;
        font-size: 0.78rem;
        color: #cccccc;
        cursor: default;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached startup ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RaceControl AI… (first time takes ~60 seconds)")
def load_agent():
    from agent import initialise
    app, embedder, collection, documents = initialise()
    topics = sorted({d["topic"] for d in documents})
    return app, embedder, collection, topics


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages  = []       # display history
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "meta" not in st.session_state:
    st.session_state.meta      = []       # per-turn route & faithfulness


# ── Load agent ─────────────────────────────────────────────────────────────────
app, embedder, collection, topics = load_agent()

from agent import ask  # noqa: E402 (after load)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏎️ RaceControl AI")
    st.markdown("""
**Domain:** 2026 FIA F1 Sporting Regulations  
**LLM:** Llama 3.3 70B via Groq  
**KB:** 83 regulation documents  
**Tool:** Jolpica F1 API + DuckDuckGo  
""")
    st.markdown("---")
    st.markdown("### 📚 Knowledge Base Topics")
    for t in topics[:30]:
        st.markdown(f"- {t}")
    if len(topics) > 30:
        st.markdown(f"*…and {len(topics)-30} more*")
    st.markdown("---")
    st.markdown("### 🔑 Session")
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}…`")
    if st.button("🔄 New Conversation"):
        st.session_state.messages  = []
        st.session_state.meta      = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "RaceControl AI uses retrieval-augmented generation (RAG) and "
        "conversation memory to answer questions about the 2026 FIA F1 "
        "Sporting Regulations. For live race data, it queries the Jolpica "
        "F1 API in real time."
    )


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="rc-header">
  <div style="font-size:2.5rem">🏁</div>
  <div>
    <h1>RaceControl <span class="rc-accent">AI</span></h1>
    <div class="subtitle">Intelligent F1 Regulations Assistant · 2026 FIA Sporting Regulations</div>
  </div>
</div>
""", unsafe_allow_html=True)


def _route_badge(route: str) -> str:
    mapping = {
        "retrieve":    ("badge-retrieve",  "📖 KB Lookup"),
        "tool":        ("badge-tool",      "🌐 Live Data"),
        "memory_only": ("badge-memory",    "💬 Memory"),
    }
    cls, label = mapping.get(route, ("badge-retrieve", route))
    return f'<span class="meta-badge {cls}">{label}</span>'


def _faith_badge(faith: float) -> str:
    if faith >= 0.7:
        return f'<span class="meta-badge badge-faith-ok">✅ Faithfulness {faith:.2f}</span>'
    return f'<span class="meta-badge badge-faith">⚠️ Faithfulness {faith:.2f}</span>'


# ── Welcome card (shown when no messages) ─────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
<div class="welcome-card">
  <h2>Welcome to RaceControl AI 🏁</h2>
  <p>Ask me anything about the 2026 FIA F1 Sporting Regulations, race procedures, penalties, tyres, and more.</p>
  <div class="example-grid">
    <span class="example-pill">What is the pit lane speed limit?</span>
    <span class="example-pill">How many PU elements per season?</span>
    <span class="example-pill">What happens in parc fermé?</span>
    <span class="example-pill">Who leads the Drivers Championship?</span>
    <span class="example-pill">What is a Drive-Through penalty?</span>
    <span class="example-pill">When is a Safety Car deployed?</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Chat history ───────────────────────────────────────────────────────────────
chat_placeholder = st.container()
with chat_placeholder:
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user"><strong>You</strong><br/>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            meta_html = ""
            if i // 2 < len(st.session_state.meta):
                m = st.session_state.meta[i // 2]
                meta_html = _route_badge(m.get("route", "")) + _faith_badge(m.get("faith", 1.0))
            st.markdown(
                f'<div class="chat-assistant"><strong>🏁 RaceControl AI</strong><br/>'
                f'{msg["content"]}'
                f'<br/>{meta_html}</div>',
                unsafe_allow_html=True,
            )


# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
col_input, col_btn = st.columns([8, 1])
with col_input:
    user_input = st.text_input(
        label="Your question",
        placeholder="e.g. What is the pit lane speed limit? / Who leads the championship?",
        label_visibility="collapsed",
        key="user_input",
    )
with col_btn:
    send = st.button("Send")

if send and user_input.strip():
    question = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Consulting the regulations…"):
        result = ask(app, question, thread_id=st.session_state.thread_id)

    answer = result.get("answer", "I'm sorry, I couldn't generate a response.")
    route  = result.get("route", "retrieve")
    faith  = result.get("faithfulness", 1.0)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.meta.append({"route": route, "faith": faith})
    st.rerun()
