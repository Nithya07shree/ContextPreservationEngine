import streamlit as st
import uuid
from generation.chat import DocumentationBot

st.set_page_config(
    page_title="Pensieve AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

def _is_dark() -> bool:
    try:
        bg = st.get_option("theme.backgroundColor") or ""
        if bg.startswith("#"):
            return int(bg[1:3], 16) < 128
        return True
    except Exception:
        return True

DARK = _is_dark()

if DARK:
    USER_BG  = "#1e1b4b"
    BOT_BG   = "#0f172a"
    TEXT     = "#e2e8f0"
    BOT_TEXT = "#cbd5e1"
    MUTED    = "#64748b"
    BORDER   = "#1e2130"
    BTN_BG   = "#1e2130"
    BTN_HOVER= "#2d3148"
    INPUT_BG = "#1e293b"
    ACCENT   = "#7c6af7"
else:
    USER_BG  = "#ede9fe"
    BOT_BG   = "#f8fafc"
    TEXT     = "#1e293b"
    BOT_TEXT = "#334155"
    MUTED    = "#94a3b8"
    BORDER   = "#e2e8f0"
    BTN_BG   = "#f1f5f9"
    BTN_HOVER= "#e2e8f0"
    INPUT_BG = "#ffffff"
    ACCENT   = "#7c6af7"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', sans-serif !important;
    color: {TEXT} !important;
}}

#MainMenu, footer {{ visibility: hidden; }}

.block-container {{
    padding: 1.5rem 2rem 2rem !important;
    max-width: 860px !important;
    margin: 0 auto;
}}

section[data-testid="stSidebar"] {{
    border-right: 1px solid {BORDER} !important;
}}

section[data-testid="stSidebar"] * {{
    color: {TEXT} !important;
}}

section[data-testid="stSidebar"] .stButton > button {{
    background: {BTN_BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {MUTED} !important;
    font-size: 0.78rem !important;
    text-align: left !important;
    padding: 8px 12px !important;
    width: 100% !important;
}}

section[data-testid="stSidebar"] .stButton > button:hover {{
    background: {BTN_HOVER} !important;
    color: {TEXT} !important;
}}

.stSelectbox > div > div {{
    background: {INPUT_BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}

.stChatInput > div {{
    background: {INPUT_BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
}}

.stChatInput textarea {{
    color: {TEXT} !important;
}}

.stChatInput textarea::placeholder {{
    color: {MUTED} !important;
}}

.stChatInput button {{
    background: {ACCENT} !important;
    border-radius: 8px !important;
}}

@keyframes blink {{
    0%, 80%, 100% {{ opacity: 0.2; transform: scale(0.8); }}
    40% {{ opacity: 1; transform: scale(1); }}
}}

.dot {{
    display: inline-block;
    width: 7px; height: 7px;
    background: {MUTED};
    border-radius: 50%;
    margin: 0 2px;
    animation: blink 1.3s ease-in-out infinite;
}}

.dot:nth-child(2) {{ animation-delay: 0.2s; }}
.dot:nth-child(3) {{ animation-delay: 0.4s; }}
</style>
""", unsafe_allow_html=True)


if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_chat" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.current_chat = cid
    st.session_state.conversations[cid] = {"title": "New Chat", "messages": []}

if "role" not in st.session_state:
    st.session_state.role = "onboarding"

if "bot" not in st.session_state:
    st.session_state.bot = DocumentationBot(user_role=st.session_state.role)

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


with st.sidebar:

    st.markdown("## 🧠 Pensieve AI")
    st.markdown("---")

    roles = ["onboarding", "junior", "senior", "admin"]

    role = st.selectbox(
        "Role",
        roles,
        index=roles.index(st.session_state.role)
    )

    if role != st.session_state.role:
        st.session_state.role = role
        st.session_state.bot.user_role = role

    st.markdown("---")

    if st.button("+ New Chat"):
        cid = str(uuid.uuid4())
        st.session_state.conversations[cid] = {"title": "New Chat", "messages": []}
        st.session_state.current_chat = cid
        st.session_state.pending_query = None
        st.rerun()

    st.markdown("### Chat History")

    for cid, chat in st.session_state.conversations.items():
        if st.button(chat["title"], key=cid):
            st.session_state.current_chat = cid
            st.session_state.pending_query = None
            st.rerun()


ROLE_COLORS = {
    "onboarding": "#56d364",
    "junior": "#58a6ff",
    "senior": "#bc8cff",
    "admin": "#ff7b72",
}

rc = ROLE_COLORS.get(st.session_state.role, ACCENT)

st.markdown(
    f"<h2 style='margin:0;'>Pensieve AI</h2>"
    f"<p style='margin:0 0 24px;font-size:0.78rem;color:{MUTED};'>"
    f"Role: <b style='color:{rc};'>{st.session_state.role}</b>",
    unsafe_allow_html=True
)

chat = st.session_state.conversations[st.session_state.current_chat]


def _render_sources(sources):

    if not sources:
        return

    st.markdown(
        f"<div style='font-size:0.65rem;color:{MUTED};"
        f"text-transform:uppercase;margin-top:12px;margin-bottom:6px;'>Sources</div>",
        unsafe_allow_html=True
    )

    for i, c in enumerate(sources):

        file_name = c.get("file_name") or c.get("file_path") or "unknown_file"
        func      = c.get("function_name") or ""
        start     = c.get("start_line")
        end       = c.get("end_line")

        code = (
            c.get("content")
            or c.get("code")
            or c.get("chunk")
            or ""
        )

        header = file_name

        if func:
            header += f" — {func}"

        if start and end:
            header += f" (L{start}-{end})"

        with st.expander(header, expanded=False):

            if code:

                st.code(
                    code,
                    language="python"
                )

            else:
                st.markdown(
                    f"<span style='color:{MUTED};font-size:0.8rem;'>"
                    f"No snippet available for this chunk."
                    f"</span>",
                    unsafe_allow_html=True
                )

for msg in chat["messages"]:

    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])

    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            _render_sources(msg.get("sources", []))


if st.session_state.pending_query:

    query = st.session_state.pending_query

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        slot = st.empty()

        slot.markdown(
            '<span class="dot"></span>'
            '<span class="dot"></span>'
            '<span class="dot"></span>',
            unsafe_allow_html=True
        )

        bot = st.session_state.bot

        response, sources = bot.ask(query)

        slot.markdown(response)

        _render_sources(sources)

    chat["messages"].append({
        "role": "user",
        "content": query,
        "sources": []
    })

    chat["messages"].append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })

    if chat["title"] == "New Chat":
        chat["title"] = query[:30]

    st.session_state.pending_query = None

    st.rerun()


query = st.chat_input("Ask about the codebase...")

if query:
    st.session_state.pending_query = query
    st.rerun()