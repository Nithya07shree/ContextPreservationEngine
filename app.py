import streamlit as st
from generation.chat import DocumentationBot
import uuid

st.set_page_config(page_title="Pensieve AI", layout="wide")

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_chat" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat = chat_id
    st.session_state.conversations[chat_id] = {
        "title": "New Chat",
        "messages": []
    }

if "role" not in st.session_state:
    st.session_state.role = "onboarding"

if "bot" not in st.session_state:
    st.session_state.bot = DocumentationBot(user_role=st.session_state.role)

with st.sidebar:

    st.title("PENSIEVE AI")

    st.markdown("---")

    # Role selector
    role = st.selectbox(
        "Role",
        ["onboarding", "junior", "senior", "admin"],
        index=["onboarding","junior","senior","admin"].index(st.session_state.role)
    )

    if role != st.session_state.role:
        st.session_state.role = role
        st.session_state.bot.user_role = role

    st.markdown("---")

    # New Chat Button
    if st.button("+ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.conversations[new_id] = {
            "title": "New Chat",
            "messages": []
        }
        st.session_state.current_chat = new_id
        st.rerun()

    st.markdown("### Chat History")

    # Show chat history
    for chat_id, chat in st.session_state.conversations.items():

        if st.button(chat["title"], key=chat_id):
            st.session_state.current_chat = chat_id
            st.rerun()

st.markdown(
    """
    <div style='display:flex;align-items:center;padding:10px 0'>
        <h2 style='margin:0'>Pensieve AI</h2>
    </div>
    """,
    unsafe_allow_html=True
)


chat = st.session_state.conversations[st.session_state.current_chat]

for msg in chat["messages"]:

    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style="
            background:#050414;
            padding:10px;
            border-radius:10px;
            margin:5px 0;
            text-align:right">
            {msg["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div style="
            background:#050414;
            padding:10px;
            border-radius:10px;
            margin:5px 0">
            {msg["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

query = st.chat_input("Ask about the codebase...")

if query:
    
    chat["messages"].append({
        "role": "user",
        "content": query
    })

    bot = st.session_state.bot

    response = bot.ask(query)

    chat["messages"].append({
        "role": "assistant",
        "content": response
    })

    if chat["title"] == "New Chat":
        chat["title"] = query[:30]

    st.rerun()