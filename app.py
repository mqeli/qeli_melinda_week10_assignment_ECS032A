from datetime import datetime
import requests
import streamlit as st


st.set_page_config(page_title="My AI Chat", layout="wide")
st.title("My AI Chat")

token = st.secrets.get("HF_TOKEN", "").strip()
api_url = "https://router.huggingface.co/v1/chat/completions"

if "chats" not in st.session_state:
    st.session_state.chats = []
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "next_chat_id" not in st.session_state:
    st.session_state.next_chat_id = 1


def create_chat() -> int:
    chat_id = st.session_state.next_chat_id
    st.session_state.next_chat_id += 1
    st.session_state.chats.append(
        {
            "id": chat_id,
            "title": "New Chat",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "messages": [],
        }
    )
    return chat_id


def get_active_chat() -> dict | None:
    for chat in st.session_state.chats:
        if chat["id"] == st.session_state.active_chat_id:
            return chat
    return None

with st.sidebar:
    st.subheader("Chats")
    if st.button("New Chat"):
        st.session_state.active_chat_id = create_chat()
        st.rerun()

    chat_list = st.container(height=350, border=True)
    with chat_list:
        for chat in st.session_state.chats:
            is_active = chat["id"] == st.session_state.active_chat_id
            label_prefix = "▶ " if is_active else ""
            with st.container():
                cols = st.columns([8, 2])
                if cols[0].button(
                    f"{label_prefix}{chat['title']} ({chat['created_at']})",
                    key=f"chat_{chat['id']}",
                ):
                    st.session_state.active_chat_id = chat["id"]
                    st.rerun()
                if cols[1].button("✕", key=f"del_{chat['id']}"):
                    st.session_state.chats = [
                        c for c in st.session_state.chats if c["id"] != chat["id"]
                    ]
                    if st.session_state.active_chat_id == chat["id"]:
                        st.session_state.active_chat_id = (
                            st.session_state.chats[0]["id"]
                            if st.session_state.chats
                            else None
                        )
                    st.rerun()

if not token:
    st.error("Missing Hugging Face token. Please set HF_TOKEN in .streamlit/secrets.toml.")
else:
    active_chat = get_active_chat()
    if active_chat is None:
        st.info("No active chat. Create a new chat from the sidebar.")
    else:
        for message in active_chat["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_input = st.chat_input("Type your message...")
        if user_input:
            active_chat["messages"].append({"role": "user", "content": user_input})
            if active_chat["title"] == "New Chat":
                active_chat["title"] = user_input[:40]

            with st.chat_message("user"):
                st.write(user_input)

            payload = {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "messages": active_chat["messages"],
                "max_tokens": 200,
            }

            try:
                response = requests.post(
                    api_url,
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                    timeout=60,
                )
                if response.status_code == 200:
                    data = response.json()
                    assistant_reply = data["choices"][0]["message"]["content"].strip()
                    active_chat["messages"].append(
                        {"role": "assistant", "content": assistant_reply}
                    )
                    with st.chat_message("assistant"):
                        st.write(assistant_reply)
                else:
                    st.error(f"API error {response.status_code}: {response.text}")
            except requests.RequestException as exc:
                st.error(f"Network error: {exc}")
